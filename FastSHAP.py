import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ErrorLib import MaskLayer1d, MaskLayer2d, KLDivLoss, DatasetRepeat, ShapleySampler
import numpy as np
from tqdm.auto import tqdm
import torch.optim as optim
from copy import deepcopy


#RETE TRAIN SEPARATI

def additive_efficient_normalization(pred, grand, null):
    gap = (grand - null) - torch.sum(pred, dim=1)
    # gap = gap.detach()
    return pred + gap.unsqueeze(1) / pred.shape[1]


def evaluate_explainer(explainer, normalization, x, grand, null, num_players, inference=False):
    # Evaluate explainer.
    pred = explainer(x)

    # Reshape SHAP values.
    if len(pred.shape) == 4:
        # Image.
        image_shape = pred.shape
        pred = pred.reshape(len(x), -1, num_players)
        pred = pred.permute(0, 2, 1)
    else:
        # Tabular.
        image_shape = None
        pred = pred.reshape(len(x), num_players, -1)

    # For pre-normalization efficiency gap.
    total = pred.sum(dim=1)

    # Apply normalization.
    if normalization:
        pred = normalization(pred, grand, null) #SERVE PER GARANTIRE LE STIME VICINE AGLI SV - EFFICIENCY CONSTRAINT

    # Reshape for inference.
    if inference:
        if image_shape is not None:
            pred = pred.permute(0, 2, 1)
            pred = pred.reshape(image_shape)

        return pred

    return pred, total


def calculate_grand_coalition(dataset, imputer, batch_size, link, device, num_workers,debug, VECTOR):
    if debug:
        print("CALCULATE_GRAND_COALITION")
    ones = torch.ones(batch_size, imputer.num_players, dtype=torch.float32, device=device) #32x12
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    with torch.no_grad():
        grand = []
        for x in loader:
            x=x[0]
            if debug:
                print("INPUT:",len(x),x)
            if VECTOR==1:
                tmp1, _=imputer(x.to(device), ones[:len(x)])
                tmp2=link(tmp1)
            else:
                _, tmp1=imputer(x.to(device), ones[:len(x)])
                tmp2=link(tmp1)
            if debug:
                print("OUTPUT SURROGATE",tmp1)
                print("GRAND(SOFTMAX):",tmp2)
            grand.append(tmp2) #SOFTMAX(SURROGATE(DATA+ONES))

        # Concatenate and return.
        grand = torch.cat(grand)
        if len(grand.shape) == 1:
            grand = grand.unsqueeze(1)

    return grand


def generate_validation_data(val_set, imputer, validation_samples, sampler, batch_size, link, device, num_workers, VECTOR):
    # Generate coalitions.
    print("")
    val_S = sampler.sample(validation_samples * len(val_set), paired_sampling=True)\
                .reshape(len(val_set), validation_samples, imputer.num_players) ####################################################

    # Get values.
    val_values = []
    for i in range(validation_samples):
        # Set up data loader.
        dset = DatasetRepeat([val_set, TensorDataset(val_S[:, i])])
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=num_workers)
        values = []

        for x, S in loader:
            if VECTOR==1:
                tmp1, _ =imputer(x.to(device), S.to(device))
            else:
                _, tmp1 =imputer(x.to(device), S.to(device))
            values.append(link(tmp1).cpu().data)

        val_values.append(torch.cat(values))

    val_values = torch.stack(val_values, dim=1)
    return val_S, val_values


def validate(val_loader, imputer, explainer, null, link, normalization, approx_null,debug):
    with torch.no_grad():
        # Setup.
        device = next(explainer.parameters()).device
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()

        for x, grand, S, values in val_loader:
            # Move to device.
            x = x.to(device)
            if debug:
                print("VALIDATION x shape",x.shape)
            S = S.to(device)
            
            grand = grand.to(device)
            values = values.to(device)

            # Evaluate explainer.
            pred, _ = evaluate_explainer( explainer, normalization, x, grand, null, imputer.num_players)
            if debug:
                print("VALIDATION Sbatch shape",S.shape)
                #print("VALIDATION Sbatch",S[0])
                print("VALIDATION pred  shape",pred.shape)
                print("VALIDATION values shape",values.shape)
            # Calculate loss.
            if approx_null:
                approx = null + torch.matmul(S, pred)
            else:
                approx = torch.matmul(S, pred)
            if debug:
                print("VALIDATION approx shape",approx.shape)
            loss = loss_fn(approx, values)

            # Update average.
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


class FastSHAP:
    def __init__(self,
                 explainer,
                 imputer,
                 normalization='none',
                 link=None):
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        self.num_players = imputer.num_players
        self.null = None
        self.vector = None
        if link is None or link == 'none':
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

        # Set up normalization.
        if normalization is None or normalization == 'none':
            self.normalization = None
        elif normalization == 'additive':
            self.normalization = additive_efficient_normalization #ONE USED
        else:
            raise ValueError('unsupported normalization: {}'.format(normalization))

    def train(self,
              train_data,
              val_data,
              batch_size,
              num_samples,
              max_epochs,
              vector,
              lr=2e-4,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              paired_sampling=True,
              validation_samples=None,
              lookback=5,
              training_seed=None,
              validation_seed=None,
              num_workers=0,
              bar=False,
              verbose=False,
              weight_decay=0.01,
              approx_null=True,
              debug=False
              ):

        # Set up explainer model.
        explainer = self.explainer #NEURAL NETWORK
        num_players = self.num_players
        imputer = self.imputer
        link = self.link
        normalization = self.normalization
        self.vector=vector
        explainer.train()
        device = next(explainer.parameters()).device

        # Verify other arguments.
        if validation_samples is None:
            validation_samples = num_samples

        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            x_train = torch.tensor(train_data, dtype=torch.float32)
            train_set = TensorDataset(x_train)
        elif isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be np.ndarray, torch.Tensor or Dataset')

        # Set up validation dataset.
        if isinstance(val_data, np.ndarray):
            x_val = torch.tensor(val_data, dtype=torch.float32)
            val_set = TensorDataset(x_val)
        elif isinstance(val_data, torch.Tensor):
            val_set = TensorDataset(val_data)
        elif isinstance(val_data, Dataset):
            val_set = val_data
        else:
            raise ValueError('train_data must be np.ndarray, torch.Tensor or Dataset')

        grand_train = calculate_grand_coalition(train_set, imputer, batch_size * num_samples, link, device, num_workers,debug=False, VECTOR=vector).cpu()

        grand_val = calculate_grand_coalition(val_set, imputer, batch_size * num_samples, link, device, num_workers,debug=False, VECTOR=vector).cpu()

        # Null coalition.
        with torch.no_grad():
            zeros = torch.zeros(1, num_players, dtype=torch.float32, device=device)
            #print(train_set[0][0].unsqueeze(0))
            if vector==1:
                tmp1,_=imputer(train_set[0][0].unsqueeze(0).to(device), zeros)
            else:
                _,tmp1=imputer(train_set[0][0].unsqueeze(0).to(device), zeros)
            null = link(tmp1) # NON PORTA CONTRIBUTO
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
        self.null=null
        if debug:
            print("NULL:",null.shape,null)

        # Set up train loader.
        train_set_tmp = DatasetRepeat([train_set, TensorDataset(grand_train)]) # PERMETTE DI AVERE ELEMENTI RIPETUTI QUANDO LA LUN E' DIVERSA
        train_loader = DataLoader(train_set_tmp, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=num_workers) ######################################## SET SHUFFLE TO FALSE
        #print("LEN ORIGINAL:",len(train_data),"LEN TRAIN SET:",len(train_set),"LEN DATALOADER:",len(train_loader)*batch_size)

        sampler = ShapleySampler(num_players)
        
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        val_S, val_values = generate_validation_data(val_set, imputer, validation_samples, sampler, batch_size * num_samples, link, device, num_workers, VECTOR=vector)

        # Set up val loader.
        val_set_tmp = DatasetRepeat([val_set, TensorDataset(grand_val, val_S, val_values)])
        val_loader = DataLoader(val_set_tmp, batch_size=batch_size * num_samples, pin_memory=True, num_workers=num_workers)
        


        # Setup for training.
        loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(explainer.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr, verbose=verbose)
        self.loss_list = []
        best_loss = np.inf
        best_epoch = -1
        best_model = None

        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Batch iterable.

                if bar:
                    batch_iter = tqdm(train_loader, desc='Training epoch')
                else:
                    batch_iter = train_loader

                for iter, v2 in enumerate(batch_iter):
                    x,grand=v2
                    # Sample S.
                    if debug and epoch==0 and iter<5:
                        print("DATA:",x.shape,x)
                        print("GRAND:",grand.shape,grand)

                    S = sampler.sample(batch_size * num_samples, paired_sampling=paired_sampling)
                    if debug and epoch==0 and iter==0:
                        print("SUBSET:",S.shape,S)
                    #print("S shape",S.shape)
                    #print("S",S)

                    # Move to device.
                    x = x.to(device)
                    S = S.to(device)
                    grand = grand.to(device)

                    # Evaluate value function.
                    if debug and epoch==0 and iter<5:
                        print("UNSQUEEZE:",x.unsqueeze(1))
                    x_tiled = x.unsqueeze(1).repeat(
                        1, num_samples, *[1 for _ in range(len(x.shape) - 1)]
                        ).reshape(batch_size * num_samples, *x.shape[1:])
                    if debug and epoch==0 and iter<5:
                        print("X_TILED",x_tiled.shape,x_tiled)


                    with torch.no_grad():
                        if vector==1:
                            tmp1, _=imputer(x_tiled, S)
                        else:
                            _, tmp1=imputer(x_tiled, S)
                        values = link(tmp1)
                    if debug and epoch==0 and iter<5:
                        print("VALUES",values.shape,values)

                    # Evaluate explainer.
                    pred, total = evaluate_explainer(explainer, normalization, x, grand, null, num_players) #NULL PER LA NORMALIZZAZIONE
                    if debug and epoch==0 and iter<5:
                        print("PRED:",pred.shape,pred)
                        print("TOTAL:",total.shape,total)

                    # Calculate loss.
                    S = S.reshape(batch_size, num_samples, num_players)
                    if debug and epoch==0 and iter<5:
                        print("S RESHAPE",S.shape,S)

                    #print("value shape", values.shape)
                    values = values.reshape(batch_size, num_samples, -1)
                    if debug and epoch==0 and iter<5:
                        print("VALUES RESHAPE",values.shape,values)

                    if approx_null:
                        approx = null + torch.matmul(S, pred)
                    else:
                        approx = torch.matmul(S, pred)

                    if debug and epoch==0 and iter<5:
                        print("APPROX",approx.shape,approx)

                    loss = loss_fn(approx, values)
                    if eff_lambda:
                        print("EFF_LAMBDA")
                        loss = loss + eff_lambda * loss_fn(total, grand - null)

                    # Take gradient step.
                    loss = loss * num_players
                    loss.backward()
                    optimizer.step()
                    explainer.zero_grad()

                # Evaluate validation loss.
                explainer.eval()
                val_loss = num_players * validate(val_loader, imputer, explainer, null, link, normalization, approx_null,debug=debug)#.item()
                explainer.train()

                # Save loss, print progress.
                #if verbose:
                #    if vector==1:
                #        print('----- Epoch = {} -----'.format(epoch + 1))
                #    print('Val loss = {:.6f}'.format(val_loss))
                #    print('')
                scheduler.step(val_loss)
                self.loss_list.append(val_loss)

                # Check for convergence.
                if self.loss_list[-1] < best_loss:
                    best_loss = self.loss_list[-1]
                    best_epoch = epoch
                    best_model = deepcopy(explainer)
                    if verbose:
                        print('----- Epoch = {} -----'.format(epoch + 1))
                        print('New best epoch, loss = {:.6f}'.format(val_loss))
                        print('')
                elif epoch - best_epoch == lookback:
                    if verbose:
                        print('Stopping early at epoch = {}'.format(epoch))
                    break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(),best_model.parameters()):
            param.data = best_param.data
        explainer.eval()

    def shap_values(self, x, vector):
        # Data conversion.
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError('data must be np.ndarray or torch.Tensor')

        # Ensure null coalition is calculated.
        device = next(self.explainer.parameters()).device
        if self.null is None:
            #print("NULL INITIALIZATION")
            with torch.no_grad():
                zeros = torch.zeros(1, self.num_players, dtype=torch.float32, device=device)
                if vector==1:
                    tmp1,_=self.imputer(x[:1].to(device, zeros))
                else:
                    _,tmp1=self.imputer(x[:1].to(device, zeros))
                null = self.link(tmp1) # NON PORTA CONTRIBUTO
            if len(null.shape) == 1:
                null = null.reshape(1, 1)
            self.null = null

        # Generate explanations.
        with torch.no_grad():
            # Calculate grand coalition (for normalization).
            if self.normalization:
                grand = calculate_grand_coalition(x, self.imputer, len(x), self.link, device, 0, debug=False, VECTOR=vector) #CALCOLO CON TUTTI A 1
            else:
                grand = None

            # Evaluate explainer.
            x = x.to(device)
            pred = evaluate_explainer(self.explainer, self.normalization, x, grand, self.null, self.imputer.num_players, inference=True)

        return pred.cpu().data.numpy()