import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ErrorLib import MaskLayer1d, MaskLayer2d, KLDivLoss, DatasetRepeat, ShapleySampler
import numpy as np
from tqdm.auto import tqdm
import torch.optim as optim
from copy import deepcopy


def additive_efficient_normalization(pred, grand, null):
    gap = (grand - null) - torch.sum(pred, dim=1)
    # gap = gap.detach()
    return pred + gap.unsqueeze(1) / pred.shape[1]


def evaluate_explainer(explainer, normalization, x, grand1, grand2, null1, null2, num_players, inference=False):
    # Evaluate explainer.
    pred1, pred2 = explainer(x)

    # Reshape SHAP values.

    # Tabular.
    image_shape = None
    pred1 = pred1.reshape(len(x), num_players, -1)
    pred2 = pred2.reshape(len(x), num_players, -1)

    # For pre-normalization efficiency gap.
    total1 = pred1.sum(dim=1)
    total2 = pred2.sum(dim=1)

    # Apply normalization.
    if normalization:
        pred1 = normalization(pred1, grand1, null1)  # SERVE PER GARANTIRE LE STIME VICINE AGLI SV - EFFICIENCY CONSTRAINT
        pred2 = normalization(pred2, grand2, null2)  # SERVE PER GARANTIRE LE STIME VICINE AGLI SV - EFFICIENCY CONSTRAINT

    # Reshape for inference.
    if inference:
        return pred1, pred2

    return pred1, total1, pred2, total2


def calculate_grand_coalition(dataset, imputer, batch_size, link, device, num_workers, debug):
    if debug:
        print("CALCULATE_GRAND_COALITION")
    ones = torch.ones(batch_size, imputer.num_players, dtype=torch.float32, device=device)  # 32x12
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    with torch.no_grad():
        grand1 = []
        grand2 = []
        for x in loader:
            x = x[0]
            if debug:
                print("INPUT:", len(x), x)

            tmp1, tmp2 = imputer(x.to(device), ones[:len(x)])
            tmp1 = link(tmp1)
            tmp2 = link(tmp2)
            if debug:
                print("GRAND(SOFTMAX)1:", tmp1)
                print("GRAND(SOFTMAX)2:", tmp2)
            grand1.append(tmp1)  # SOFTMAX(SURROGATE(DATA+ONES))
            grand2.append(tmp2)  # SOFTMAX(SURROGATE(DATA+ONES))

        # Concatenate and return.
        grand1 = torch.cat(grand1)
        if len(grand1.shape) == 1:
            grand1 = grand1.unsqueeze(1)

        grand2 = torch.cat(grand2)
        if len(grand2.shape) == 1:
            grand2 = grand2.unsqueeze(1)

    return grand1, grand2


def generate_validation_data(val_set, imputer, validation_samples, sampler, batch_size, link, device, num_workers):
    # Generate coalitions.
    print("")
    val_S = sampler.sample(validation_samples * len(val_set), paired_sampling=True).reshape(len(val_set), validation_samples, imputer.num_players)

    # Get values.
    val_values1 = []
    val_values2 = []
    for i in range(validation_samples):
        # Set up data loader.
        dset = DatasetRepeat([val_set, TensorDataset(val_S[:, i])])
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        values1 = []
        values2 = []

        for x, S in loader:
            tmp1, tmp2 = imputer(x.to(device), S.to(device))
            values1.append(link(tmp1).cpu().data)
            values2.append(link(tmp2).cpu().data)

        val_values1.append(torch.cat(values1))
        val_values2.append(torch.cat(values2))

    val_values1 = torch.stack(val_values1, dim=1)
    val_values2 = torch.stack(val_values2, dim=1)
    return val_S, val_values1, val_values2


def validate(val_loader, imputer, explainer, null1, null2, link, normalization, approx_null, debug_val, constraint, alpha):
    with torch.no_grad():
        # Setup.
        device = next(explainer.parameters()).device
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()

        for x, grand1, grand2, S, values1, values2 in val_loader: #[val_set, TensorDataset(grand_val1, grand_val2, val_S, val_values1, val_values2)]
            # Move to device.
            x = x.to(device)
            if debug_val:
                print("VALIDATION x shape", x.shape)
            S = S.to(device)

            grand1 = grand1.to(device)
            grand2 = grand2.to(device)
            values1 = values1.to(device)
            values2 = values2.to(device)


            # Evaluate explainer.
            pred1, _, pred2, _ = evaluate_explainer(explainer, normalization, x, grand1, grand2, null1, null2, imputer.num_players)
            if debug_val:
                print("VALIDATION Sbatch shape", S.shape)
                # print("VALIDATION Sbatch",S[0])
                print("VALIDATION pred1 shape", pred1.shape)
                print("VALIDATION values1 shape", values1.shape)
                print("VALIDATION pred2 shape", pred2.shape)
                print("VALIDATION values2 shape", values2.shape)
            # Calculate loss.
            if approx_null:
                approx1 = null1 + torch.matmul(S, pred1)
                approx2 = null2 + torch.matmul(S, pred2)
            else:
                approx1 = torch.matmul(S, pred1)
                approx2 = torch.matmul(S, pred2)
            if debug_val:
                print("VALIDATION approx shape1", approx1.shape)
                print("VALIDATION approx shape2", approx2.shape)
            loss1 = loss_fn(approx1, values1)
            loss2 = loss_fn(approx2, values2)
            if constraint:
                vec1 = pred1[:, 0].unsqueeze(1)  # neg
                vec2 = pred2[:, 0].unsqueeze(1)  # pos
                vec3 = torch.cat((vec1, vec2), 1)

                vec4 = pred1[:, 1].unsqueeze(1)  # pos
                vec5 = pred2[:, 1].unsqueeze(1)  # neg
                vec6 = torch.cat((vec5, vec4), 1)

                # vec1 = approx2[:,1] - approx1[:, 0]
                # vec2 = approx1[:, 1] - approx2[:, 0]
                # vec3 = torch.cat((vec1.unsqueeze(1), vec2.unsqueeze(1)), 1)
                #
                # vec4 = values2[:, 1] - values1[:, 0]
                # vec5 = values1[:, 1] - values2[:, 0]
                # vec6 = torch.cat((vec5.unsqueeze(1), vec4.unsqueeze(1)), 1)

                loss3 = loss_fn(vec3, vec6)

                loss =  (1-alpha)*(loss1 + loss2) + (alpha)*(loss3)
            else:
                loss = loss1 + loss2

            # Update average.
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


class MultiFastSHAP:
    def __init__(self,
                 explainer,
                 imputer,
                 normalization='none',
                 link=None):
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        self.num_players = imputer.num_players
        self.null1 = None
        self.null2 = None
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
            self.normalization = additive_efficient_normalization  # ONE USED
        else:
            raise ValueError('unsupported normalization: {}'.format(normalization))

    def train(self,
              train_data,
              val_data,
              batch_size,
              num_samples,
              max_epochs,
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
              debug=False,
              debug_val=False,
              constraint = False,
              alpha = 0.5,
              ):

        # Set up explainer model.
        explainer = self.explainer  # NEURAL NETWORK
        num_players = self.num_players
        imputer = self.imputer
        link = self.link
        normalization = self.normalization
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

        grand_train1, grand_train2 = calculate_grand_coalition(train_set, imputer, batch_size * num_samples, link, device, num_workers, debug=False)

        grand_val1, grand_val2 = calculate_grand_coalition(val_set, imputer, batch_size * num_samples, link, device, num_workers, debug=False)

        # Null coalition.
        with torch.no_grad():
            zeros = torch.zeros(1, num_players, dtype=torch.float32, device=device)
            # print(train_set[0][0].unsqueeze(0))
            tmp1, tmp2 = imputer(train_set[0][0].unsqueeze(0).to(device), zeros)
            null1 = link(tmp1)  # NON PORTA CONTRIBUTO
            null2 = link(tmp2)  # NON PORTA CONTRIBUTO

            if len(null1.shape) == 1:
                null1 = null1.reshape(1, 1)
            if len(null2.shape) == 1:
                null2 = null2.reshape(1, 1)
        self.null1 = null1
        self.null2 = null2
        if debug:
            print("NULL:", null1.shape, null1)

        # Set up train loader.
        train_set_tmp = DatasetRepeat([train_set, TensorDataset(grand_train1, grand_train2)])  # PERMETTE DI AVERE ELEMENTI RIPETUTI QUANDO LA LUN E' DIVERSA
        train_loader = DataLoader(train_set_tmp, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=num_workers)
        # print("LEN ORIGINAL:",len(train_data),"LEN TRAIN SET:",len(train_set),"LEN DATALOADER:",len(train_loader)*batch_size)

        sampler = ShapleySampler(num_players)

        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        val_S, val_values1, val_values2 = generate_validation_data(val_set, imputer, validation_samples, sampler, batch_size * num_samples, link, device, num_workers)

        # Set up val loader.
        val_set_tmp = DatasetRepeat([val_set, TensorDataset(grand_val1, grand_val2, val_S, val_values1, val_values2)])
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
            # Batch iterable.ó

            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            for iter, v2 in enumerate(batch_iter):
                x, grand1, grand2 = v2
                # Sample S.
                if debug and epoch == 0 and iter < 5:
                    print("DATA:", x.shape, x)
                    print("GRAND1:", grand1.shape, grand1)
                    print("GRAND2:", grand2.shape, grand2)

                S = sampler.sample(batch_size * num_samples, paired_sampling=paired_sampling)
                if debug and epoch == 0 and iter == 0:
                    print("SUBSET:", S.shape, S)
                # print("S shape",S.shape)
                # print("S",S)

                # Move to device.
                x = x.to(device)
                S = S.to(device)
                grand1 = grand1.to(device)
                grand2 = grand2.to(device)

                # Evaluate value function.
                if debug and epoch == 0 and iter < 5:
                    print("UNSQUEEZE:", x.unsqueeze(1))
                x_tiled = x.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(x.shape) - 1)]
                ).reshape(batch_size * num_samples, *x.shape[1:])
                if debug and epoch == 0 and iter < 5:
                    print("X_TILED", x_tiled.shape, x_tiled)

                with torch.no_grad():
                    tmp1, tmp2 = imputer(x_tiled, S)
                    values1 = link(tmp1)
                    values2 = link(tmp2)
                if debug and epoch == 0 and iter < 5:
                    print("VALUES1", values1.shape, values1)
                    print("VALUES2", values2.shape, values2)

                # Evaluate explainer.
                pred1, total1, pred2, total2 = evaluate_explainer(explainer, normalization, x, grand1, grand2, null1, null2, num_players)  # NULL PER LA NORMALIZZAZIONE
                if debug and epoch == 0 and iter < 5:
                    print("PRED1:", pred1.shape, pred1)
                    print("TOTAL1:", total1.shape, total1)
                    print("PRED2:", pred2.shape, pred2)
                    print("TOTAL2:", total2.shape, total2)

                # Calculate loss.
                S = S.reshape(batch_size, num_samples, num_players)
                if debug and epoch == 0 and iter < 5:
                    print("S RESHAPE", S.shape, S)

                # print("value shape", values.shape)
                values1 = values1.reshape(batch_size, num_samples, -1)
                values2 = values2.reshape(batch_size, num_samples, -1)
                if debug and epoch == 150 and iter < 5:
                    print("VALUES RESHAPE1", values1.shape, values1)
                    print("VALUES RESHAPE2", values2.shape, values2)

                if approx_null:
                    approx1 = null1 + torch.matmul(S, pred1)
                    approx2 = null2 + torch.matmul(S, pred2)
                else:
                    approx1 = torch.matmul(S, pred1)
                    approx2 = torch.matmul(S, pred2)

                if debug and epoch == 150 and iter < 5:
                    print("APPROX1", approx1.shape, approx1)
                    print("APPROX2", approx2.shape, approx2)

                if debug and epoch == 150 and iter < 5:
                    temp1=values1-null1
                    temp2=values2-null2
                    print("PRED1:", torch.matmul(S, pred1).shape, torch.matmul(S, pred1))
                    print("VALUES1-NULL1", temp1.shape, temp1)
                    print("PRED2:", torch.matmul(S, pred2).shape, torch.matmul(S, pred2))
                    print("VALUES2-NULL2", temp2.shape, temp2)

                loss1 = loss_fn(approx1, values1)
                loss2 = loss_fn(approx2, values2)
                if eff_lambda:
                    print("EFF_LAMBDA")
                    #loss = loss + eff_lambda * loss_fn(total, grand - null)

                if constraint:
                    vec1 = pred1[:, 0].unsqueeze(1) #neg
                    vec2 = pred2[:, 0].unsqueeze(1) #pos
                    vec3 = torch.cat((vec1, vec2), 1)

                    vec4 = pred1[:, 1].unsqueeze(1) #pos
                    vec5 = pred2[:, 1].unsqueeze(1) #neg
                    vec6 = torch.cat((vec5, vec4), 1)

                    # vec1 = approx2[:, 1] - approx1[:, 0]
                    # vec2 = approx1[:, 1] - approx2[:, 0]
                    # vec3 = torch.cat((vec1.unsqueeze(1), vec2.unsqueeze(1)), 1)
                    #
                    # vec4 = values2[:, 1] - values1[:, 0]
                    # vec5 = values1[:, 1] - values2[:, 0]
                    # vec6 = torch.cat((vec5.unsqueeze(1), vec4.unsqueeze(1)), 1)

                    if debug and epoch == 0 and iter < 5:
                        print("VEC3", vec3.shape, vec3)
                        print("VEC6", vec6.shape, vec6)

                    loss3 = loss_fn(vec3, vec6)

                    loss = num_players * ((1-alpha)*(loss1 + loss2) + (alpha)*(loss3))
                else:
                    loss = num_players * (loss1 + loss2)
                loss.backward()
                optimizer.step()
                explainer.zero_grad()

            # Evaluate validation loss.
            explainer.eval()
            val_loss = num_players * validate(val_loader, imputer, explainer, null1, null2, link, normalization, approx_null, debug_val=debug_val, constraint=constraint, alpha=alpha)  # .item()
            explainer.train()

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
        for param, best_param in zip(explainer.parameters(), best_model.parameters()):
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

        # Generate explanations.
        with torch.no_grad():
            # Calculate grand coalition (for normalization).
            grand1, grand2 = calculate_grand_coalition(x, self.imputer, len(x), self.link, device, 0, debug=False)  # CALCOLO CON TUTTI A 1

            # Evaluate explainer.
            x = x.to(device)
            pred1, pred2 = evaluate_explainer(self.explainer, self.normalization, x, grand1, grand2, self.null1, self.null2, self.imputer.num_players, inference=True)

        return pred1.cpu().data.numpy(), pred2.cpu().data.numpy()