
import Datasets
from sklearn.ensemble import RandomForestClassifier
from OriginalModel import OriginalModelVV
import torch
import torch.nn as nn
from ErrorLib import MaskLayer1d, KLDivLoss
from MultiSurrogate import Surrogate_VV
import os
from FastSHAP import FastSHAP
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from shapreg import ShapleyRegression, PredictionGame
import pickle
import matplotlib.pyplot as plt
import os
from MultiFastSHAP import MultiFastSHAP
import time

print(os.getcwd())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

#%% DATASET
X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset = Datasets.Monks()

#%% ORIGINAL MODEL
modelRF = RandomForestClassifier(random_state=0)
modelRF.fit(X_train_s, Y_train)

om_VV=OriginalModelVV(modelRF)

#%%
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#%% SURROGATE MODEL
class MultiTaskModel(nn.Module):
    def __init__(self, Layer_size):
        super(MultiTaskModel,self).__init__()
        self.body = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, Layer_size), #FA IL CAT DELLA MASK (SUBSET S)
            nn.ReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.ReLU(inplace=True),
        )
        self.head1 = nn.Sequential(
            nn.Linear(Layer_size, 2),
        )
        self.head2 = nn.Sequential(
            nn.Linear(Layer_size, 2),
        )

    def forward(self, x):
        x = self.body(x)
        v1 = self.head1(x)
        v2 = self.head2(x)
        return v1, v2


#%% TRAINING SURROGATE MODEL
def original_model_VV(x):
    pred1, pred2 = om_VV.get_pred_VV(x.cpu().numpy()) #MODELLO ORIGINALE, PRED ALWAYS ON POSITIVE CLASS
    return torch.tensor(pred1, dtype=torch.float32, device=x.device), torch.tensor(pred2, dtype=torch.float32, device=x.device)

LOAD=False
if LOAD:
    if os.path.isfile(f'models/{dataset} surrogate_VV.pt'):
        print('Loading saved surrogate model')
        surr_VV = torch.load(f'models/{dataset} surrogate_VV.pt').to(device)
        surrogate_VV = Surrogate_VV(surr_VV, num_features)    
else:
    print('Training surrogate model')
    surr_VV=MultiTaskModel(512).to(device) #512
    surrogate_VV = Surrogate_VV(surr_VV, num_features) #model, 12

    surrogate_VV.train_original_model_VV(
        X_train_s,
        X_val_s,
        original_model_VV,
        batch_size=8,
        max_epochs=200,
        loss_fn1=KLDivLoss(),#nn.MSELoss(),
        #loss_fn2=KLDivLoss(),#nn.MSELoss(),#CrossEntropyLoss(),
        alpha=1,
        beta=1,
        validation_samples=10,
        validation_batch_size=10000,
        verbose=True,
        lr=1e-4,
        min_lr=1e-6,
        lr_factor=0.5,
        weight_decay=0.01,
        debug=False,
        training_seed=29,
        lookback=20,
        )

#%% SAVE SURROGATE MODEL
SAVE=False
if SAVE:
    surr_VV.cpu()
    torch.save(surr_VV, f'models/{dataset} surrogate_VV.pt')
    surr_VV.to(device)
#%% FASTSHAP 1

SEED=291297
torch.manual_seed(SEED)

LOAD=False
if LOAD and os.path.isfile(f'models/{dataset} explainer1.pt'):
    print('Loading saved explainer model')
    explainer1 = torch.load(f'models/{dataset} explainer1.pt').to(device)
    fastshap1 = FastSHAP(explainer1, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
else:
    print('Training explainer model')
    LAYER_SIZE=512
    explainer1 = nn.Sequential(
        nn.Linear(num_features, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)

    # Set up FastSHAP object
    fastshap1 = FastSHAP(explainer1, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

    # Train
    fastshap1.train(
        X_train_s,
        X_val_s,
        batch_size=8,
        num_samples=8, ##############
        max_epochs=400,#200
        vector=1,
        validation_samples=128,
        verbose=True,
        paired_sampling=True,
        approx_null=True,
        lr=1e-2,
        min_lr=1e-5,
        lr_factor=0.5,
        weight_decay=0.05,
        training_seed=SEED,
        lookback=20,
        debug=False) ########################################à

#%% SAVE EXPLAINER1
SAVE=False
if SAVE:
    explainer1.cpu()
    torch.save(explainer1, f'models/{dataset} explainer1.pt')
    explainer1.to(device)

#%% FASTSHAP 2
LOAD=False
if LOAD and os.path.isfile(f'models/{dataset} explainer2.pt'):
    print('Loading saved explainer model')
    explainer2 = torch.load(f'models/{dataset} explainer2.pt').to(device)
    fastshap2 = FastSHAP(explainer1, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
else:
    LAYER_SIZE=512
    explainer2 = nn.Sequential(
        nn.Linear(num_features, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, LAYER_SIZE),
        nn.LeakyReLU(inplace=True),
        nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)

    # Set up FastSHAP object
    fastshap2 = FastSHAP(explainer2, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

    # Train
    fastshap2.train(
        X_train_s,
        X_val_s,
        batch_size=8,
        num_samples=8,
        max_epochs=400,#200
        vector=2,
        validation_samples=128,
        verbose=True,
        paired_sampling=True,
        approx_null=True,
        lr=1e-2,
        min_lr=1e-5,
        lr_factor=0.5,
        weight_decay=0.05,
        training_seed=SEED,
        lookback=20,
        debug=False)

#%% SAVE EXPLAINER2
SAVE=False
if SAVE:
    explainer2.cpu()
    torch.save(explainer2, f'models/{dataset} explainer2.pt')
    explainer2.to(device)

#%% Multi-FASTSHAP
class MultiTaskExplainer(nn.Module):
    def __init__(self, Layer_size):
        super(MultiTaskExplainer,self).__init__()
        self.body = nn.Sequential(
            nn.Linear(num_features, Layer_size), #FA IL CAT DELLA MASK (SUBSET S)
            nn.LeakyReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.LeakyReLU(inplace=True),
        )
        self.head1 = nn.Sequential(
            nn.Linear(Layer_size, 2*num_features),
        )
        self.head2 = nn.Sequential(
            nn.Linear(Layer_size, 2*num_features),
        )

    def forward(self, x):
        x = self.body(x)
        v1 = self.head1(x)
        v2 = self.head2(x)
        return v1, v2

#%% FASTSHAP 3
explainer3 = MultiTaskExplainer(512).to(device)

fastshap3 = MultiFastSHAP(explainer3, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

# Train
fastshap3.train(
    X_train_s,
    X_val_s,
    batch_size=8,
    num_samples=8, ##############
    max_epochs=400,#200
    validation_samples=128,
    verbose=True,
    paired_sampling=True,
    approx_null=True,
    lr=1e-2,
    min_lr=1e-5,
    lr_factor=0.5,
    weight_decay=0.05,
    training_seed=SEED,
    lookback=20,
    debug=False) ########################################à

#%% SAVE EXPLAINER3
explainer3.cpu()
torch.save(explainer3, f'models/{dataset} explainer3.pt')
explainer3.to(device)


#%% COMPARISON
def imputer_lower(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    pred1, pred2 = surrogate_VV(x, S)#.softmax(dim=-1)
    pred1=pred1.softmax(dim=-1)
    pred2=pred2.softmax(dim=-1)
    tmp1=pred1.detach().numpy()
    tmp2=pred2.detach().numpy()
    #print(tmp1)
    #print(tmp2)
    tmp=[]
    for index in range(len(tmp1)):
        tmp.append([ tmp1[index][0],tmp2[index][0]])
    #print(tmp)
    mean=np.array(tmp)
    mean=torch.as_tensor(mean)
    #mean=mean.softmax(dim=-1)
    return mean.cpu().data.numpy()

# Setup for KernelSHAP
def imputer_upper(x, S):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    pred1, pred2 = surrogate_VV(x, S)#.softmax(dim=-1)
    pred1=pred1.softmax(dim=-1)
    pred2=pred2.softmax(dim=-1)
    tmp1=pred1.detach().numpy()
    tmp2=pred2.detach().numpy()
    #print(tmp1)
    #print(tmp2)
    tmp=[]
    for index in range(len(tmp1)):
        tmp.append([ tmp2[index][1],tmp1[index][1]])
    #print(tmp)
    mean=np.array(tmp)
    mean=torch.as_tensor(mean)
    #mean=mean.softmax(dim=-1)
    return mean.cpu().data.numpy()


mean_error_rate_fs=[]
mean_error_rate_mfs=[]
mean_error_rate_ts=[]
mean_error_rate_ks=[]
mean_error_rate_ms=[]

L2_distances_tf=[]
L2_distances_tmf=[]
L2_distances_tk=[]
L2_distances_tm=[]
L2_distances_tf_err=[]
L2_distances_tmf_err=[]
L2_distances_tk_err=[]
L2_distances_tm_err=[]
L1_distances_tf=[]
L1_distances_tmf=[]
L1_distances_tk=[]
L1_distances_tm=[]
L1_distances_tf_err=[]
L1_distances_tmf_err=[]
L1_distances_tk_err=[]
L1_distances_tm_err=[]

average_fs=[]
average_mfs=[]
average_ts=[]
average_ks=[]
average_ms=[]
average_fs_err=[]
average_mfs_err=[]
average_ts_err=[]
average_ks_err=[]
average_ms_err=[]

list_time_fs=[]
list_time_mfs=[]
list_time_tk=[]
list_time_ms=[]

X_train_s_TMP=pd.DataFrame(X_train_s, columns=feature_names)
#X_test_s_TMP=pd.DataFrame(X_test_s, columns=feature_names)

kernelshap_iters=128

#%%

#for ind in tqdm(range(len(X_test_s))):
for ind in tqdm(range(len(X_train_s[:100]))):
    x = X_train_s[ind:ind+1]
    y = int(Y_train[ind])

    # Run FastSHAP
    t1=time.time()
    fastshap_values1 = fastshap1.shap_values(x, vector=1)[0]
    fastshap_values2 = fastshap2.shap_values(x, vector=2)[0]
    t2=time.time()
    list_time_fs.append(t2-t1)
    
    fastshap_values_mean=[]
    fastshap_values_ci=[]
    error_rate_fs=0
    for el1, el2 in zip(fastshap_values1, fastshap_values2):
        if y==0:
            if el1[0]>el2[1]:
                error_rate_fs+=1
        else:
            if el2[0]>el1[1]:
                error_rate_fs+=1
        m1=(el1[0]+el2[1])/2
        m2=(el2[0]+el1[1])/2
        c1=np.abs(m1-el1[0])
        c2=np.abs(m2-el2[0])

        fastshap_values_mean.append([m1, m2])
        fastshap_values_ci.append([c1, c2])

    fastshap_values_mean=np.array(fastshap_values_mean)
    fastshap_values_ci=np.array(fastshap_values_ci)

    mean_error_rate_fs.append(error_rate_fs)

    # Run MultiFastSHAP
    t1 = time.time()
    multi1, multi2 = fastshap3.shap_values(x, vector=1)
    multi1 = multi1[0, :, :]
    multi2 = multi2[0, :, :]
    t2 = time.time()
    list_time_mfs.append(t2 - t1)

    multifastshap_values_mean = []
    multifastshap_values_ci = []
    error_rate_mfs = 0
    for el1, el2 in zip(multi1, multi2):
        if y == 0:
            if el1[0] > el2[1]:
                error_rate_mfs += 1
        else:
            if el2[0] > el1[1]:
                error_rate_mfs += 1
        m1 = (el1[0] + el2[1]) / 2
        m2 = (el2[0] + el1[1]) / 2
        c1 = np.abs(m1 - el1[0])
        c2 = np.abs(m2 - el2[0])

        multifastshap_values_mean.append([m1, m2])
        multifastshap_values_ci.append([c1, c2])

    multifastshap_values_mean = np.array(multifastshap_values_mean)
    multifastshap_values_ci = np.array(multifastshap_values_ci)

    mean_error_rate_mfs.append(error_rate_mfs)


    # Run TrueSHAP/KernelSHAP

    t1=time.time()
    game_l = PredictionGame(imputer_lower, x)
    shap_values_l, all_results_l = ShapleyRegression(game_l, batch_size=32, paired_sampling=False, detect_convergence=True,bar=False, return_all=True)

    game_u = PredictionGame(imputer_upper, x)
    shap_values_u, all_results_u = ShapleyRegression(game_u, batch_size=32, paired_sampling=False, detect_convergence=True,bar=False, return_all=True)
    t2=time.time()
    list_time_tk.append(t2-t1)
    
    tmp1=shap_values_l.values[:, y]
    tmp2=shap_values_u.values[:, y]
    error_rate_ts=0
    for el1,el2 in zip(tmp1,tmp2):
        #print(el1,el2)
        if el1>el2:
            error_rate_ts+=1
    mean_error_rate_ts.append(error_rate_ts)
    mean_ts=(tmp2+tmp1)/2
    err_ts=np.abs(mean_ts-tmp2)
    
    tmp3=all_results_l['values'][list(all_results_l['iters']).index(kernelshap_iters)][:, y]
    tmp4=all_results_u['values'][list(all_results_u['iters']).index(kernelshap_iters)][:, y]
    error_rate_ks=0
    for el1,el2 in zip(tmp3,tmp4):
        #print(el1,el2)
        if el1>el2:
            error_rate_ks+=1
    mean_error_rate_ks.append(error_rate_ks)
    mean_ks=(tmp3+tmp4)/2
    err_ks=np.abs(mean_ks-tmp4)
    
    mean_fs=fastshap_values_mean[:,y]
    err_fs=fastshap_values_ci[:,y]

    mean_mfs = multifastshap_values_mean[:, y]
    err_mfs = multifastshap_values_ci[:, y]


    # Run MonteCarlo
    t1=time.time()
    mean_mc, err_mc, emc=0,0,0#MonteCarlo(ind, X_train_s_TMP, modelRF, om)
    t2=time.time()
    list_time_ms.append(t2-t1)
    
    mean_error_rate_ms.append(emc)


    # Compute distances
    distance_tf=np.linalg.norm(np.array(mean_ts)-np.array(mean_fs))
    distance_tmf = np.linalg.norm(np.array(mean_ts) - np.array(mean_mfs))
    distance_tk=np.linalg.norm(np.array(mean_ts)-np.array(mean_ks))
    distance_tm=np.linalg.norm(np.array(mean_ts)-np.array(mean_mc))
    distance_tf_err=np.linalg.norm(np.array(err_ts)-np.array(err_fs))
    distance_tmf_err = np.linalg.norm(np.array(err_ts) - np.array(err_mfs))
    distance_tk_err=np.linalg.norm(np.array(err_ts)-np.array(err_ks))
    distance_tm_err=np.linalg.norm(np.array(err_ts)-np.array(err_mc))
    
    L2_distances_tf.append(distance_tf)
    L2_distances_tmf.append(distance_tmf)
    L2_distances_tk.append(distance_tk)
    L2_distances_tm.append(distance_tm)
    L2_distances_tf_err.append(distance_tf_err)
    L2_distances_tmf_err.append(distance_tmf_err)
    L2_distances_tk_err.append(distance_tk_err)
    L2_distances_tm_err.append(distance_tm_err)
    
    distance_tf_l1=np.linalg.norm(np.array(mean_ts)-np.array(mean_fs),ord=1)
    distance_tmf_l1 = np.linalg.norm(np.array(mean_ts) - np.array(mean_mfs), ord=1)
    distance_tk_l1=np.linalg.norm(np.array(mean_ts)-np.array(mean_ks),ord=1)
    distance_tm_l1=np.linalg.norm(np.array(mean_ts)-np.array(mean_mc),ord=1)
    distance_tf_err_l1=np.linalg.norm(np.array(err_ts)-np.array(err_fs),ord=1)
    distance_tmf_err_l1 = np.linalg.norm(np.array(err_ts) - np.array(err_mfs), ord=1)
    distance_tk_err_l1=np.linalg.norm(np.array(err_ts)-np.array(err_ks),ord=1)
    distance_tm_err_l1=np.linalg.norm(np.array(err_ts)-np.array(err_mc),ord=1)
    
    L1_distances_tf.append(distance_tf_l1)
    L1_distances_tmf.append(distance_tmf_l1)
    L1_distances_tk.append(distance_tk_l1)
    L1_distances_tm.append(distance_tm_l1)
    L1_distances_tf_err.append(distance_tf_err_l1)
    L1_distances_tmf_err.append(distance_tmf_err_l1)
    L1_distances_tk_err.append(distance_tk_err_l1)
    L1_distances_tm_err.append(distance_tm_err_l1)
    
    average_fs.append(mean_fs)
    average_mfs.append(mean_mfs)
    average_ts.append(mean_ts)
    average_ks.append(mean_ks)
    average_ms.append(mean_mc)
    average_fs_err.append(err_fs)
    average_mfs_err.append(err_mfs)
    average_ts_err.append(err_ts)
    average_ks_err.append(err_ks)
    average_ms_err.append(err_mc)
    #break

#%% SAVE VARIABLES
SAVE=False
if SAVE:
    with open(f'dump/{dataset}L2_distances_tf.pkl', 'wb') as f:
        pickle.dump(L2_distances_tf, f)
    with open(f'dump/{dataset}L2_distances_tk.pkl', 'wb') as f:
        pickle.dump(L2_distances_tk, f)
    with open(f'dump/{dataset}L2_distances_tm.pkl', 'wb') as f:
        pickle.dump(L2_distances_tm, f)
    with open(f'dump/{dataset}L2_distances_tf_err.pkl', 'wb') as f:
        pickle.dump(L2_distances_tf_err, f)
    with open(f'dump/{dataset}L2_distances_tk_err.pkl', 'wb') as f:
        pickle.dump(L2_distances_tk_err, f)
    with open(f'dump/{dataset}L2_distances_tm_err.pkl', 'wb') as f:
        pickle.dump(L2_distances_tm_err, f)
        
    with open(f'dump/{dataset}L1_distances_tf.pkl', 'wb') as f:
        pickle.dump(L1_distances_tf, f)
    with open(f'dump/{dataset}L1_distances_tk.pkl', 'wb') as f:
        pickle.dump(L1_distances_tk, f)
    with open(f'dump/{dataset}L1_distances_tm.pkl', 'wb') as f:
        pickle.dump(L1_distances_tm, f)
    with open(f'dump/{dataset}L1_distances_tf_err.pkl', 'wb') as f:
        pickle.dump(L1_distances_tf_err, f)
    with open(f'dump/{dataset}L1_distances_tk_err.pkl', 'wb') as f:
        pickle.dump(L1_distances_tk_err, f)
    with open(f'dump/{dataset}L1_distances_tm_err.pkl', 'wb') as f:
        pickle.dump(L1_distances_tm_err, f)
        
    with open(f'dump/{dataset}average_fs.pkl', 'wb') as f:
        pickle.dump(average_fs, f)
    with open(f'dump/{dataset}average_ts.pkl', 'wb') as f:
        pickle.dump(average_ts, f)
    with open(f'dump/{dataset}average_ks.pkl', 'wb') as f:
        pickle.dump(average_ks, f)
    with open(f'dump/{dataset}average_ms.pkl', 'wb') as f:
        pickle.dump(average_ms, f)
        
    with open(f'dump/{dataset}average_fs_err.pkl', 'wb') as f:
        pickle.dump(average_fs_err, f)
    with open(f'dump/{dataset}average_ts_err.pkl', 'wb') as f:
        pickle.dump(average_ts_err, f)
    with open(f'dump/{dataset}average_ks_err.pkl', 'wb') as f:
        pickle.dump(average_ks_err, f)
    with open(f'dump/{dataset}average_ms_err.pkl', 'wb') as f:
        pickle.dump(average_ms_err, f)
        
    with open(f'dump/{dataset}mean_error_rate_fs.pkl', 'wb') as f:
        pickle.dump(mean_error_rate_fs, f)
    with open(f'dump/{dataset}mean_error_rate_ts.pkl', 'wb') as f:
        pickle.dump(mean_error_rate_ts, f)
    with open(f'dump/{dataset}mean_error_rate_ks.pkl', 'wb') as f:
        pickle.dump(mean_error_rate_ks, f)
    with open(f'dump/{dataset}mean_error_rate_ms.pkl', 'wb') as f:
        pickle.dump(mean_error_rate_ms, f)

    with open(f'dump/{dataset}list_time_fs.pkl', 'wb') as f:
        pickle.dump(list_time_fs, f)
    with open(f'dump/{dataset}list_time_tk.pkl', 'wb') as f:
        pickle.dump(list_time_tk, f)
    with open(f'dump/{dataset}list_time_ms.pkl', 'wb') as f:
        pickle.dump(list_time_ms, f)

#%% LOAD VARIABLES
LOAD=False
if LOAD:
    file = open(f'dump/{dataset}average_fs_err.pkl', 'rb')
    average_fs_err=pickle.load(file)
    file = open(f'dump/{dataset}average_ts_err.pkl', 'rb')
    average_ts_err=pickle.load(file)
    file = open(f'dump/{dataset}average_ks_err.pkl', 'rb')
    average_ks_err=pickle.load(file)
    file = open(f'dump/{dataset}average_ms_err.pkl', 'rb')
    average_ms_err=pickle.load(file)


    file = open(f'dump/{dataset}average_fs.pkl', 'rb')
    average_fs=pickle.load(file)
    file = open(f'dump/{dataset}average_ts.pkl', 'rb')
    average_ts=pickle.load(file)
    file = open(f'dump/{dataset}average_ks.pkl', 'rb')
    average_ks=pickle.load(file)
    file = open(f'dump/{dataset}average_ms.pkl', 'rb')
    average_ms=pickle.load(file)

#%%
average_fs_err=np.array(average_fs_err)
average_mfs_err=np.array(average_mfs_err)
average_ts_err=np.array(average_ts_err)
average_ks_err=np.array(average_ks_err)
average_ms_err=np.array(average_ms_err)

e1=np.mean(average_ts_err)
e2=np.mean(average_fs_err)
e3=np.mean(average_ks_err)
e4=np.mean(average_ms_err)
e5=np.mean(average_mfs_err)
print("TRUE", round(e1,6))
print("FAST",round(e2,6))
print("KERN",round(e3,6))
print("MCAR",round(e4,6))
print("MFS",round(e5,6))

#%%
print("AVERAGE L2 - MEAN - TRUE-FAST:",np.mean(L2_distances_tf))
print("AVERAGE L2 - MEAN - TRUE-MULT:",np.mean(L2_distances_tmf))
print("AVERAGE L2 - MEAN - TRUE-KERN:",np.mean(L2_distances_tk))
print("AVERAGE L2 - MEAN - TRUE-MONT:",np.mean(L2_distances_tm))
print("AVERAGE L2 - ERR - TRUE-FAST:",np.mean(L2_distances_tf_err))
print("AVERAGE L2 - ERR - TRUE-MULT:",np.mean(L2_distances_tmf_err))
print("AVERAGE L2 - ERR - TRUE-KERN:",np.mean(L2_distances_tk_err))
print("AVERAGE L2 - ERR - TRUE-MONT:",np.mean(L2_distances_tm_err))

print("AVERAGE L1 - MEAN - TRUE-FAST:",np.mean(L1_distances_tf))
print("AVERAGE L1 - MEAN - TRUE-MULT:",np.mean(L1_distances_tmf))
print("AVERAGE L1 - MEAN - TRUE-KERN:",np.mean(L1_distances_tk))
print("AVERAGE L1 - MEAN - TRUE-MONT:",np.mean(L1_distances_tm))
print("AVERAGE L1 - ERR - TRUE-FAST:",np.mean(L1_distances_tf_err))
print("AVERAGE L1 - ERR - TRUE-MULT:",np.mean(L1_distances_tmf_err))
print("AVERAGE L1 - ERR - TRUE-KERN:",np.mean(L1_distances_tk_err))
print("AVERAGE L1 - ERR - TRUE-MONT:",np.mean(L1_distances_tm_err))

print("AVERAGE SAMPLE TIME FAST:",np.mean(list_time_fs))
print("AVERAGE SAMPLE TIME MULT:",np.mean(list_time_mfs))
print("AVERAGE SAMPLE TIME TRUE/KERNEL:",np.mean(list_time_tk))
print("AVERAGE SAMPLE TIME MONTECARLO:",np.mean(list_time_ms))

print("TOTAL SAMPLE TIME FAST:",np.sum(list_time_fs))
print("TOTAL SAMPLE TIME MULT:",np.sum(list_time_mfs))
print("TOTAL SAMPLE TIME TRUE/KERNEL:",np.sum(list_time_tk))
print("TOTAL SAMPLE TIME MONTECARLO:",np.sum(list_time_ms))

print("AVERAGE ERROR FAST:",np.mean(mean_error_rate_fs))
print("AVERAGE ERROR MULT:",np.mean(mean_error_rate_mfs))
print("AVERAGE ERROR TRUE:",np.mean(mean_error_rate_ts))
print("AVERAGE ERROR KERNEL:",np.mean(mean_error_rate_ks))
print("AVERAGE ERROR MONTECARLO:",np.mean(mean_error_rate_ms))

#%%
average_fs=np.array(average_fs)
average_mfs=np.array(average_mfs)
average_ts=np.array(average_ts)
average_ks=np.array(average_ks)
average_ms=np.array(average_ms)
average_fs_err=np.array(average_fs_err)
average_mfs_err=np.array(average_mfs_err)
average_ts_err=np.array(average_ts_err)
average_ks_err=np.array(average_ks_err)
average_ms_err=np.array(average_ms_err)

m1=np.mean(average_ts,axis=0)
m2=np.mean(average_fs,axis=0)
m3=np.mean(average_ks,axis=0)
m4=np.mean(average_ms,axis=0)
m5=np.mean(average_mfs,axis=0)
e1=np.mean(average_ts_err,axis=0)
e2=np.mean(average_fs_err,axis=0)
e3=np.mean(average_ks_err,axis=0)
e4=np.mean(average_ms_err,axis=0)
e5=np.mean(average_mfs_err,axis=0)

#%% PLOT GLOBAL SV
plt.figure(figsize=(16, 9))
width = 0.75
kernelshap_iters = 128
N=4
error_kw=dict(lw=3, capsize=10, capthick=3)

plt.bar(np.arange(num_features) - 2*width/N, m1, width / N, label='Interval True SHAP',  yerr=e1, error_kw=error_kw, color='tab:green')
plt.bar(np.arange(num_features) - 1*width/N, m2, width / N, label='Interval FastSHAP',  yerr=e2, error_kw=error_kw, color='tab:orange')
plt.bar(np.arange(num_features) - 0.0*width/N, m5, width / N, label='Interval MultiFastSHAP',  yerr=e5, error_kw=error_kw, color='tab:olive')
plt.bar(np.arange(num_features) + 1*width/N, m3, width / N, label='Interval KernelSHAP', yerr=e3, error_kw=error_kw, color='tab:red')
plt.bar(np.arange(num_features) + 2*width/N, m4, width / N, label='Interval MonteCarlo',  yerr=e4, error_kw=error_kw, color='tab:purple')

plt.legend(fontsize=16)
plt.tick_params(labelsize=14)
plt.ylabel('SHAP Values', fontsize=16)
plt.xticks(np.arange(num_features), feature_names, rotation=35, rotation_mode='anchor', ha='right')
#plt.savefig(f'figure/{dataset} Global Interval SV comparison', bbox_inches = "tight")

plt.show()
