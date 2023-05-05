
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
from importlib import reload
import random
from Metrics import compute_metrics

#%% DATASET
X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset = Datasets.Heart()

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
SEED=291297
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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
SAVE=True
if SAVE:
    surr_VV.cpu()
    torch.save(surr_VV, f'models/{dataset} surrogate_VV.pt')
    surr_VV.to(device)
#%% FASTSHAP 1
SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


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
        debug=False,
        ) ########################################à

#%% SAVE EXPLAINER1
SAVE=True
if SAVE:
    explainer1.cpu()
    torch.save(explainer1, f'models/{dataset} explainer1.pt')
    explainer1.to(device)

#%% FASTSHAP 2
SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


LOAD=True
if LOAD and os.path.isfile(f'models/{dataset} explainer2.pt'):
    print('Loading saved explainer model')
    explainer2 = torch.load(f'models/{dataset} explainer2.pt').to(device)
    fastshap2 = FastSHAP(explainer2, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
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
SAVE=True
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
SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


LOAD=False
if LOAD and os.path.isfile(f'models/{dataset} explainer3.pt'):
    print('Loading saved explainer model')
    explainer3 = torch.load(f'models/{dataset} explainer3.pt').to(device)
    fastshap3 = MultiFastSHAP(explainer3, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
else:
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
        debug=True,
        constraint=False
    ) ########################################à

#%% SAVE EXPLAINER3
SAVE=True
if SAVE:
    explainer3.cpu()
    torch.save(explainer3, f'models/{dataset} explainer3.pt')
    explainer3.to(device)

#%% FASTSHAP 4
SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

LOAD=True
if LOAD and os.path.isfile(f'models/{dataset} explainer4.pt'):
    print('Loading saved explainer model')
    explainer4 = torch.load(f'models/{dataset} explainer4.pt').to(device)
    fastshap4 = MultiFastSHAP(explainer4, surrogate_VV, normalization='additive',link=nn.Softmax(dim=-1))
else:
    explainer4 = MultiTaskExplainer(512).to(device)
    fastshap4 = MultiFastSHAP(explainer4, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

    alpha=0.001#0.01, 0.001, 0.005
    # Train
    fastshap4.train(
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
        debug_val=False,
        debug=False,
        constraint=True,
        alpha=alpha
    ) ########################################à

#%% SAVE EXPLAINER4
SAVE=True
if SAVE:
    explainer4.cpu()
    torch.save(explainer4, f'models/{dataset} explainer4_{alpha}.pt')
    explainer4.to(device)

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

#%% COMPUTE METRICS
average_fs, average_mfs, average_mfsc, average_ts, average_ks, average_ms, \
  average_fs_err, average_mfs_err, average_mfsc_err, average_ts_err, average_ks_err, average_ms_err, \
    mean_error_rate_fs, mean_error_rate_mfs, mean_error_rate_mfsc, mean_error_rate_ts, mean_error_rate_ks, mean_error_rate_ms, \
        list_time_fs, list_time_mfs, list_time_mfsc, list_time_tk, list_time_ms, \
            L1_distances_tf, L1_distances_tmf, L1_distances_tmfc, L1_distances_tk, L1_distances_tm, \
                L1_distances_tf_err, L1_distances_tmf_err, L1_distances_tmfc_err, L1_distances_tk_err, L1_distances_tm_err, \
                    L2_distances_tm, L2_distances_tm_err, L2_distances_tmf, L2_distances_tmf_err, L2_distances_tf, L2_distances_tf_err, \
                        L2_distances_tmfc, L2_distances_tmfc_err, L2_distances_tk, L2_distances_tk_err \
                            = compute_metrics(X_train_s, feature_names, Y_train, fastshap1, fastshap2, fastshap3, fastshap4, dataset, device, surrogate_VV, SAVE=False)
#%%
average_fs_err=np.array(average_fs_err)
average_mfs_err=np.array(average_mfs_err)
average_mfsc_err=np.array(average_mfsc_err)
average_ts_err=np.array(average_ts_err)
average_ks_err=np.array(average_ks_err)
average_ms_err=np.array(average_ms_err)

e1=np.mean(average_ts_err)
e2=np.mean(average_fs_err)
e3=np.mean(average_ks_err)
e4=np.mean(average_ms_err)
e5=np.mean(average_mfs_err)
e6=np.mean(average_mfsc_err)
print("TRUE", round(e1,6))
print("FAST",round(e2,6))
print("KERN",round(e3,6))
print("MCAR",round(e4,6))
print("MFS",round(e5,6))
print("MFSC",round(e6,6))

#%% print results
print("AVERAGE L2 - MEAN - TRUE-FAST:",np.mean(L2_distances_tf))
print("AVERAGE L2 - MEAN - TRUE-MULT:",np.mean(L2_distances_tmf))
print("AVERAGE L2 - MEAN - TRUE-MULC:",np.mean(L2_distances_tmfc))
print("AVERAGE L2 - MEAN - TRUE-KERN:",np.mean(L2_distances_tk))
print("AVERAGE L2 - MEAN - TRUE-MONT:",np.mean(L2_distances_tm))
print("AVERAGE L2 - ERR - TRUE-FAST:",np.mean(L2_distances_tf_err))
print("AVERAGE L2 - ERR - TRUE-MULT:",np.mean(L2_distances_tmf_err))
print("AVERAGE L2 - ERR - TRUE-MULC:",np.mean(L2_distances_tmfc_err))
print("AVERAGE L2 - ERR - TRUE-KERN:",np.mean(L2_distances_tk_err))
print("AVERAGE L2 - ERR - TRUE-MONT:",np.mean(L2_distances_tm_err))

print("AVERAGE L1 - MEAN - TRUE-FAST:",np.mean(L1_distances_tf))
print("AVERAGE L1 - MEAN - TRUE-MULT:",np.mean(L1_distances_tmf))
print("AVERAGE L1 - MEAN - TRUE-MULC:",np.mean(L1_distances_tmfc))
print("AVERAGE L1 - MEAN - TRUE-KERN:",np.mean(L1_distances_tk))
print("AVERAGE L1 - MEAN - TRUE-MONT:",np.mean(L1_distances_tm))
print("AVERAGE L1 - ERR - TRUE-FAST:",np.mean(L1_distances_tf_err))
print("AVERAGE L1 - ERR - TRUE-MULT:",np.mean(L1_distances_tmf_err))
print("AVERAGE L1 - ERR - TRUE-MULC:",np.mean(L1_distances_tmfc_err))
print("AVERAGE L1 - ERR - TRUE-KERN:",np.mean(L1_distances_tk_err))
print("AVERAGE L1 - ERR - TRUE-MONT:",np.mean(L1_distances_tm_err))

print("AVERAGE SAMPLE TIME FAST:",np.mean(list_time_fs))
print("AVERAGE SAMPLE TIME MULT:",np.mean(list_time_mfs))
print("AVERAGE SAMPLE TIME MULC:",np.mean(list_time_mfsc))
print("AVERAGE SAMPLE TIME TRUE/KERNEL:",np.mean(list_time_tk))
print("AVERAGE SAMPLE TIME MONTECARLO:",np.mean(list_time_ms))

print("TOTAL SAMPLE TIME FAST:",np.sum(list_time_fs))
print("TOTAL SAMPLE TIME MULT:",np.sum(list_time_mfs))
print("TOTAL SAMPLE TIME MULC:",np.sum(list_time_mfsc))
print("TOTAL SAMPLE TIME TRUE/KERNEL:",np.sum(list_time_tk))
print("TOTAL SAMPLE TIME MONTECARLO:",np.sum(list_time_ms))

print("AVERAGE ERROR FAST:",np.mean(mean_error_rate_fs))
print("AVERAGE ERROR MULT:",np.mean(mean_error_rate_mfs))
print("AVERAGE ERROR MULC:",np.mean(mean_error_rate_mfsc))
print("AVERAGE ERROR TRUE:",np.mean(mean_error_rate_ts))
print("AVERAGE ERROR KERNEL:",np.mean(mean_error_rate_ks))
print("AVERAGE ERROR MONTECARLO:",np.mean(mean_error_rate_ms))

#%% print results to file
with open(f'metrics/{dataset}_results_Constraint_{alpha}.txt', 'a+') as f:
    print("AVERAGE L2 - MEAN - TRUE-FAST:", np.mean(L2_distances_tf), file=f)
    print("AVERAGE L2 - MEAN - TRUE-MULT:", np.mean(L2_distances_tmf), file=f)
    print("AVERAGE L2 - MEAN - TRUE-MULC:", np.mean(L2_distances_tmfc), file=f)
    print("AVERAGE L2 - MEAN - TRUE-KERN:", np.mean(L2_distances_tk), file=f)
    print("AVERAGE L2 - MEAN - TRUE-MONT:", np.mean(L2_distances_tm), file=f)
    print("AVERAGE L2 - ERR - TRUE-FAST:", np.mean(L2_distances_tf_err), file=f)
    print("AVERAGE L2 - ERR - TRUE-MULT:", np.mean(L2_distances_tmf_err), file=f)
    print("AVERAGE L2 - ERR - TRUE-MULC:", np.mean(L2_distances_tmfc_err), file=f)
    print("AVERAGE L2 - ERR - TRUE-KERN:", np.mean(L2_distances_tk_err), file=f)
    print("AVERAGE L2 - ERR - TRUE-MONT:", np.mean(L2_distances_tm_err), file=f)

    print("AVERAGE L1 - MEAN - TRUE-FAST:", np.mean(L1_distances_tf), file=f)
    print("AVERAGE L1 - MEAN - TRUE-MULT:", np.mean(L1_distances_tmf), file=f)
    print("AVERAGE L1 - MEAN - TRUE-MULC:", np.mean(L1_distances_tmfc), file=f)
    print("AVERAGE L1 - MEAN - TRUE-KERN:", np.mean(L1_distances_tk), file=f)
    print("AVERAGE L1 - MEAN - TRUE-MONT:", np.mean(L1_distances_tm), file=f)
    print("AVERAGE L1 - ERR - TRUE-FAST:", np.mean(L1_distances_tf_err), file=f)
    print("AVERAGE L1 - ERR - TRUE-MULT:", np.mean(L1_distances_tmf_err), file=f)
    print("AVERAGE L1 - ERR - TRUE-MULC:", np.mean(L1_distances_tmfc_err), file=f)
    print("AVERAGE L1 - ERR - TRUE-KERN:", np.mean(L1_distances_tk_err), file=f)
    print("AVERAGE L1 - ERR - TRUE-MONT:", np.mean(L1_distances_tm_err), file=f)

    print("AVERAGE SAMPLE TIME FAST:", np.mean(list_time_fs), file=f)
    print("AVERAGE SAMPLE TIME MULT:", np.mean(list_time_mfs), file=f)
    print("AVERAGE SAMPLE TIME MULC:", np.mean(list_time_mfsc), file=f)
    print("AVERAGE SAMPLE TIME TRUE/KERNEL:", np.mean(list_time_tk), file=f)
    print("AVERAGE SAMPLE TIME MONTECARLO:", np.mean(list_time_ms), file=f)

    print("TOTAL SAMPLE TIME FAST:", np.sum(list_time_fs), file=f)
    print("TOTAL SAMPLE TIME MULT:", np.sum(list_time_mfs), file=f)
    print("TOTAL SAMPLE TIME MULC:", np.sum(list_time_mfsc), file=f)
    print("TOTAL SAMPLE TIME TRUE/KERNEL:", np.sum(list_time_tk), file=f)
    print("TOTAL SAMPLE TIME MONTECARLO:", np.sum(list_time_ms), file=f)

    print("AVERAGE ERROR FAST:", np.mean(mean_error_rate_fs), file=f)
    print("AVERAGE ERROR MULT:", np.mean(mean_error_rate_mfs), file=f)
    print("AVERAGE ERROR MULC:", np.mean(mean_error_rate_mfsc), file=f)
    print("AVERAGE ERROR TRUE:", np.mean(mean_error_rate_ts), file=f)
    print("AVERAGE ERROR KERNEL:", np.mean(mean_error_rate_ks), file=f)
    print("AVERAGE ERROR MONTECARLO:", np.mean(mean_error_rate_ms), file=f)
f.close()

#%%
average_fs=np.array(average_fs)
average_mfs=np.array(average_mfs)
average_mfsc=np.array(average_mfsc)
average_ts=np.array(average_ts)
average_ks=np.array(average_ks)
average_ms=np.array(average_ms)
average_fs_err=np.array(average_fs_err)
average_mfs_err=np.array(average_mfs_err)
average_mfsc_err=np.array(average_mfsc_err)
average_ts_err=np.array(average_ts_err)
average_ks_err=np.array(average_ks_err)
average_ms_err=np.array(average_ms_err)

m1=np.mean(average_ts,axis=0)
m2=np.mean(average_fs,axis=0)
m3=np.mean(average_ks,axis=0)
m4=np.mean(average_ms,axis=0)
m5=np.mean(average_mfs,axis=0)
m6=np.mean(average_mfsc,axis=0)
e1=np.mean(average_ts_err,axis=0)
e2=np.mean(average_fs_err,axis=0)
e3=np.mean(average_ks_err,axis=0)
e4=np.mean(average_ms_err,axis=0)
e5=np.mean(average_mfs_err,axis=0)
e6=np.mean(average_mfsc_err,axis=0)

#%% PLOT GLOBAL SV
plt.figure(figsize=(16, 9))
width = 0.75
kernelshap_iters = 128
N=6
error_kw=dict(lw=3, capsize=10, capthick=3)

plt.bar(np.arange(num_features) - 2.5*width/N, m1, width / N, label='Interval True SHAP',  yerr=e1, error_kw=error_kw, color='tab:green')
plt.bar(np.arange(num_features) - 1.5*width/N, m2, width / N, label='Interval FastSHAP',  yerr=e2, error_kw=error_kw, color='tab:orange')
plt.bar(np.arange(num_features) - 0.5*width/N, m5, width / N, label='Interval MultiFastSHAP',  yerr=e5, error_kw=error_kw, color='tab:olive')
plt.bar(np.arange(num_features) + 0.5*width/N, m6, width / N, label='Interval MultiFastSHAP-C',  yerr=e6, error_kw=error_kw, color='tab:brown')
plt.bar(np.arange(num_features) + 1.5*width/N, m3, width / N, label='Interval KernelSHAP', yerr=e3, error_kw=error_kw, color='tab:red')
plt.bar(np.arange(num_features) + 2.5*width/N, m4, width / N, label='Interval MonteCarlo',  yerr=e4, error_kw=error_kw, color='tab:purple')

plt.legend(fontsize=16)
plt.tick_params(labelsize=14)
plt.ylabel('SHAP Values', fontsize=16)
plt.xticks(np.arange(num_features), feature_names, rotation=35, rotation_mode='anchor', ha='right')
plt.savefig(f'plots/{dataset} Global Interval SV comparison - Constraint_{alpha}.jpg', bbox_inches = "tight")

plt.show()

#%% PLOT GLOBAL SV
plt.figure(figsize=(16, 9))
width = 0.75
kernelshap_iters = 128
N=4
error_kw=dict(lw=3, capsize=10, capthick=3)

plt.bar(np.arange(num_features) - 1.5*width/N, m1, width / N, label='Interval True SHAP',  yerr=e1, error_kw=error_kw, color='tab:green')
plt.bar(np.arange(num_features) - 0.5*width/N, m2, width / N, label='Interval FastSHAP',  yerr=e2, error_kw=error_kw, color='tab:orange')
plt.bar(np.arange(num_features) + 0.5*width/N, m5, width / N, label='Interval MultiFastSHAP',  yerr=e5, error_kw=error_kw, color='tab:olive')
plt.bar(np.arange(num_features) + 1.5*width/N, m6, width / N, label='Interval MultiFastSHAP-C',  yerr=e6, error_kw=error_kw, color='tab:brown')

plt.legend(fontsize=16)
plt.tick_params(labelsize=14)
plt.ylabel('SHAP Values', fontsize=16)
plt.xticks(np.arange(num_features), feature_names, rotation=35, rotation_mode='anchor', ha='right')
plt.savefig(f'plots/{dataset} Global Interval SV comparison_Constraint {alpha}_TMC.jpg', bbox_inches = "tight")

plt.show()

#%% PLOT GLOBAL SV - REPRODUCIBILITY
plt.figure(figsize=(16, 9))
width = 0.75
kernelshap_iters = 128
N=4
error_kw=dict(lw=3, capsize=10, capthick=3)

plt.bar(np.arange(num_features) - 1.5*width/N, m1, width / N, label='Interval True SHAP',  yerr=e1, error_kw=error_kw, color='tab:green')
plt.bar(np.arange(num_features) - 0.5*width/N, m2, width / N, label='Interval FastSHAP',  yerr=e2, error_kw=error_kw, color='tab:orange')
plt.bar(np.arange(num_features) + 0.5*width/N, m3, width / N, label='Interval KernelSHAP', yerr=e3, error_kw=error_kw, color='tab:red')
plt.bar(np.arange(num_features) + 1.5*width/N, m4, width / N, label='Interval MonteCarlo',  yerr=e4, error_kw=error_kw, color='tab:purple')

plt.legend(fontsize=16)
plt.tick_params(labelsize=14)
plt.ylabel('SHAP Values', fontsize=16)
plt.xticks(np.arange(num_features), feature_names, rotation=35, rotation_mode='anchor', ha='right')
# plt.savefig(f'plots/{dataset} Global Interval SV comparison - Constraint_{alpha}.jpg', bbox_inches = "tight")
plt.show()