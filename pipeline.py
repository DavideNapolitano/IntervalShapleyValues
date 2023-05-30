
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
from scipy.stats import t as t_statistic

#%% DATASET
X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset = Datasets.Monks()

#%% ORIGINAL MODEL
modelRF = RandomForestClassifier(random_state=0)
modelRF.fit(X_train_s, Y_train)

om_VV=OriginalModelVV(modelRF)

#%%
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#%% NN MODELs
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


def Confidence_Interval(x, debug=False, confidence_level=0.95):
    N_runs=len(x)
    t_val=t_statistic.ppf((confidence_level+1)/2,df=N_runs-1)
    avg=np.mean(x)
    std=np.std(x,ddof=1)
    ci=t_val*std/np.sqrt(N_runs)
    re=2*ci/avg
    if debug:
        print(f"T: {t_val}, AVG: {avg}, STD: {std}")
        print(f"CI: {ci}")
        print(f"Lower: {avg-ci}, Avg: {avg}, Upper: {avg+ci}, RE: {re}")
    return avg, ci

#%% LOOP SEEDS
for alpha in [0.001]:#, 0.002, 0.003, 0.004,  0.005]:
    print('-'*100)
    print(f'alpha={alpha}')

    L2_distances_tf_list=[]
    L2_distances_tmf_list=[]
    L2_distances_tmfc_list=[]
    L2_distances_tmfc2_list=[]
    L2_distances_tk_list=[]
    L2_distances_tm_list=[]

    L2_distances_tf_err_list=[]
    L2_distances_tmf_err_list=[]
    L2_distances_tmfc_err_list=[]
    L2_distances_tmfc2_err_list=[]
    L2_distances_tk_err_list=[]
    L2_distances_tm_err_list=[]

    L1_distances_tf_list=[]
    L1_distances_tmf_list=[]
    L1_distances_tmfc_list=[]
    L1_distances_tmfc2_list=[]
    L1_distances_tk_list=[]
    L1_distances_tm_list=[]

    L1_distances_tf_err_list=[]
    L1_distances_tmf_err_list=[]
    L1_distances_tmfc_err_list=[]
    L1_distances_tmfc2_err_list=[]
    L1_distances_tk_err_list=[]
    L1_distances_tm_err_list=[]

    list_time_fs_list=[]
    list_time_mfs_list=[]
    list_time_mfsc_list=[]
    list_time_mfsc2_list=[]
    list_time_tk_list=[]
    list_time_ms_list=[]

    mean_error_rate_fs_list=[]
    mean_error_rate_mfs_list=[]
    mean_error_rate_mfsc_list=[]
    mean_error_rate_mfsc2_list=[]
    mean_error_rate_ts_list=[]
    mean_error_rate_ks_list = []
    mean_error_rate_ms_list=[]

    for seed in [29, 12, 97, 2912, 291297]:
        print(f'\tseed={seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        def original_model_VV(x):
            pred1, pred2 = om_VV.get_pred_VV(x.cpu().numpy())  # MODELLO ORIGINALE, PRED ALWAYS ON POSITIVE CLASS
            return torch.tensor(pred1, dtype=torch.float32, device=x.device), torch.tensor(pred2, dtype=torch.float32,
                                                                                           device=x.device)

        surr_VV = MultiTaskModel(512).to(device)  # 512
        surrogate_VV = Surrogate_VV(surr_VV, num_features)  # model, 12

        print('\t\tTraining surrogate model')
        surrogate_VV.train_original_model_VV(
            X_train_s,
            X_val_s,
            original_model_VV,
            batch_size=8,
            max_epochs=200,
            loss_fn1=KLDivLoss(),  # nn.MSELoss(),
            # loss_fn2=KLDivLoss(),#nn.MSELoss(),#CrossEntropyLoss(),
            alpha=1,
            beta=1,
            validation_samples=10,
            validation_batch_size=10000,
            verbose=False,
            lr=1e-4,
            min_lr=1e-6,
            lr_factor=0.5,
            weight_decay=0.01,
            debug=False,
            training_seed=seed,
            lookback=20,
        )

        # TRAINING FASTSHAP1
        LAYER_SIZE = 512
        explainer1 = nn.Sequential(
            nn.Linear(num_features, LAYER_SIZE),
            nn.LeakyReLU(inplace=True),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(inplace=True),
            nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)

        # Set up FastSHAP object
        fastshap1 = FastSHAP(explainer1, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

        # Train
        print('\t\tTraining FastSHAP1')
        fastshap1.train(
            X_train_s,
            X_val_s,
            batch_size=8,
            num_samples=8,  ##############
            max_epochs=400,  # 200
            vector=1,
            validation_samples=128,
            verbose=False,
            paired_sampling=True,
            approx_null=True,
            lr=1e-2,
            min_lr=1e-5,
            lr_factor=0.5,
            weight_decay=0.05,
            training_seed=seed,
            lookback=20,
            debug=False,
        )  ########################################à

        # TRAINING FASTSHAP2
        LAYER_SIZE = 512
        explainer2 = nn.Sequential(
            nn.Linear(num_features, LAYER_SIZE),
            nn.LeakyReLU(inplace=True),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.LeakyReLU(inplace=True),
            nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)

        # Set up FastSHAP object
        fastshap2 = FastSHAP(explainer2, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

        # Train
        print('\t\tTraining FastSHAP2')
        fastshap2.train(
            X_train_s,
            X_val_s,
            batch_size=8,
            num_samples=8,
            max_epochs=400,  # 200
            vector=2,
            validation_samples=128,
            verbose=False,
            paired_sampling=True,
            approx_null=True,
            lr=1e-2,
            min_lr=1e-5,
            lr_factor=0.5,
            weight_decay=0.05,
            training_seed=seed,
            lookback=20,
            debug=False)

        # TRAINING FASTSHAP3
        explainer3 = MultiTaskExplainer(512).to(device)
        fastshap3 = MultiFastSHAP(explainer3, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

        # Train
        print('\t\tTraining FastSHAP3')
        fastshap3.train(
            X_train_s,
            X_val_s,
            batch_size=8,
            num_samples=8,  ##############
            max_epochs=400,  # 200
            validation_samples=128,
            verbose=False,
            paired_sampling=True,
            approx_null=True,
            lr=1e-2,
            min_lr=1e-5,
            lr_factor=0.5,
            weight_decay=0.05,
            training_seed=seed,
            lookback=20,
            debug=False,
            constraint=-1
        )  ########################################à

        # TRAINING FASTSHAP4
        explainer4 = MultiTaskExplainer(512).to(device)
        fastshap4 = MultiFastSHAP(explainer4, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

        # Train
        print('\t\tTraining FastSHAP4')
        fastshap4.train(
            X_train_s,
            X_val_s,
            batch_size=8,
            num_samples=8,  ##############
            max_epochs=400,  # 200
            validation_samples=128,
            verbose=False,
            paired_sampling=True,
            approx_null=True,
            lr=1e-2,
            min_lr=1e-5,
            lr_factor=0.5,
            weight_decay=0.05,
            training_seed=seed,
            lookback=20,
            debug_val=False,
            debug=False,
            constraint=1,
            alpha=alpha
        )  ########################################à

        # TRAINING FASTSHAP5
        explainer5 = MultiTaskExplainer(512).to(device)
        fastshap5 = MultiFastSHAP(explainer5, surrogate_VV, normalization="additive", link=nn.Softmax(dim=-1))

        # Train
        print('\t\tTraining FastSHAP5')
        fastshap5.train(
            X_train_s,
            X_val_s,
            batch_size=8,
            num_samples=8,  ##############
            max_epochs=400,  # 200
            validation_samples=128,
            verbose=False,
            paired_sampling=True,
            approx_null=True,
            lr=1e-2,
            min_lr=1e-5,
            lr_factor=0.5,
            weight_decay=0.05,
            training_seed=seed,
            lookback=20,
            debug_val=False,
            debug=False,
            constraint=2,
            alpha=alpha
        )

        # COMPUTE METRICS
        print('\t\tComputing metrics')
        average_fs, average_mfs, average_mfsc, average_mfsc2, average_ts, average_ks, average_ms, \
        average_fs_err, average_mfs_err, average_mfsc_err, average_mfsc2_err, average_ts_err, average_ks_err, average_ms_err, \
        mean_error_rate_fs, mean_error_rate_mfs, mean_error_rate_mfsc, mean_error_rate_mfsc2, mean_error_rate_ts, mean_error_rate_ks, mean_error_rate_ms, \
        list_time_fs, list_time_mfs, list_time_mfsc, list_time_mfsc2, list_time_tk, list_time_ms, \
        L1_distances_tf, L1_distances_tmf, L1_distances_tmfc, L1_distances_tmfc2, L1_distances_tk, L1_distances_tm, \
        L1_distances_tf_err, L1_distances_tmf_err, L1_distances_tmfc_err, L1_distances_tmfc2_err, L1_distances_tk_err, L1_distances_tm_err, \
        L2_distances_tm, L2_distances_tm_err, L2_distances_tmf, L2_distances_tmf_err, L2_distances_tf, L2_distances_tf_err, \
        L2_distances_tmfc, L2_distances_tmfc2, L2_distances_tmfc_err, L2_distances_tmfc2_err, L2_distances_tk, L2_distances_tk_err \
                                    = compute_metrics(X_train_s, feature_names, Y_train, fastshap1, fastshap2, fastshap3, fastshap4, fastshap5, dataset, device, surrogate_VV, modelRF, om_VV, seed, alpha, SAVE=True)


        print("\t\tCompute Statistics")
        L2_distances_tf_list.append(np.mean(L2_distances_tf))
        L2_distances_tmf_list.append(np.mean(L2_distances_tmf))
        L2_distances_tmfc_list.append(np.mean(L2_distances_tmfc))
        L2_distances_tmfc2_list.append(np.mean(L2_distances_tmfc2))
        L2_distances_tk_list.append(np.mean(L2_distances_tk))
        L2_distances_tm_list.append(np.mean(L2_distances_tm))

        L2_distances_tf_err_list.append(np.mean(L2_distances_tf_err))
        L2_distances_tmf_err_list.append(np.mean(L2_distances_tmf_err))
        L2_distances_tmfc_err_list.append(np.mean(L2_distances_tmfc_err))
        L2_distances_tmfc2_err_list.append(np.mean(L2_distances_tmfc2_err))
        L2_distances_tk_err_list.append(np.mean(L2_distances_tk_err))
        L2_distances_tm_err_list.append(np.mean(L2_distances_tm_err))

        L1_distances_tf_list.append(np.mean(L1_distances_tf))
        L1_distances_tmf_list.append(np.mean(L1_distances_tmf))
        L1_distances_tmfc_list.append(np.mean(L1_distances_tmfc))
        L1_distances_tmfc2_list.append(np.mean(L1_distances_tmfc2))
        L1_distances_tk_list.append(np.mean(L1_distances_tk))
        L1_distances_tm_list.append(np.mean(L1_distances_tm))

        L1_distances_tf_err_list.append(np.mean(L1_distances_tf_err))
        L1_distances_tmf_err_list.append(np.mean(L1_distances_tmf_err))
        L1_distances_tmfc_err_list.append(np.mean(L1_distances_tmfc_err))
        L1_distances_tmfc2_err_list.append(np.mean(L1_distances_tmfc2_err))
        L1_distances_tk_err_list.append(np.mean(L1_distances_tk_err))
        L1_distances_tm_err_list.append(np.mean(L1_distances_tm_err))

        list_time_fs_list.append(np.mean(list_time_fs))
        list_time_mfs_list.append(np.mean(list_time_mfs))
        list_time_mfsc_list.append(np.mean(list_time_mfsc))
        list_time_mfsc2_list.append(np.mean(list_time_mfsc2))
        list_time_tk_list.append(np.mean(list_time_tk))
        list_time_ms_list.append(np.mean(list_time_ms))

        mean_error_rate_fs_list.append(np.mean(mean_error_rate_fs))
        mean_error_rate_mfs_list.append(np.mean(mean_error_rate_mfs))
        mean_error_rate_mfsc_list.append(np.mean(mean_error_rate_mfsc))
        mean_error_rate_mfsc2_list.append(np.mean(mean_error_rate_mfsc2))
        mean_error_rate_ts_list.append(np.mean(mean_error_rate_ts))
        mean_error_rate_ks_list.append(np.mean(mean_error_rate_ks))
        mean_error_rate_ms_list.append(np.mean(mean_error_rate_ms))


        with open(f'pipeline/metrics/{dataset}_results_Constraint_alpha={alpha}_seed={seed}.txt', 'a+') as f:
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
            print("AVERAGE SAMPLE TIME MLC2:", np.mean(list_time_mfsc2), file=f)
            print("AVERAGE SAMPLE TIME TRUE/KERNEL:", np.mean(list_time_tk), file=f)
            print("AVERAGE SAMPLE TIME MONTECARLO:", np.mean(list_time_ms), file=f)

            print("AVERAGE ERROR FAST:", np.mean(mean_error_rate_fs), file=f)
            print("AVERAGE ERROR MULT:", np.mean(mean_error_rate_mfs), file=f)
            print("AVERAGE ERROR MULC:", np.mean(mean_error_rate_mfsc), file=f)
            print("AVERAGE ERROR TRUE:", np.mean(mean_error_rate_ts), file=f)
            print("AVERAGE ERROR KERNEL:", np.mean(mean_error_rate_ks), file=f)
            print("AVERAGE ERROR MONTECARLO:", np.mean(mean_error_rate_ms), file=f)
        f.close()


    L2_distances_tf_res=Confidence_Interval(L2_distances_tf_list)
    L2_distances_tmf_res=Confidence_Interval(L2_distances_tmf_list)
    L2_distances_tmfc_res=Confidence_Interval(L2_distances_tmfc_list)
    L2_distances_tmfc2_res=Confidence_Interval(L2_distances_tmfc2_list)
    L2_distances_tk_res=Confidence_Interval(L2_distances_tk_list)
    L2_distances_tm_res=Confidence_Interval(L2_distances_tm_list)

    L2_distances_tf_err_res=Confidence_Interval(L2_distances_tf_err_list)
    L2_distances_tmf_err_res=Confidence_Interval(L2_distances_tmf_err_list)
    L2_distances_tmfc_err_res=Confidence_Interval(L2_distances_tmfc_err_list)
    L2_distances_tmfc2_err_res=Confidence_Interval(L2_distances_tmfc2_err_list)
    L2_distances_tk_err_res=Confidence_Interval(L2_distances_tk_err_list)
    L2_distances_tm_err_res=Confidence_Interval(L2_distances_tm_err_list)

    L1_distances_tf_res=Confidence_Interval(L1_distances_tf_list)
    L1_distances_tmf_res=Confidence_Interval(L1_distances_tmf_list)
    L1_distances_tmfc_res=Confidence_Interval(L1_distances_tmfc_list)
    L1_distances_tmfc2_res=Confidence_Interval(L1_distances_tmfc2_list)
    L1_distances_tk_res=Confidence_Interval(L1_distances_tk_list)
    L1_distances_tm_res=Confidence_Interval(L1_distances_tm_list)

    L1_distances_tf_err_res=Confidence_Interval(L1_distances_tf_err_list)
    L1_distances_tmf_err_res=Confidence_Interval(L1_distances_tmf_err_list)
    L1_distances_tmfc_err_res=Confidence_Interval(L1_distances_tmfc_err_list)
    L1_distances_tmfc2_err_res=Confidence_Interval(L1_distances_tmfc2_err_list)
    L1_distances_tk_err_res=Confidence_Interval(L1_distances_tk_err_list)
    L1_distances_tm_err_res=Confidence_Interval(L1_distances_tm_err_list)

    list_time_fs_res=Confidence_Interval(list_time_fs_list)
    list_time_mfs_res=Confidence_Interval(list_time_mfs_list)
    list_time_mfsc_res=Confidence_Interval(list_time_mfsc_list)
    list_time_mfsc2_res=Confidence_Interval(list_time_mfsc2_list)
    list_time_tk_res=Confidence_Interval(list_time_tk_list)
    list_time_ms_res=Confidence_Interval(list_time_ms_list)

    mean_error_rate_fs_res=Confidence_Interval(mean_error_rate_fs_list)
    mean_error_rate_mfs_res=Confidence_Interval(mean_error_rate_mfs_list)
    mean_error_rate_mfsc_res=Confidence_Interval(mean_error_rate_mfsc_list)
    mean_error_rate_mfsc2_res=Confidence_Interval(mean_error_rate_mfsc2_list)
    mean_error_rate_ts_res=Confidence_Interval(mean_error_rate_ts_list)
    mean_error_rate_ks_res=Confidence_Interval(mean_error_rate_ks_list)
    mean_error_rate_ms_res=Confidence_Interval(mean_error_rate_ms_list)

    print('#'*100)
    print("AVERAGE L2 - MEAN - TRUE-FAST:",L2_distances_tf_res[0],'±',L2_distances_tf_res[1])
    print("AVERAGE L2 - MEAN - TRUE-MULT:",L2_distances_tmf_res[0],'±',L2_distances_tmf_res[1])
    print("AVERAGE L2 - MEAN - TRUE-MULC:",L2_distances_tmfc_res[0],'±',L2_distances_tmfc_res[1])
    print("AVERAGE L2 - MEAN - TRUE-MLC2:", L2_distances_tmfc2_res[0],'±',L2_distances_tmfc2_res[1])
    print("AVERAGE L2 - MEAN - TRUE-KERN:",L2_distances_tk_res[0],'±',L2_distances_tk_res[1])
    print("AVERAGE L2 - MEAN - TRUE-MONT:",L2_distances_tm_res[0],'±',L2_distances_tm_res[1])
    print('-'*100)
    print("AVERAGE L2 - ERR - TRUE-FAST:",L2_distances_tf_err_res[0],'±',L2_distances_tf_err_res[1])
    print("AVERAGE L2 - ERR - TRUE-MULT:",L2_distances_tmf_err_res[0],'±',L2_distances_tmf_err_res[1])
    print("AVERAGE L2 - ERR - TRUE-MULC:",L2_distances_tmfc_err_res[0],'±',L2_distances_tmfc_err_res[1])
    print("AVERAGE L2 - ERR - TRUE-MLC2:", L2_distances_tmfc2_err_res[0],'±',L2_distances_tmfc2_err_res[1])
    print("AVERAGE L2 - ERR - TRUE-KERN:",L2_distances_tk_err_res[0],'±',L2_distances_tk_err_res[1])
    print("AVERAGE L2 - ERR - TRUE-MONT:",L2_distances_tm_err_res[0],'±',L2_distances_tm_err_res[1])
    print('#'*100)
    print("AVERAGE L1 - MEAN - TRUE-FAST:", L1_distances_tf_res[0],'±',L1_distances_tf_res[1])
    print("AVERAGE L1 - MEAN - TRUE-MULT:", L1_distances_tmf_res[0],'±',L1_distances_tmf_res[1])
    print("AVERAGE L1 - MEAN - TRUE-MULC:", L1_distances_tmfc_res[0],'±',L1_distances_tmfc_res[1])
    print("AVERAGE L1 - MEAN - TRUE-MLC2:", L1_distances_tmfc2_res[0],'±',L1_distances_tmfc2_res[1])
    print("AVERAGE L1 - MEAN - TRUE-KERN:", L1_distances_tk_res[0],'±',L1_distances_tk_res[1])
    print("AVERAGE L1 - MEAN - TRUE-MONT:", L1_distances_tm_res[0],'±',L1_distances_tm_res[1])
    print('-'*100)
    print("AVERAGE L1 - ERR - TRUE-FAST:",L1_distances_tf_err_res[0],'±',L1_distances_tf_err_res[1])
    print("AVERAGE L1 - ERR - TRUE-MULT:",L1_distances_tmf_err_res[0],'±',L1_distances_tmf_err_res[1])
    print("AVERAGE L1 - ERR - TRUE-MULC:",L1_distances_tmfc_err_res[0],'±',L1_distances_tmfc_err_res[1])
    print("AVERAGE L1 - ERR - TRUE-MLC2:", L1_distances_tmfc2_err_res[0],'±',L1_distances_tmfc2_err_res[1])
    print("AVERAGE L1 - ERR - TRUE-KERN:",L1_distances_tk_err_res[0],'±',L1_distances_tk_err_res[1])
    print("AVERAGE L1 - ERR - TRUE-MONT:",L1_distances_tm_err_res[0],'±',L1_distances_tm_err_res[1])
    print('#'*100)
    print("AVERAGE SAMPLE TIME FAST:", list_time_fs_res[0],'±',list_time_fs_res[1])
    print("AVERAGE SAMPLE TIME MULT:", list_time_mfs_res[0],'±',list_time_mfs_res[1])
    print("AVERAGE SAMPLE TIME MULC:", list_time_mfsc_res[0],'±',list_time_mfsc_res[1])
    print("AVERAGE SAMPLE TIME MLC2:", list_time_mfsc2_res[0],'±',list_time_mfsc2_res[1])
    print("AVERAGE SAMPLE TIME TRUE/KERNEL:", list_time_tk_res[0],'±',list_time_tk_res[1])
    print("AVERAGE SAMPLE TIME MONTECARLO:", list_time_ms_res[0],'±',list_time_ms_res[1])
    print('#'*100)
    print("AVERAGE ERROR FAST:",mean_error_rate_fs_res[0],'±',mean_error_rate_fs_res[1])
    print("AVERAGE ERROR MULT:",mean_error_rate_mfs_res[0],'±',mean_error_rate_mfs_res[1])
    print("AVERAGE ERROR MULC:",mean_error_rate_mfsc_res[0],'±',mean_error_rate_mfsc_res[1])
    print("AVERAGE ERROR MLC2:",mean_error_rate_mfsc2_res[0],'±',mean_error_rate_mfsc2_res[1])
    print("AVERAGE ERROR TRUE:",mean_error_rate_ts_res[0],'±',mean_error_rate_ts_res[1])
    print("AVERAGE ERROR KERNEL:",mean_error_rate_ks_res[0],'±',mean_error_rate_ks_res[1])
    print("AVERAGE ERROR MONTECARLO:",mean_error_rate_ms_res[0],'±',mean_error_rate_ms_res[1])
    print('#'*100)

    with open(f'pipeline/results/{dataset}_results_Constraint_alpha={alpha}_CI.txt', 'a+') as f:
        print('#' * 100, file=f)
        print("NAME\tMEAN\tCI",  file=f)
        print("AVERAGE L2 - MEAN - TRUE-FAST:\t", L2_distances_tf_res[0], '\t', L2_distances_tf_res[1], file=f)
        print("AVERAGE L2 - MEAN - TRUE-MULT:\t", L2_distances_tmf_res[0], '\t', L2_distances_tmf_res[1], file=f)
        print("AVERAGE L2 - MEAN - TRUE-MULC:\t", L2_distances_tmfc_res[0], '\t', L2_distances_tmfc_res[1], file=f)
        print("AVERAGE L2 - MEAN - TRUE-MLC2:\t", L2_distances_tmfc2_res[0], '\t', L2_distances_tmfc2_res[1], file=f)
        print("AVERAGE L2 - MEAN - TRUE-KERN:\t", L2_distances_tk_res[0], '\t', L2_distances_tk_res[1], file=f)
        print("AVERAGE L2 - MEAN - TRUE-MONT:\t", L2_distances_tm_res[0], '\t', L2_distances_tm_res[1], file=f)
        print('-' * 100, file=f)
        print("AVERAGE L2 - ERR - TRUE-FAST:\t", L2_distances_tf_err_res[0], '\t', L2_distances_tf_err_res[1], file=f)
        print("AVERAGE L2 - ERR - TRUE-MULT:\t", L2_distances_tmf_err_res[0], '\t', L2_distances_tmf_err_res[1], file=f)
        print("AVERAGE L2 - ERR - TRUE-MULC:\t", L2_distances_tmfc_err_res[0], '\t', L2_distances_tmfc_err_res[1],
              file=f)
        print("AVERAGE L2 - ERR - TRUE-MLC2:\t", L2_distances_tmfc2_err_res[0], '\t', L2_distances_tmfc2_err_res[1],
              file=f)
        print("AVERAGE L2 - ERR - TRUE-KERN:\t", L2_distances_tk_err_res[0], '\t', L2_distances_tk_err_res[1], file=f)
        print("AVERAGE L2 - ERR - TRUE-MONT:\t", L2_distances_tm_err_res[0], '\t', L2_distances_tm_err_res[1], file=f)
        print('#' * 100, file=f)
        print("AVERAGE L1 - MEAN - TRUE-FAST:\t", L1_distances_tf_res[0], '\t', L1_distances_tf_res[1], file=f)
        print("AVERAGE L1 - MEAN - TRUE-MULT:\t", L1_distances_tmf_res[0], '\t', L1_distances_tmf_res[1], file=f)
        print("AVERAGE L1 - MEAN - TRUE-MULC:\t", L1_distances_tmfc_res[0], '\t', L1_distances_tmfc_res[1], file=f)
        print("AVERAGE L1 - MEAN - TRUE-MLC2:\t", L1_distances_tmfc2_res[0], '\t', L1_distances_tmfc2_res[1], file=f)
        print("AVERAGE L1 - MEAN - TRUE-KERN:\t", L1_distances_tk_res[0], '\t', L1_distances_tk_res[1], file=f)
        print("AVERAGE L1 - MEAN - TRUE-MONT:\t", L1_distances_tm_res[0], '\t', L1_distances_tm_res[1], file=f)
        print('-' * 100, file=f)
        print("AVERAGE L1 - ERR - TRUE-FAST:\t", L1_distances_tf_err_res[0], '\t', L1_distances_tf_err_res[1], file=f)
        print("AVERAGE L1 - ERR - TRUE-MULT:\t", L1_distances_tmf_err_res[0], '\t', L1_distances_tmf_err_res[1], file=f)
        print("AVERAGE L1 - ERR - TRUE-MULC:\t", L1_distances_tmfc_err_res[0], '\t', L1_distances_tmfc_err_res[1],
              file=f)
        print("AVERAGE L1 - ERR - TRUE-MLC2:\t", L1_distances_tmfc2_err_res[0], '\t', L1_distances_tmfc2_err_res[1],
              file=f)
        print("AVERAGE L1 - ERR - TRUE-KERN:\t", L1_distances_tk_err_res[0], '\t', L1_distances_tk_err_res[1], file=f)
        print("AVERAGE L1 - ERR - TRUE-MONT:\t", L1_distances_tm_err_res[0], '\t', L1_distances_tm_err_res[1], file=f)
        print('#' * 100, file=f)
        print("AVERAGE SAMPLE TIME FAST:\t", list_time_fs_res[0], '\t', list_time_fs_res[1], file=f)
        print("AVERAGE SAMPLE TIME MULT:\t", list_time_mfs_res[0], '\t', list_time_mfs_res[1], file=f)
        print("AVERAGE SAMPLE TIME MULC:\t", list_time_mfsc_res[0], '\t', list_time_mfsc_res[1], file=f)
        print("AVERAGE SAMPLE TIME MLC2:\t", list_time_mfsc2_res[0], '\t', list_time_mfsc2_res[1], file=f)
        print("AVERAGE SAMPLE TIME TRUE/KERNEL:\t", list_time_tk_res[0], '\t', list_time_tk_res[1], file=f)
        print("AVERAGE SAMPLE TIME MONTECARLO:\t", list_time_ms_res[0], '\t', list_time_ms_res[1], file=f)
        print('#' * 100, file=f)
        print("AVERAGE ERROR FAST:\t", mean_error_rate_fs_res[0], '\t', mean_error_rate_fs_res[1], file=f)
        print("AVERAGE ERROR MULT:\t", mean_error_rate_mfs_res[0], '\t', mean_error_rate_mfs_res[1], file=f)
        print("AVERAGE ERROR MULC:\t", mean_error_rate_mfsc_res[0], '\t', mean_error_rate_mfsc_res[1], file=f)
        print("AVERAGE ERROR MLC2:\t", mean_error_rate_mfsc2_res[0], '\t', mean_error_rate_mfsc2_res[1], file=f)
        print("AVERAGE ERROR TRUE:\t", mean_error_rate_ts_res[0], '\t', mean_error_rate_ts_res[1], file=f)
        print("AVERAGE ERROR KERNEL:\t", mean_error_rate_ks_res[0], '\t', mean_error_rate_ks_res[1], file=f)
        print("AVERAGE ERROR MONTECARLO:\t", mean_error_rate_ms_res[0], '\t', mean_error_rate_ms_res[1], file=f)
        print('#' * 100, file=f)
    f.close()