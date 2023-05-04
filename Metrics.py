import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import time
from shapreg import ShapleyRegression, PredictionGame
import pickle

def compute_metrics(DATA, FEATURE_NAMES, LABELS, fastshap1, fastshap2, fastshap3, fastshap4, dataset, device, surrogate_VV, SAVE=False):
    def imputer_lower(x, S):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        S = torch.tensor(S, dtype=torch.float32, device=device)
        pred1, pred2 = surrogate_VV(x, S)  # .softmax(dim=-1)
        pred1 = pred1.softmax(dim=-1)
        pred2 = pred2.softmax(dim=-1)
        tmp1 = pred1.detach().numpy()
        tmp2 = pred2.detach().numpy()
        # print(tmp1)
        # print(tmp2)
        tmp = []
        for index in range(len(tmp1)):
            tmp.append([tmp1[index][0], tmp2[index][0]])
        # print(tmp)
        mean = np.array(tmp)
        mean = torch.as_tensor(mean)
        # mean=mean.softmax(dim=-1)
        return mean.cpu().data.numpy()

    # Setup for KernelSHAP
    def imputer_upper(x, S):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        S = torch.tensor(S, dtype=torch.float32, device=device)
        pred1, pred2 = surrogate_VV(x, S)  # .softmax(dim=-1)
        pred1 = pred1.softmax(dim=-1)
        pred2 = pred2.softmax(dim=-1)
        tmp1 = pred1.detach().numpy()
        tmp2 = pred2.detach().numpy()
        # print(tmp1)
        # print(tmp2)
        tmp = []
        for index in range(len(tmp1)):
            tmp.append([tmp2[index][1], tmp1[index][1]])
        # print(tmp)
        mean = np.array(tmp)
        mean = torch.as_tensor(mean)
        # mean=mean.softmax(dim=-1)
        return mean.cpu().data.numpy()

    mean_error_rate_fs = []
    mean_error_rate_mfs = []
    mean_error_rate_mfsc = []
    mean_error_rate_ts = []
    mean_error_rate_ks = []
    mean_error_rate_ms = []

    L2_distances_tf = []
    L2_distances_tmf = []
    L2_distances_tmfc = []
    L2_distances_tk = []
    L2_distances_tm = []
    L2_distances_tf_err = []
    L2_distances_tmf_err = []
    L2_distances_tmfc_err = []
    L2_distances_tk_err = []
    L2_distances_tm_err = []

    L1_distances_tf = []
    L1_distances_tmf = []
    L1_distances_tmfc = []
    L1_distances_tk = []
    L1_distances_tm = []
    L1_distances_tf_err = []
    L1_distances_tmf_err = []
    L1_distances_tmfc_err = []
    L1_distances_tk_err = []
    L1_distances_tm_err = []

    average_fs = []
    average_mfs = []
    average_mfsc = []
    average_ts = []
    average_ks = []
    average_ms = []
    average_fs_err = []
    average_mfs_err = []
    average_mfsc_err = []
    average_ts_err = []
    average_ks_err = []
    average_ms_err = []

    list_time_fs = []
    list_time_mfs = []
    list_time_mfsc = []
    list_time_tk = []
    list_time_ms = []

    X_train_s_TMP = pd.DataFrame(DATA, columns=FEATURE_NAMES)
    # X_test_s_TMP=pd.DataFrame(X_test_s, columns=feature_names)
    kernelshap_iters=128

    for ind in tqdm(range(len(DATA[:100]))):
        x = DATA[ind:ind + 1]
        y = int(LABELS[ind])

        # Run FastSHAP
        t1 = time.time()
        fastshap_values1 = fastshap1.shap_values(x, vector=1)[0]
        fastshap_values2 = fastshap2.shap_values(x, vector=2)[0]
        t2 = time.time()
        list_time_fs.append(t2 - t1)

        fastshap_values_mean = []
        fastshap_values_ci = []
        error_rate_fs = 0
        for el1, el2 in zip(fastshap_values1, fastshap_values2):
            if y == 0:
                if el1[0] > el2[1]:
                    error_rate_fs += 1
            else:
                if el2[0] > el1[1]:
                    error_rate_fs += 1
            m1 = (el1[0] + el2[1]) / 2
            m2 = (el2[0] + el1[1]) / 2
            c1 = np.abs(m1 - el1[0])
            c2 = np.abs(m2 - el2[0])

            fastshap_values_mean.append([m1, m2])
            fastshap_values_ci.append([c1, c2])

        fastshap_values_mean = np.array(fastshap_values_mean)
        fastshap_values_ci = np.array(fastshap_values_ci)

        mean_error_rate_fs.append(error_rate_fs)

        # Run MultiFastSHAP
        t1 = time.time()
        multi1, multi2 = fastshap3.shap_values(x, vector=1)
        t2 = time.time()
        list_time_mfs.append(t2 - t1)
        multi1 = multi1[0, :, :]
        multi2 = multi2[0, :, :]

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

        # Run MultiFastSHAP
        t1 = time.time()
        multi1, multi2 = fastshap4.shap_values(x, vector=1)
        t2 = time.time()
        list_time_mfsc.append(t2 - t1)
        multi1 = multi1[0, :, :]
        multi2 = multi2[0, :, :]

        multifastshap_const_values_mean = []
        multifastshap_const_values_ci = []
        error_rate_mfsc = 0
        for el1, el2 in zip(multi1, multi2):
            if y == 0:
                if el1[0] > el2[1]:
                    error_rate_mfsc += 1
            else:
                if el2[0] > el1[1]:
                    error_rate_mfsc += 1
            m1 = (el1[0] + el2[1]) / 2
            m2 = (el2[0] + el1[1]) / 2
            c1 = np.abs(m1 - el1[0])
            c2 = np.abs(m2 - el2[0])

            multifastshap_const_values_mean.append([m1, m2])
            multifastshap_const_values_ci.append([c1, c2])

        multifastshap_const_values_mean = np.array(multifastshap_const_values_mean)
        multifastshap_const_values_ci = np.array(multifastshap_const_values_ci)

        mean_error_rate_mfsc.append(error_rate_mfsc)

        # Run TrueSHAP/KernelSHAP

        t1 = time.time()
        game_l = PredictionGame(imputer_lower, x)
        shap_values_l, all_results_l = ShapleyRegression(game_l, batch_size=32, paired_sampling=False,
                                                         detect_convergence=True, bar=False, return_all=True)

        game_u = PredictionGame(imputer_upper, x)
        shap_values_u, all_results_u = ShapleyRegression(game_u, batch_size=32, paired_sampling=False,
                                                         detect_convergence=True, bar=False, return_all=True)
        t2 = time.time()
        list_time_tk.append(t2 - t1)

        tmp1 = shap_values_l.values[:, y]
        tmp2 = shap_values_u.values[:, y]
        error_rate_ts = 0
        for el1, el2 in zip(tmp1, tmp2):
            # print(el1,el2)
            if el1 > el2:
                error_rate_ts += 1
        mean_error_rate_ts.append(error_rate_ts)
        mean_ts = (tmp2 + tmp1) / 2
        err_ts = np.abs(mean_ts - tmp2)

        tmp3 = all_results_l['values'][list(all_results_l['iters']).index(kernelshap_iters)][:, y]
        tmp4 = all_results_u['values'][list(all_results_u['iters']).index(kernelshap_iters)][:, y]
        error_rate_ks = 0
        for el1, el2 in zip(tmp3, tmp4):
            # print(el1,el2)
            if el1 > el2:
                error_rate_ks += 1
        mean_error_rate_ks.append(error_rate_ks)
        mean_ks = (tmp3 + tmp4) / 2
        err_ks = np.abs(mean_ks - tmp4)

        mean_fs = fastshap_values_mean[:, y]
        err_fs = fastshap_values_ci[:, y]

        mean_mfs = multifastshap_values_mean[:, y]
        err_mfs = multifastshap_values_ci[:, y]

        mean_mfsc = multifastshap_const_values_mean[:, y]
        err_mfsc = multifastshap_const_values_ci[:, y]

        # Run MonteCarlo
        t1 = time.time()
        mean_mc, err_mc, emc = 0, 0, 0  # MonteCarlo(ind, X_train_s_TMP, modelRF, om)
        t2 = time.time()
        list_time_ms.append(t2 - t1)

        mean_error_rate_ms.append(emc)

        # Compute distances
        distance_tf = np.linalg.norm(np.array(mean_ts) - np.array(mean_fs))
        distance_tmf = np.linalg.norm(np.array(mean_ts) - np.array(mean_mfs))
        distance_tmfc = np.linalg.norm(np.array(mean_ts) - np.array(mean_mfsc))
        distance_tk = np.linalg.norm(np.array(mean_ts) - np.array(mean_ks))
        distance_tm = np.linalg.norm(np.array(mean_ts) - np.array(mean_mc))
        distance_tf_err = np.linalg.norm(np.array(err_ts) - np.array(err_fs))
        distance_tmf_err = np.linalg.norm(np.array(err_ts) - np.array(err_mfs))
        distance_tmfc_err = np.linalg.norm(np.array(err_ts) - np.array(err_mfsc))
        distance_tk_err = np.linalg.norm(np.array(err_ts) - np.array(err_ks))
        distance_tm_err = np.linalg.norm(np.array(err_ts) - np.array(err_mc))

        L2_distances_tf.append(distance_tf)
        L2_distances_tmf.append(distance_tmf)
        L2_distances_tmfc.append(distance_tmfc)
        L2_distances_tk.append(distance_tk)
        L2_distances_tm.append(distance_tm)
        L2_distances_tf_err.append(distance_tf_err)
        L2_distances_tmf_err.append(distance_tmf_err)
        L2_distances_tmfc_err.append(distance_tmfc_err)
        L2_distances_tk_err.append(distance_tk_err)
        L2_distances_tm_err.append(distance_tm_err)

        distance_tf_l1 = np.linalg.norm(np.array(mean_ts) - np.array(mean_fs), ord=1)
        distance_tmf_l1 = np.linalg.norm(np.array(mean_ts) - np.array(mean_mfs), ord=1)
        distance_tmfc_l1 = np.linalg.norm(np.array(mean_ts) - np.array(mean_mfsc), ord=1)
        distance_tk_l1 = np.linalg.norm(np.array(mean_ts) - np.array(mean_ks), ord=1)
        distance_tm_l1 = np.linalg.norm(np.array(mean_ts) - np.array(mean_mc), ord=1)
        distance_tf_err_l1 = np.linalg.norm(np.array(err_ts) - np.array(err_fs), ord=1)
        distance_tmf_err_l1 = np.linalg.norm(np.array(err_ts) - np.array(err_mfs), ord=1)
        distance_tmfc_err_l1 = np.linalg.norm(np.array(err_ts) - np.array(err_mfsc), ord=1)
        distance_tk_err_l1 = np.linalg.norm(np.array(err_ts) - np.array(err_ks), ord=1)
        distance_tm_err_l1 = np.linalg.norm(np.array(err_ts) - np.array(err_mc), ord=1)

        L1_distances_tf.append(distance_tf_l1)
        L1_distances_tmf.append(distance_tmf_l1)
        L1_distances_tmfc.append(distance_tmfc_l1)
        L1_distances_tk.append(distance_tk_l1)
        L1_distances_tm.append(distance_tm_l1)
        L1_distances_tf_err.append(distance_tf_err_l1)
        L1_distances_tmf_err.append(distance_tmf_err_l1)
        L1_distances_tmfc_err.append(distance_tmfc_err_l1)
        L1_distances_tk_err.append(distance_tk_err_l1)
        L1_distances_tm_err.append(distance_tm_err_l1)

        average_fs.append(mean_fs)
        average_mfs.append(mean_mfs)
        average_mfsc.append(mean_mfsc)
        average_ts.append(mean_ts)
        average_ks.append(mean_ks)
        average_ms.append(mean_mc)
        average_fs_err.append(err_fs)
        average_mfs_err.append(err_mfs)
        average_mfsc_err.append(err_mfsc)
        average_ts_err.append(err_ts)
        average_ks_err.append(err_ks)
        average_ms_err.append(err_mc)


    if SAVE:
        with open(f'dump/{dataset}L2_distances_tf.pkl', 'wb') as f:
            pickle.dump(L2_distances_tf, f)
        with open(f'dump/{dataset}L2_distances_tmf.pkl', 'wb') as f:
            pickle.dump(L2_distances_tmf, f)
        with open(f'dump/{dataset}L2_distances_tmfc.pkl', 'wb') as f:
            pickle.dump(L2_distances_tmfc, f)
        with open(f'dump/{dataset}L2_distances_tk.pkl', 'wb') as f:
            pickle.dump(L2_distances_tk, f)
        with open(f'dump/{dataset}L2_distances_tm.pkl', 'wb') as f:
            pickle.dump(L2_distances_tm, f)
        with open(f'dump/{dataset}L2_distances_tf_err.pkl', 'wb') as f:
            pickle.dump(L2_distances_tf_err, f)
        with open(f'dump/{dataset}L2_distances_tmf_err.pkl', 'wb') as f:
            pickle.dump(L2_distances_tmf_err, f)
        with open(f'dump/{dataset}L2_distances_tmfc_err.pkl', 'wb') as f:
            pickle.dump(L2_distances_tmfc_err, f)
        with open(f'dump/{dataset}L2_distances_tk_err.pkl', 'wb') as f:
            pickle.dump(L2_distances_tk_err, f)
        with open(f'dump/{dataset}L2_distances_tm_err.pkl', 'wb') as f:
            pickle.dump(L2_distances_tm_err, f)

        with open(f'dump/{dataset}L1_distances_tf.pkl', 'wb') as f:
            pickle.dump(L1_distances_tf, f)
        with open(f'dump/{dataset}L1_distances_tmf.pkl', 'wb') as f:
            pickle.dump(L1_distances_tmf, f)
        with open(f'dump/{dataset}L1_distances_tmfc.pkl', 'wb') as f:
            pickle.dump(L1_distances_tmfc, f)
        with open(f'dump/{dataset}L1_distances_tk.pkl', 'wb') as f:
            pickle.dump(L1_distances_tk, f)
        with open(f'dump/{dataset}L1_distances_tm.pkl', 'wb') as f:
            pickle.dump(L1_distances_tm, f)
        with open(f'dump/{dataset}L1_distances_tf_err.pkl', 'wb') as f:
            pickle.dump(L1_distances_tf_err, f)
        with open(f'dump/{dataset}L1_distances_tmf_err.pkl', 'wb') as f:
            pickle.dump(L1_distances_tmf_err, f)
        with open(f'dump/{dataset}L1_distances_tmfc_err.pkl', 'wb') as f:
            pickle.dump(L1_distances_tmfc_err, f)
        with open(f'dump/{dataset}L1_distances_tk_err.pkl', 'wb') as f:
            pickle.dump(L1_distances_tk_err, f)
        with open(f'dump/{dataset}L1_distances_tm_err.pkl', 'wb') as f:
            pickle.dump(L1_distances_tm_err, f)

        with open(f'dump/{dataset}average_fs.pkl', 'wb') as f:
            pickle.dump(average_fs, f)
        with open(f'dump/{dataset}average_mfs.pkl', 'wb') as f:
            pickle.dump(average_mfs, f)
        with open(f'dump/{dataset}average_mfsc.pkl', 'wb') as f:
            pickle.dump(average_mfsc, f)
        with open(f'dump/{dataset}average_ts.pkl', 'wb') as f:
            pickle.dump(average_ts, f)
        with open(f'dump/{dataset}average_ks.pkl', 'wb') as f:
            pickle.dump(average_ks, f)
        with open(f'dump/{dataset}average_ms.pkl', 'wb') as f:
            pickle.dump(average_ms, f)

        with open(f'dump/{dataset}average_fs_err.pkl', 'wb') as f:
            pickle.dump(average_fs_err, f)
        with open(f'dump/{dataset}average_mfs_err.pkl', 'wb') as f:
            pickle.dump(average_mfs_err, f)
        with open(f'dump/{dataset}average_mfsc_err.pkl', 'wb') as f:
            pickle.dump(average_mfsc_err, f)
        with open(f'dump/{dataset}average_ts_err.pkl', 'wb') as f:
            pickle.dump(average_ts_err, f)
        with open(f'dump/{dataset}average_ks_err.pkl', 'wb') as f:
            pickle.dump(average_ks_err, f)
        with open(f'dump/{dataset}average_ms_err.pkl', 'wb') as f:
            pickle.dump(average_ms_err, f)

        with open(f'dump/{dataset}mean_error_rate_fs.pkl', 'wb') as f:
            pickle.dump(mean_error_rate_fs, f)
        with open(f'dump/{dataset}mean_error_rate_mfs.pkl', 'wb') as f:
            pickle.dump(mean_error_rate_mfs, f)
        with open(f'dump/{dataset}mean_error_rate_mfsc.pkl', 'wb') as f:
            pickle.dump(mean_error_rate_mfsc, f)
        with open(f'dump/{dataset}mean_error_rate_ts.pkl', 'wb') as f:
            pickle.dump(mean_error_rate_ts, f)
        with open(f'dump/{dataset}mean_error_rate_ks.pkl', 'wb') as f:
            pickle.dump(mean_error_rate_ks, f)
        with open(f'dump/{dataset}mean_error_rate_ms.pkl', 'wb') as f:
            pickle.dump(mean_error_rate_ms, f)

        with open(f'dump/{dataset}list_time_fs.pkl', 'wb') as f:
            pickle.dump(list_time_fs, f)
        with open(f'dump/{dataset}list_time_mfs.pkl', 'wb') as f:
            pickle.dump(list_time_mfs, f)
        with open(f'dump/{dataset}list_time_mfsc.pkl', 'wb') as f:
            pickle.dump(list_time_mfsc, f)
        with open(f'dump/{dataset}list_time_tk.pkl', 'wb') as f:
            pickle.dump(list_time_tk, f)
        with open(f'dump/{dataset}list_time_ms.pkl', 'wb') as f:
            pickle.dump(list_time_ms, f)

    return average_fs, average_mfs, average_mfsc, average_ts, average_ks, average_ms, \
              average_fs_err, average_mfs_err, average_mfsc_err, average_ts_err, average_ks_err, average_ms_err, \
                mean_error_rate_fs, mean_error_rate_mfs, mean_error_rate_mfsc, mean_error_rate_ts, mean_error_rate_ks, mean_error_rate_ms, \
                    list_time_fs, list_time_mfs, list_time_mfsc, list_time_tk, list_time_ms, \
                        L1_distances_tf, L1_distances_tmf, L1_distances_tmfc, L1_distances_tk, L1_distances_tm, \
                            L1_distances_tf_err, L1_distances_tmf_err, L1_distances_tmfc_err, L1_distances_tk_err, L1_distances_tm_err, \
                                L2_distances_tm, L2_distances_tm_err, L2_distances_tmf, L2_distances_tmf_err, L2_distances_tf, L2_distances_tf_err, \
                                    L2_distances_tmfc, L2_distances_tmfc_err, L2_distances_tk, L2_distances_tk_err

