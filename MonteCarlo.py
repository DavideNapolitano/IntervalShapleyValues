import random
import numpy as np

def MonteCarlo(IND, DATA, MODEL, OM, NUM_FEAT):

    average_ms=[]
    average_err_ms=[]

    x=DATA.iloc[IND]
    TH=0
    
    error_sv_m=0
    for j in range(NUM_FEAT):
        # print("-"*100)
        # print(feature_names[j])
        M = 1000
        n_features = len(x)
        marginal_contributions = []
        marginal_contributions_lower = []
        marginal_contributions_upper = []

        feature_idxs = list(range(n_features))
        feature_idxs.remove(j)
        for itr in range(M):
            z = DATA.sample(1).values[0]
            if itr<TH:
                print("sample:",len(DATA.sample(1)))
                print("z:",z)
            x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features))) #estraggo 0.8*feature_idx
            if itr<TH:
                print("x_idx:",x_idx, sorted(x_idx))
            z_idx = [idx for idx in feature_idxs if idx not in x_idx] # features non estratte. Ricorda che una feauture, quella su cui si calcola lo SV, è sempre esclusa
            if itr<TH:
                print("z_idx:",z_idx)

            # construct two new instances
            x_plus_j = np.array([x[i] if i in x_idx + [j] else z[i] for i in range(n_features)])
            x_minus_j = np.array([z[i] if i in z_idx + [j] else x[i] for i in range(n_features)])
            
            ##############################################################################
            # calculate marginal contribution
            if itr<TH:
                print("x_plus:",x_plus_j)
                print("x_plus_reshape:",x_plus_j.reshape(1, -1))
                print("pred_plus:",MODEL.predict_proba(x_plus_j.reshape(1, -1)))
            x_plus_j=x_plus_j.reshape(1, -1)#np.expand_dims(x_plus_j, axis=0)
            x_minus_j=x_minus_j.reshape(1, -1)#np.expand_dims(x_minus_j, axis=0)
            
            
            plus_p, plus_ci=OM.get_pred_ci(x_plus_j)
            plus_p=plus_p[0]
            plus_ci=plus_ci[0]
            # print(plus_p)
            # print(plus_ci)
            plus_l=np.array(plus_p)-np.array(plus_ci)
            plus_u=np.array(plus_p)+np.array(plus_ci)
            # print(plus_l)
            # print(plus_u)
            v1_plus=np.array([plus_l[0],plus_u[1]])
            v2_plus=np.array([plus_l[1], plus_u[0]])

            minus_p, minus_ci=om.get_pred_ci(x_minus_j)
            minus_p=minus_p[0]
            minus_ci=minus_ci[0]
            # print(minus_p)
            # print(minus_ci)
            minus_l=np.array(minus_p)-np.array(minus_ci)
            minus_u=np.array(minus_p)+np.array(minus_ci)
            # print(minus_l)
            # print(minus_u)
            v1_minus=np.array([minus_l[0],minus_u[1]])
            v2_minus=np.array([minus_l[1], minus_u[0]])
            

            tmp_plus=MODEL.predict_proba(x_plus_j.reshape(1, -1))[0]
            tmp_minus=MODEL.predict_proba(x_minus_j.reshape(1, -1))[0]
            if itr<TH:
                print(plus_p[0])
                print(tmp_plus)
                print(minus_p[0])
                print(tmp_minus)
                
            marginal_contribution =  plus_p[y] - minus_p[y]
            marginal_contribution_l= max(0,plus_l[y])-max(0,minus_l[y])
            marginal_contribution_u=min(1,plus_u[y])-min(1,minus_u[y])
            
            marginal_contributions.append(marginal_contribution)
            marginal_contributions_lower.append(marginal_contribution_l)
            marginal_contributions_upper.append(marginal_contribution_u)

        phi_j_x = sum(marginal_contributions) / len(marginal_contributions)  # our shaply value
        phi_j_x_l = sum(marginal_contributions_lower) / len(marginal_contributions_lower)  # our shaply value
        phi_j_x_u = sum(marginal_contributions_upper) / len(marginal_contributions_upper)  # our shaply value
        # print(f"Shaply value for feature {feature_names[j]}: {phi_j_x:.5}")
        # print(f"Shaply value LOWER for feature {feature_names[j]}: {phi_j_x_l:.5}")
        # print(f"Shaply value UPPER for feature {feature_names[j]}: {phi_j_x_u:.5}")
        if phi_j_x_u<phi_j_x_l:
            #print("Inverted Intervals")
            error_sv_m+=1
        
        mean=(phi_j_x_l+phi_j_x_u)/2
        err=np.abs(mean-phi_j_x_l)
        average_ms.append(mean)
        average_err_ms.append(err)
        
    return average_ms, average_err_ms, error_sv_m