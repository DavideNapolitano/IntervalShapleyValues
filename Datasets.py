from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import shap
import pandas as pd
import wget


def Monks():
    dataset="Monks"

    data = np.loadtxt('datasets\monks-1.test', dtype=object, delimiter=' ') #SWAP
    X = data[:,2:-1]
    Y =data[:,1]

    data_test = np.loadtxt('datasets\monks-1.train', dtype=object, delimiter=' ') #SWAP
    X_test = data_test[:,2:-1]
    Y_test = data_test[:,1]

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Data scaling
    num_features = X_train.shape[1]
    feature_names = ["a1","a2","a3","a4","a5","a6"]
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_s = ss.transform(X_train)
    X_val_s = ss.transform(X_val)
    X_test_s = ss.transform(X_test)

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset


def Census():
    dataset="Census"
    X=shap.datasets.adult()[0].to_numpy()
    X=X.astype("object")
    Y=shap.datasets.adult()[1]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    num_features = X_train.shape[1]
    feature_names = shap.datasets.adult()[0].columns.tolist()
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_s = ss.transform(X_train)
    X_val_s = ss.transform(X_val)
    X_test_s = ss.transform(X_test)

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset	


def WBCD():
    dataset="WBCD"
    columns=["ID","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
    data = pd.read_csv('datasets/breast-cancer-wisconsin.data', dtype=object, delimiter=',',header=None,na_values="?")
    data.columns=columns
    data=data.dropna(axis=0)

    Y=data["Class"]
    Y=Y.astype(np.int64)
    tmp=[]
    for el in Y:
        if el==2:
            tmp.append(1)
        else:
            tmp.append(0)
    Y=np.array(tmp)
    X=data.drop(columns=["ID","Class"])
    columns=X.columns
    X=X.to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    num_features = X_train.shape[1]
    feature_names = columns
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_s = ss.transform(X_train)
    X_val_s = ss.transform(X_val)
    X_test_s = ss.transform(X_test)

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset

def Heart():

    dataset = pd.read_csv('datasets\heart.csv', sep=",")
    dataset=dataset.drop("id",axis=1)

    mapper={"present":1,"absent":0}
    y=dataset["label"].values
    Y=[mapper[el] for el in y]
    Y=np.array(Y)
    X=dataset.drop("label",axis=1)
    columns=X.columns
    X=X.values
    print(len(X),len(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    num_features = X_train.shape[1]
    feature_names = columns
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_s = ss.transform(X_train)
    X_val_s = ss.transform(X_val)
    X_test_s = ss.transform(X_test)
    dataset="Heart"

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset

def Credit():
    dataset="Credit"
    dataset = pd.read_csv('datasets\credit_card.csv', sep=",")
    dataset=dataset.drop("ID", axis=1)

    #mapper={"present":1,"absent":0}
    Y=dataset["DEFAULT_PAYMENT"].values
    #Y=[mapper[el] for el in y]
    X=dataset.drop("DEFAULT_PAYMENT",axis=1)
    columns=X.columns
    X=X.values
    print(len(X),len(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=7)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    num_features = X_train.shape[1]
    feature_names = columns
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_s = ss.transform(X_train)
    X_val_s = ss.transform(X_val)
    X_test_s = ss.transform(X_test)

    return X_train_s, X_val_s, X_test_s, Y_train, Y_val, Y_test, feature_names, num_features, dataset