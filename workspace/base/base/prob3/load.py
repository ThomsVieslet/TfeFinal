import os
import numpy as np
import keras
import tensorflow
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import *
from sklearn.preprocessing import OneHotEncoder

def load_test(point = None):
    train_X = np.load("datasets/custom/train_X.npy",allow_pickle=True)
    train_y = np.load("datasets/custom/train_y.npy",allow_pickle=True)
    test_X = np.load("datasets/custom/test_X.npy",allow_pickle=True)
    test_y = np.load("datasets/custom/test_y.npy",allow_pickle=True)
    features_names = np.load("datasets/custom/names.npy",allow_pickle=True)

    if point is not None:
        ind_rem = []
        j = 0
        for i in point:
            if "feature_" in i:
                if point[i] == "0":
                    ind_rem.append(j)


                j =  j + 1

        trans_train_X = np.delete(train_X, ind_rem, axis=1)
        trans_test_X = np.delete(test_X, ind_rem, axis=1)
        #new_features_names = np.delete(features_names, ind_rem)

    return (trans_train_X, train_y), (trans_test_X, test_y)

def load():
    print("Loading...")
    data = pd.read_csv("prob3/datasets/phpYYZ4Qc.csv")




    train, rest_data = train_test_split(data, test_size=0.2)

    train_y = train.iloc[:,32].to_numpy()
    train_X = train.drop("rej", axis='columns').to_numpy()


    valid, test = train_test_split(rest_data, test_size=0.5)
    valid_y = valid.iloc[:,32].to_numpy()
    valid_X = valid.drop("rej", axis='columns').to_numpy()


    test_y = test.iloc[:,32].to_numpy()
    test_X = test.drop("rej", axis='columns').to_numpy()

    np.save("prob3/datasets/original/train_X", train_X)
    np.save("prob3/datasets/original/train_y", train_y)
    np.save("prob3/datasets/original/valid_X", valid_X)
    np.save("prob3/datasets/original/valid_y", valid_y)
    np.save("prob3/datasets/original/test_X", test_X)
    np.save("prob3/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [32])
    np.save("prob3/datasets/original/names", features_names)
    

    np.save("prob3/datasets/custom/train_X", train_X)
    np.save("prob3/datasets/custom/train_y", train_y)
    np.save("prob3/datasets/custom/valid_X", valid_X)
    np.save("prob3/datasets/custom/valid_y", valid_y)
    np.save("prob3/datasets/custom/test_X", test_X)
    np.save("prob3/datasets/custom/test_y", test_y)

    np.save("prob3/datasets/custom/names", features_names)
