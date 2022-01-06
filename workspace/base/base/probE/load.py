import os
import numpy as np
import keras
import tensorflow
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import *
from sklearn.preprocessing import OneHotEncoder
import arff

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
    data = pd.read_csv("probE/datasets/kdd_JapaneseVowels.csv")
    arff_data = arff.load(open("probE/datasets/kdd_JapaneseVowels.arff"))


    for i in arff_data["attributes"]:
        if 'Class' in i[0] or 'class' in i[0]:
            np.save("probE/datasets/original/Class", np.asarray(i[1]))

    #print(np.argwhere(np.isnan(data.to_numpy())))


    train, rest_data = train_test_split(data, test_size=0.2)

    train_y = train.iloc[:,14].to_numpy()
    train_X = train.drop("binaryClass", axis='columns').to_numpy()




    valid, test = train_test_split(rest_data, test_size=0.5)
    valid_y = valid.iloc[:,14].to_numpy()
    valid_X = valid.drop("binaryClass", axis='columns').to_numpy()


    test_y = test.iloc[:,14].to_numpy()
    test_X = test.drop("binaryClass", axis='columns').to_numpy()

    np.save("probE/datasets/original/train_X", train_X)
    np.save("probE/datasets/original/train_y", train_y)
    np.save("probE/datasets/original/valid_X", valid_X)
    np.save("probE/datasets/original/valid_y", valid_y)
    np.save("probE/datasets/original/test_X", test_X)
    np.save("probE/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [14])
    np.save("probE/datasets/original/names", features_names)




    np.save("probE/datasets/custom/train_X", train_X)
    np.save("probE/datasets/custom/train_y", train_y)
    np.save("probE/datasets/custom/valid_X", valid_X)
    np.save("probE/datasets/custom/valid_y", valid_y)
    np.save("probE/datasets/custom/test_X", test_X)
    np.save("probE/datasets/custom/test_y", test_y)

    np.save("probE/datasets/custom/names", features_names)
