import os
import numpy as np
import keras
import tensorflow
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import *
from sklearn.preprocessing import OneHotEncoder
import arff

"""
********************************************************************************
    * load training, validation and test sets from original datset.
********************************************************************************
"""

def load():
    print("Loading...")
    data = pd.read_csv("probI/datasets/puma8NH.csv")
    arff_data = arff.load(open("probI/datasets/puma8NH.arff"))


    for i in arff_data["attributes"]:
        if 'Class' in i[0] or 'class' in i[0]:
            np.save("probI/datasets/original/Class", np.asarray(i[1]))

    #print(np.argwhere(np.isnan(data.to_numpy())))


    train, rest_data = train_test_split(data, test_size=0.2)

    train_y = train.iloc[:,8].to_numpy()
    train_X = train.drop("binaryClass", axis='columns').to_numpy()




    valid, test = train_test_split(rest_data, test_size=0.5)
    valid_y = valid.iloc[:,8].to_numpy()
    valid_X = valid.drop("binaryClass", axis='columns').to_numpy()


    test_y = test.iloc[:,8].to_numpy()
    test_X = test.drop("binaryClass", axis='columns').to_numpy()

    np.save("probI/datasets/original/train_X", train_X)
    np.save("probI/datasets/original/train_y", train_y)
    np.save("probI/datasets/original/valid_X", valid_X)
    np.save("probI/datasets/original/valid_y", valid_y)
    np.save("probI/datasets/original/test_X", test_X)
    np.save("probI/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [8])
    np.save("probI/datasets/original/names", features_names)




    np.save("probI/datasets/custom/train_X", train_X)
    np.save("probI/datasets/custom/train_y", train_y)
    np.save("probI/datasets/custom/valid_X", valid_X)
    np.save("probI/datasets/custom/valid_y", valid_y)
    np.save("probI/datasets/custom/test_X", test_X)
    np.save("probI/datasets/custom/test_y", test_y)

    np.save("probI/datasets/custom/names", features_names)
