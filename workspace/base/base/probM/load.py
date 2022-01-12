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
    data = pd.read_csv("probM/datasets/mc1.csv")
    arff_data = arff.load(open("probM/datasets/mc1.arff"))


    for i in arff_data["attributes"]:
        if 'Class' in i[0] or 'class' in i[0] or 'c' in i[0]:
            np.save("probM/datasets/original/Class", np.asarray(i[1]))

    #print(np.argwhere(np.isnan(data.to_numpy())))


    train, rest_data = train_test_split(data, test_size=0.2)

    train_y = train.iloc[:,38].to_numpy()
    train_X = train.drop("c", axis='columns').to_numpy()




    valid, test = train_test_split(rest_data, test_size=0.5)
    valid_y = valid.iloc[:,38].to_numpy()
    valid_X = valid.drop("c", axis='columns').to_numpy()


    test_y = test.iloc[:,38].to_numpy()
    test_X = test.drop("c", axis='columns').to_numpy()

    np.save("probM/datasets/original/train_X", train_X)
    np.save("probM/datasets/original/train_y", train_y)
    np.save("probM/datasets/original/valid_X", valid_X)
    np.save("probM/datasets/original/valid_y", valid_y)
    np.save("probM/datasets/original/test_X", test_X)
    np.save("probM/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [38])
    np.save("probM/datasets/original/names", features_names)




    np.save("probM/datasets/custom/train_X", train_X)
    np.save("probM/datasets/custom/train_y", train_y)
    np.save("probM/datasets/custom/valid_X", valid_X)
    np.save("probM/datasets/custom/valid_y", valid_y)
    np.save("probM/datasets/custom/test_X", test_X)
    np.save("probM/datasets/custom/test_y", test_y)

    np.save("probM/datasets/custom/names", features_names)
