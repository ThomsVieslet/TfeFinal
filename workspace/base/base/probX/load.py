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
    data = pd.read_csv("probX/datasets/phpV5QYya.csv")
    arff_data = arff.load(open("probX/datasets/phpV5QYya.arff"))


    for i in arff_data["attributes"]:
        if 'Class' in i[0] or 'Result' in i[0]:
            np.save("probX/datasets/original/Class", np.asarray(i[1]))

    #print(np.argwhere(np.isnan(data.to_numpy())))


    train, rest_data = train_test_split(data, test_size=0.2)

    train_y = train.iloc[:,30].to_numpy()
    train_X = train.drop("Result", axis='columns').to_numpy()




    valid, test = train_test_split(rest_data, test_size=0.5)
    valid_y = valid.iloc[:,30].to_numpy()
    valid_X = valid.drop("Result", axis='columns').to_numpy()


    test_y = test.iloc[:,30].to_numpy()
    test_X = test.drop("Result", axis='columns').to_numpy()

    np.save("probX/datasets/original/train_X", train_X)
    np.save("probX/datasets/original/train_y", train_y)
    np.save("probX/datasets/original/valid_X", valid_X)
    np.save("probX/datasets/original/valid_y", valid_y)
    np.save("probX/datasets/original/test_X", test_X)
    np.save("probX/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [30])
    np.save("probX/datasets/original/names", features_names)




    np.save("probX/datasets/custom/train_X", train_X)
    np.save("probX/datasets/custom/train_y", train_y)
    np.save("probX/datasets/custom/valid_X", valid_X)
    np.save("probX/datasets/custom/valid_y", valid_y)
    np.save("probX/datasets/custom/test_X", test_X)
    np.save("probX/datasets/custom/test_y", test_y)

    np.save("probX/datasets/custom/names", features_names)
