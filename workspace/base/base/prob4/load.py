import os
import numpy as np
import keras
import tensorflow
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import *
from sklearn.preprocessing import OneHotEncoder

"""
********************************************************************************
    * load training, validation and test sets from original datset.
********************************************************************************
"""

def load():
    print("Loading...")
    data = pd.read_csv("prob4/datasets/wind.csv")



    train, rest_data = train_test_split(data, test_size=0.2)

    train_y = train.iloc[:,14].to_numpy()
    train_X = train.drop("MAL", axis='columns').to_numpy()




    valid, test = train_test_split(rest_data, test_size=0.5)
    valid_y = valid.iloc[:,14].to_numpy()
    valid_X = valid.drop("MAL", axis='columns').to_numpy()


    test_y = test.iloc[:,14].to_numpy()
    test_X = test.drop("MAL", axis='columns').to_numpy()

    np.save("prob4/datasets/original/train_X", train_X)
    np.save("prob4/datasets/original/train_y", train_y)
    np.save("prob4/datasets/original/valid_X", valid_X)
    np.save("prob4/datasets/original/valid_y", valid_y)
    np.save("prob4/datasets/original/test_X", test_X)
    np.save("prob4/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [14])
    np.save("prob4/datasets/original/names", features_names)




    np.save("prob4/datasets/custom/train_X", train_X)
    np.save("prob4/datasets/custom/train_y", train_y)
    np.save("prob4/datasets/custom/valid_X", valid_X)
    np.save("prob4/datasets/custom/valid_y", valid_y)
    np.save("prob4/datasets/custom/test_X", test_X)
    np.save("prob4/datasets/custom/test_y", test_y)

    np.save("prob4/datasets/custom/names", features_names)
