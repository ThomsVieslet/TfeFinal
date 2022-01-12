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
load:
    * load the original dataset
    * split it in train, validation and test set
********************************************************************************
"""

def load(point = None):
    print("Loading...")
    data = pd.read_csv("prob1/datasets/avocado.csv", parse_dates=["Date"])
    train_X, train_y, valid_X, valid_y, test_y, test_X = clean_avocado(data)

    np.save("prob1/datasets/original/train_X", train_X)
    np.save("prob1/datasets/original/train_y", train_y)
    np.save("prob1/datasets/original/valid_X", valid_X)
    np.save("prob1/datasets/original/valid_y", valid_y)
    np.save("prob1/datasets/original/test_X", test_X)
    np.save("prob1/datasets/original/test_y", test_y)

    features_names = np.delete(data.columns, [0,2])
    np.save("prob1/datasets/original/names", features_names)




    np.save("prob1/datasets/custom/train_X", train_X)
    np.save("prob1/datasets/custom/train_y", train_y)
    np.save("prob1/datasets/custom/valid_X", valid_X)
    np.save("prob1/datasets/custom/valid_y", valid_y)
    np.save("prob1/datasets/custom/test_X", test_X)
    np.save("prob1/datasets/custom/test_y", test_y)

    np.save("prob1/datasets/custom/names", features_names)




"""
********************************************************************************
clean_avocado:
    * clean avocado dataset
    * prepare avocado dataset
********************************************************************************
"""

def clean_avocado(data):
    firstDate = data["Date"].min()
    newDates = []
    for i in data["Date"]:
        newDates.append(int((i-firstDate).total_seconds()))

    data["Date"] = newDates
    organic = []
    for i in  data["type"]:
        if i == "Conventional":
            organic.append(False)
        else:
            organic.append(True)
    data.drop('type', inplace=True, axis=1)
    data["Organic"] = organic

    enc = OneHotEncoder(categories='auto')
    enc.fit((data["region"].to_numpy()).reshape(1, -1))

    data["region"] = enc.transform((data["region"].to_numpy()).reshape(1, -1))



    train, rest_data = train_test_split(data, test_size=0.2)
    train_y = train.iloc[:,2].to_numpy()
    train_X = train.iloc[:, :-2].to_numpy()


    valid, test = train_test_split(rest_data, test_size=0.5)

    valid_y = valid.iloc[:,2].to_numpy()
    valid_X = valid.iloc[:, :-2].to_numpy()

    test_y = test.iloc[:,2].to_numpy()
    test_X = test.iloc[:, :-2].to_numpy()


    return train_X, train_y, valid_X, valid_y, test_y, test_X
