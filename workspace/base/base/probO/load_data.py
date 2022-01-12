import os
import numpy as np
import keras
import tensorflow

"""
********************************************************************************
load_test:
    * Arguments:  point is a dictionnary expressing a configuration or None
    * Return: train and test set. With the selected features if point != None
********************************************************************************
"""

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


"""
********************************************************************************
load_data:
    * Arguments:  point is a dictionnary expressing a configuration or None
    * Return: train and validation set. With the selected features if
        point != None
********************************************************************************
"""
def load_data(point = None):
  train_X = np.load("probO/datasets/custom/train_X.npy",allow_pickle=True)
  train_y = np.load("probO/datasets/custom/train_y.npy",allow_pickle=True)
  valid_X = np.load("probO/datasets/custom/valid_X.npy",allow_pickle=True)
  valid_y = np.load("probO/datasets/custom/valid_y.npy",allow_pickle=True)
  features_names = np.load("probO/datasets/custom/names.npy",allow_pickle=True)
  nClasses = np.load("probO/datasets/original/Class.npy",allow_pickle=True)

  trans_train_X = train_X
  trans_valid_X = valid_X
  new_features_names = features_names


  if point is not None:
      ind_rem = []
      j = 0
      for i in point:
          if "feature_" in i:
              if point[i] == "0":
                  ind_rem.append(j)


              j =  j + 1

      trans_train_X = np.delete(train_X, ind_rem, axis=1)
      trans_valid_X = np.delete(valid_X, ind_rem, axis=1)
      #new_features_names = np.delete(features_names, ind_rem)

  return (trans_train_X, train_y), (trans_valid_X, valid_y), nClasses
