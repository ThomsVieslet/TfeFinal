import numpy as np
import keras.backend as K
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import utils
from keras.utils import np_utils
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import r2_score
from keras.utils import to_categorical

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if os.getcwd() == "/home/thoms/Desktop/stage/tfe_automl_vieslet/workspace/base/base/probY":
    from probY.load_data import load_data, load_test
else:
    from load_data import load_data, load_test
from joblib import dump, load
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

"""
********************************************************************************
    * Arguments: point (the configuration as a dict) and test
        (specify if testing)
    * Return: the evaluation of the trained model on the validation set if
        test=False and on the test set if test=True.
********************************************************************************
"""


HISTORY = list()


def run(point, test= False):
    global HISTORY


    cnt = 0
    cntNull = 0
    for i in point:
        if "feature_" in i:
            cnt = cnt + 1
            if str(point[i]) == '0':
                cntNull = cntNull + 1

    if cntNull == cnt:
        HISTORY.append(0)
        return 0

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    nClasses = None

    j = 0
    feature = False
    for i in point.keys():
        if "feature" in i:
            feature = True
    if not test:
        if feature:
            (x_train, y_train), (x_valid, y_valid), nClasses = load_data(point)
        else:
            (x_train, y_train), (x_valid, y_valid), nClasses = load_data()
    else:
        if feature:
            (x_train, y_train), (x_valid, y_valid), nClasses = load_test(point)
        else:
            (x_train, y_train), (x_valid, y_valid), nClasses = load_test()



    if "model" in point.keys():
        if point["model"] == "randomForestClassifier":
            max_depth = None
            if point["max_depth"] != 1:
                max_depth = point["max_depth"]


            regr = RandomForestClassifier(max_depth=max_depth, criterion=point["criterion"],
                    min_samples_split=point["min_samples_split"],  min_samples_leaf=point["min_samples_leaf"], max_features=point["max_features"],
                    random_state=0, bootstrap=True, oob_score=True)

            history = regr.fit(x_train, y_train)

            filename = ""
            if test:
                filename = './content/model_HPT'
            else:
                filename = './probY/content/model_HPT'
            dump(regr, filename + '.joblib')



            hist = regr.score(x_valid, y_valid)
            HISTORY.append(hist)

            return hist

        elif point["model"] == "kneighborsClassifier":
            regr = KNeighborsClassifier(n_neighbors=point["n_neighbors"], weights=point["weights"], algorithm=point["algorithm"], leaf_size=point["leaf_size"])
            history = regr.fit(x_train, y_train)

            filename = ""
            if test:
                filename = './content/model_HPT'
            else:
                filename = './probY/content/model_HPT'
            dump(regr, filename + '.joblib')



            hist = regr.score(x_valid, y_valid)
            HISTORY.append(hist)

            return hist
        else:
            new_y_train = np.zeros((len(y_train), len(nClasses)))
            new_y_valid = np.zeros((len(y_valid), len(nClasses)))
            for i in range(len(y_train)):
                c = 0
                for j in nClasses:
                    if str(y_train[i]) == j:
                        c = c + 1
                        break
                    c = c + 1

                new_y_train[i, c-1] = 1

            for i in range(len(y_valid)):
                c = 0
                for j in nClasses:
                    if str(y_train[i]) == j:
                        c = c + 1
                        break
                    c = c + 1
                new_y_valid[i, c-1] = 1

            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_valid = scaler.transform(x_valid)

            regr = MLPClassifier(random_state=1, hidden_layer_sizes=(128, point['nLayers']), activation=point['activation'],
                solver= point['optimizer'], batch_size=point['batchSize'], learning_rate=point['lr'], max_iter=150)


            history = regr.fit(x_train, new_y_train)

            filename = ""
            if test:
                filename = './content/model_HPT'
            else:
                filename = './probY/content/model_HPT'
            dump(regr, filename + '.joblib')

            pred = regr.predict(x_valid)
            hist = 0
            if np.isnan(np.sum(pred)):
                hist = -float("inf")
            else:
                hist = regr.score(x_valid, new_y_valid, sample_weight=None)

            HISTORY.append(hist)
            return hist




    else:
        max_depth = None
        if point["max_depth"] != 1:
            max_depth = point["max_depth"]


        regr = RandomForestClassifier(max_depth=max_depth, criterion=point["criterion"],
                min_samples_split=point["min_samples_split"],  min_samples_leaf=point["min_samples_leaf"], max_features=point["max_features"],
                random_state=0, bootstrap=True, oob_score=True)

        history = regr.fit(x_train, y_train)
        filename = ""
        if test:
            filename = './content/model_HPT'
        else:
            filename = './probY/content/model_HPT'
        dump(regr, filename + '.joblib')



        hist = regr.score(x_valid, y_valid)
        HISTORY.append(hist)

        return hist



if __name__ == "__main__":
    point = {"units": 10, "activation": "relu", "lr": 0.01}
    objective = run(point)
    print("objective: ", objective)
    import matplotlib.pyplot as plt

    plt.plot(HISTORY["categorical_crossentropy"])
    plt.xlabel("Epochs")
    plt.ylabel("Objective: $R^2$")
    plt.grid()
    plt.show()
