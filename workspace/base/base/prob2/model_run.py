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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from load_data import load_data, load_test
from joblib import dump, load


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

    j = 0
    feature = False
    for i in point.keys():
        if "feature" in i:
            feature = True
    if not test:
        if feature:
            (x_train, y_train), (x_valid, y_valid) = load_data(point)
        else:
            (x_train, y_train), (x_valid, y_valid) = load_data()
    else:
        if feature:
            (x_train, y_train), (x_valid, y_valid) = load_test(point)
        else:
            (x_train, y_train), (x_valid, y_valid) = load_test()



    if "model" in point.keys():
        if point["model"] == "randomForestRegressor":
            max_depth = None
            if point["max_depth"] != 1:
                max_depth = point["max_depth"]


            regr = RandomForestRegressor(max_depth=max_depth, criterion=point["criterion"],
                    min_samples_split=point["min_samples_split"],  min_samples_leaf=point["min_samples_leaf"], max_features=point["max_features"],
                    random_state=0, bootstrap=True, oob_score=True)

            history = regr.fit(x_train, y_train)

            filename = ""
            if test:
                filename = './content/model_HPT'
            else:
                filename = './prob2/content/model_HPT'
            dump(regr, filename + '.joblib')



            hist = regr.score(x_valid, y_valid)
            HISTORY.append(hist)

            return hist

        elif point["model"] == "kneighborsRegressor":
            regr = KNeighborsRegressor(n_neighbors=point["n_neighbors"], weights=point["weights"], algorithm=point["algorithm"], leaf_size=point["leaf_size"])
            history = regr.fit(x_train, y_train)

            filename = ""
            if test:
                filename = './content/model_HPT'
            else:
                filename = './prob2/content/model_HPT'
            dump(regr, filename + '.joblib')



            hist = regr.score(x_valid, y_valid)
            HISTORY.append(hist)

            return hist
        else:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_valid = scaler.transform(x_valid)

            regr = MLPRegressor(random_state=1, hidden_layer_sizes=(128, point['nLayers']), activation=point['activation'],
                solver= point['optimizer'], batch_size=point['batchSize'], learning_rate=point['lr'], max_iter=150)


            history = regr.fit(x_train, y_train)

            filename = ""
            if test:
                filename = './content/model_HPT'
            else:
                filename = './prob2/content/model_HPT'
            dump(regr, filename + '.joblib')

            pred = regr.predict(x_valid)
            hist = 0
            if np.isnan(np.sum(pred)):
                hist = -float("inf")
            else:
                hist = r2_score(y_valid, pred)

            HISTORY.append(hist)
            return hist





    else:
        max_depth = None
        if point["max_depth"] != 1:
            max_depth = point["max_depth"]


        regr = RandomForestRegressor(max_depth=max_depth, criterion=point["criterion"],
                min_samples_split=point["min_samples_split"],  min_samples_leaf=point["min_samples_leaf"], max_features=point["max_features"],
                random_state=0, bootstrap=True, oob_score=True)

        history = regr.fit(x_train, y_train)
        filename = ""
        if test:
            filename = './content/model_HPT'
        else:
            filename = './prob2/content/model_HPT'
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
