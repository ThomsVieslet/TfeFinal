import numpy as np
import importlib
import pandas as pd
import os

for x in os.walk("results"):
    if str(x[0]) != "results" and str(x[0]) != "results/history":

        data = pd.read_csv(x[0]+"/results.csv")
        loss = "loss_best"
        for i in data.columns:
            if "loss" in i or "objective" in i:
                loss = i
        index = np.argmax(data[loss])
        best_conf ={}
        for i in data.columns:
            print(type((data[i])[index]))
            if str(type((data[i])[index])) == "<class 'float'>" or str(type((data[i])[index])) == "<class 'numpy.float64'>":
                if (data[i])[index].is_integer():
                    print("heee")
                    best_conf[i] = int((data[i])[index])
                    print(best_conf[i])
                else:
                    best_conf[i] = (data[i])[index]
            else:
                best_conf[i] = (data[i])[index]


        mod_run = importlib.import_module("model_run", package=None)

        print(x[0])
        print(best_conf)
        print(mod_run.run(best_conf, test=True))
