import numpy as np
import importlib
import pandas as pd
import os
import  subprocess


directories = next(os.walk(os. getcwd()+'/results_history'))[1]


res = pd.DataFrame(columns=["Problem", "Technique", "Budget", "Validation Accuracy",
    "Test Accuracy", "activation", "algorithm", "batchSize", "criterion", "leaf_size",
    "lr",	"max_depth", "max_features", "min_samples_leaf", "min_samples_split", "model",
     "nLayers", "n_neighbors", "optimizer",	"weights",	"loss"])

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


for i in directories:
    if "prob" in i:
        files = next(os.walk(os. getcwd()+'/results_history/'+i))[2]
        for j in files:
            if "CASH" in j:
                data = pd.read_csv(os.getcwd()+'/results_history/'+i+"/"+j)
                loss = "loss_best"
                for l in data.columns:
                    if "loss" in l or "objective" in l:
                        loss = l
                index = np.argmax(data[loss])
                valid_accuracy = np.max(data[loss])
                best_conf ={}
                for k in data.columns:
                    if str(type((data[k])[index])) == "<class 'float'>" or str(type((data[k])[index])) == "<class 'numpy.float64'>":
                        if (data[k])[index].is_integer():
                            best_conf[k] = int((data[k])[index])
                        else:
                            best_conf[k] = (data[k])[index]
                    else:
                        best_conf[k] = (data[k])[index]

                os.chdir(i)
                mod_run = importlib.import_module(i+".model_run", package=None)
                for m in best_conf:
                    if str(type(best_conf[m])) == "<class \'str\'>":
                        if isfloat(best_conf[m]):
                            if (float(best_conf[m])).is_integer():
                                best_conf[m] = int(float(best_conf[m]))
                            else:
                                best_conf[m] = float(best_conf[m])
                test_accuracy = mod_run.run(best_conf, test=True)
                os.chdir("../")
                """
                if j.split("_")[0] == "1":
                    dict = {"Problem":i, "Technique": j.split("_")[2]+"1", "Budget":j.split("_")[3], "Validation Accuracy": valid_accuracy,
                    "Test Accuracy": test_accuracy}
                else:
                    dict = {"Problem":i, "Technique": j.split("_")[1], "Budget":j.split("_")[2], "Validation Accuracy": valid_accuracy,
                    "Test Accuracy": test_accuracy}
                """
                dict = best_conf
                
                dict["Problem"] = i
                dict["Technique"] = j.split("_")[1]
                dict["Budget"] = j.split("_")[2]
                dict["Validation Accuracy"] = valid_accuracy
                dict["Test Accuracy"] = test_accuracy


                res = res.append(dict, ignore_index=True)



print(res.head())
res.to_csv("Results_500.csv")

"""
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
"""
