import numpy as np
import importlib
import pandas as pd
import os
import  subprocess


directories = next(os.walk(os. getcwd()+"/results_history_500"))[1]

new_data = pd.DataFrame(columns=["Problem", "Nb_Features", "ambs", "ga", "pso", "rs", "tpe"])

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


rankings = {"ambs": list(), "ga": list(), "pso": list(), "rs": list(), "tpe": list()}


for i in directories:
    sims = next(os.walk(os. getcwd()+"/results_history_500/"+ i))[2]
    scoresL = list()
    algo_names = list()
    to_add = dict()
    to_add["Problem"] = i
    for j in sims:
        data = pd.read_csv(os. getcwd()+"/results_history_500/"+ i+"/"+j)

        cntFeat = 0
        for n in data.columns:
            if "feature_" in n:
                cntFeat = cntFeat + 1

        to_add["Nb_Features"] = cntFeat



        loss = "loss_best"
        for m in data.columns:
            if "loss" in m or "objective" in m:
                loss = m
        index = np.argmax(data[loss])
        valid_accuracy = np.max(data[loss])
        best_conf ={}
        for r in data.columns:
            if str(type((data[r])[index])) == "<class 'float'>" or str(type((data[r])[index])) == "<class 'numpy.float64'>":
                if (data[r])[index].is_integer():
                    best_conf[r] = int((data[r])[index])
                else:
                    best_conf[r] = (data[r])[index]
            else:
                best_conf[r] = (data[r])[index]

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

        to_add[j.split("_")[1]] = round(test_accuracy,4)
        
        algo_names.append(j.split("_")[1])
        scoresL.append(test_accuracy)
        os.chdir("../")

    scores = pd.Series(scoresL)
    index_ = algo_names
    scores.index = index_
    ranks = scores.rank(ascending=False)



    for j in index_:
        to_add[j] = str(to_add[j]) +" ("+str(ranks[j])+")"
        rankings[j].append(ranks[j])


    new_data = new_data.append(to_add, ignore_index=True)


new_data.to_csv("Summary_500_RS")
