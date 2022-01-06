import numpy as np
import importlib
import pandas as pd
import os
import  subprocess


directories = next(os.walk(os. getcwd()+"/results_history"))[1]

new_data = pd.DataFrame(columns=["Problem", "Nb_Features", "ambs", "ga", "pso", "rs", "tpe"])

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


rankings = {"ambs": list(), "ga": list(), "pso": list(), "rs": list(), "tpe": list()}


for i in directories:
    sims = next(os.walk(os. getcwd()+"/results_history"+ "/"+ i))[1]
    scoresL = list()
    algo_names = list()
    to_add = dict()
    for j in sims:
        files = next(os.walk(os. getcwd()+"/results_history"+ "/"+ i+"/"+j))[2]
        l = 0
        test_accuracies = np.zeros(5)
        for k in files:
            if l >= 5:
                break

            data = pd.read_csv(os. getcwd()+"/results_history"+ "/"+ i+"/"+j+"/"+k)

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


            test_accuracies[l] = test_accuracy
            l = l + 1
            os.chdir("../")


        to_add[j.split("_")[1]] = str(round(np.mean(test_accuracies), 4)) + chr(177) + str(round(np.std(test_accuracies), 4))
        algo_names.append(j.split("_")[1])
        scoresL.append(np.mean(test_accuracies))


    to_add["Problem"] = i


    scores = pd.Series(scoresL)
    index_ = algo_names
    scores.index = index_
    ranks = scores.rank(ascending=False)



    for j in index_:
        to_add[j] = to_add[j] +" ("+str(ranks[j])+")"
        rankings[j].append(ranks[j])


    new_data = new_data.append(to_add, ignore_index=True)






print(new_data)
new_data.to_csv("Summary_100.csv")
