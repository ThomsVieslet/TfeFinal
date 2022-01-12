import numpy as np
import importlib
import pandas as pd
import os
import  subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("Results_500.csv")


boolList = list()
for i in range(data.shape[0]):
    if not (data["Problem"])[i] in ["prob1", "prob2", "prob3", "prob4", "prob5"]:
        boolList.append(True)
    else:
        boolList.append(False)

data = data[boolList]

data = data.reset_index()

new_data = pd.DataFrame(columns=["Problem", "Nb_Features",  "Test_accuracy", "tpe", "ga1", "ambs", "ga2", "pso"])

rankings = {"tpe": list(), "ga1": list(), "ambs": list(), "ga2": list(), "pso": list()}

i = 0
current = "probH"
while i < len(data["Problem"]):
    to_add = dict()
    scoresL = list()
    while  i < len(data["Problem"]) and data["Problem"][i] == current :
        current = data["Problem"][i]
        to_add["Problem"] = data["Problem"][i]
        cntFeat = 0
        for j in data.columns:
            if "feature_" in j:
                if (data[j])[i] == 0.0:
                    cntFeat = cntFeat + 1
                elif (data[j])[i] == 1.0:
                    cntFeat = cntFeat + 1

        to_add["Nb_Features"] = cntFeat
        if data["Technique"][i] == "pso1":
            to_add["pso"] = data["Test Accuracy"][i]
        else:
            to_add[data["Technique"][i]] = data["Test Accuracy"][i]
        scoresL.append(data["Test Accuracy"][i])
        i = i + 1


    scores = pd.Series(scoresL)
    index_ = new_data.columns[3:]
    scores.index = index_
    ranks = scores.rank(ascending=False)


    for j in index_:
        to_add[j] = str(to_add[j]) +" ("+str(ranks[j])+")"
        rankings[j].append(ranks[j])

    new_data = new_data.append(to_add, ignore_index=True)

    if i >= len(data["Problem"]):
        break
    current = data["Problem"][i]

new_data.to_csv("Summary_500.csv")

for i in rankings:
    print(i)
    print(np.mean(rankings[i]))
    print(np.std(rankings[i]))
