import numpy as np
import importlib
import pandas as pd
import os
import  subprocess
import matplotlib.pyplot as plt

sims = next(os.walk(os. getcwd()+"/results_history_500/probR"))[2]

dataScore = pd.DataFrame(columns=[ "ambs", "ga", "pso", "rs", "tpe"])
dataModel = pd.DataFrame(columns=[ "ambs", "ga", "pso", "rs", "tpe"])

for i in sims:
    #files = next(os.walk(os. getcwd()+"/results_history_500/probR"+ "/"+i))[2]
    runs = list()
    loss = ""
    if i.split("_")[1] == "ambs":
        loss = "objective"
    else:
        loss = "loss"

    """
    l = 0
    for j in files:
        if l >= 5:
            break
        data = pd.read_csv(os. getcwd()+"/results_history_500/probR"+ "/"+i+"/"+j)
        runs.append(data)

        l = l + 1
    """
    tmp = pd.read_csv(os. getcwd()+"/results_history_500/probR"+ "/"+i)
    #tmp = pd.concat(runs, ignore_index=True)
    dataScore[i.split("_")[1]] = tmp[loss]
    dataModel[i.split("_")[1]] = tmp["model"]

dataScore.to_csv("scoresLine1.csv")
dataModel.to_csv("modelsLine1.csv")

modelTable = pd.DataFrame(data=0,columns=[ "ambs", "ga", "pso", "rs", "tpe"], index=["randomForestClassifier", "kneighborsClassifier", "nnClassifier"])

j = 0
while j < len(dataModel["ambs"]):

    for l in modelTable.columns:
        print(l)
        modelTable[l][dataModel[l][j]] = modelTable[l][dataModel[l][j]] + 1

    print(j)
    j = j + 1

modelTable = modelTable/5
modelTable.to_csv("modelPercent1.csv")



green_diamond = dict(markerfacecolor='g', marker='D')
fig1, ax1 = plt.subplots()
ax1.set_title('Repartition of Methods\' scores across evaluations')
bplot1 = ax1.boxplot(dataScore, flierprops=green_diamond, patch_artist=True)
plt.xticks([1, 2, 3, 4, 5], dataScore.columns)

colors = ["c","red","green","grey", "blue"]
k = 0
for patch in bplot1['boxes']:
        patch.set(color=colors[k])
        k = k + 1

# adding horizontal grid lines
ax1.yaxis.grid(True)
ax1.set_xlabel('Optimization Method')
ax1.set_ylabel('Validation Scores')


plt.savefig('Boxplots500_1')
