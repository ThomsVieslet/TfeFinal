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
    if (data["Problem"])[i] in ["prob1", "prob2", "prob3", "prob4", "prob5"]:
        boolList.append(True)
    else:
        boolList.append(False)

data = data[boolList]

data = data.reset_index()

new_data = pd.DataFrame(columns=["Problem", "Nb_Features", "Taken_Feat_Ratio", "Test_accuracy", "Model", "Algo"])

previous = "prob4"
i = 0

while i < len(data["Problem"]):
    best = dict()
    current = -float("inf")
    while (data["Problem"])[i] == previous:
        previous = (data["Problem"])[i]
        if (data["Test Accuracy"])[i] > current and (data["Technique"])[i] != "ga2":
            current = (data["Test Accuracy"])[i]
            cntFeat = 0
            cntTaken = 0
            for j in data.columns:
                if "feature_" in j:
                    if (data[j])[i] == 0.0:
                        cntFeat = cntFeat + 1
                    elif (data[j])[i] == 1.0:
                        cntFeat = cntFeat + 1
                        cntTaken = cntTaken + 1
            best = {"Problem":(data["Problem"])[i], "Nb_Features": cntFeat, "Taken_Feat_Ratio": cntTaken/cntFeat,
                "Test_accuracy":(data["Test Accuracy"])[i] , "Model": (data["model"])[i], "Algo": (data["Technique"])[i]}
        i = i + 1
        if i >= len(data["Problem"]):
            break

    if i < data.shape[0]:
        previous = (data["Problem"])[i]
    print(best)
    new_data = new_data.append(best, ignore_index=True)



print(new_data)


frequences = pd.DataFrame(data=0,columns=["ambs", "tpe", "ga1", "pso1", "Models Frequencies"], index=["randomForestRegressor", "kneighborsRegressor", "nnRegressor", "Methods Frequencies"])
print(frequences)


k = 0
for i in new_data["Algo"]:
    j = new_data["Model"][k]
    frequences.loc[j , i] = frequences.loc[j , i] + 1
    k = k + 1

sum = np.zeros((len(frequences["ambs"])))
for i in range(len(frequences.loc["randomForestRegressor", :])):
    sum = frequences.iloc[:, i] + sum

frequences["Models Frequencies"] = sum


sum = np.zeros((len(frequences.loc["randomForestRegressor", :])))
for i in range(len(frequences.loc[:, "ambs"])):
    sum = frequences.iloc[i, :] + sum

frequences.loc["Methods Frequencies", :] = sum

frequences = frequences/len(new_data)

frequences.to_csv("frequenciesReg.csv")
"""

print(new_data["Nb_Features"].drop(19))

plt.scatter(new_data["Taken_Feat_Ratio"], new_data["Test_accuracy"])
plt.xlabel("Leprechauns")
plt.ylabel("Gold")
plt.legend(loc='upper left')
plt.show()
plt.savefig('scatter')

colors = ["c","red","green","blue"]
algo = ["ambs", "tpe", "ga1", "pso1"]


ambsD =

for i in new_data["Algo"]:
    c.append(colors[algo.index(i)])

fig, ax = plt.subplots()

k = 0
for color in c:
    ax.scatter(new_data["Taken_Feat_Ratio"][k], new_data["Test_accuracy"][k], c=color,  label=new_data["Algo"][k])
    k = k + 1

ax.legend()
plt.savefig('scatter')
"""
