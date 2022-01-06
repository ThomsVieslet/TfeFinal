import numpy as np
import importlib
import pandas as pd
import os
import  subprocess
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import random
import matplotlib
from IPython.display import Image


sims = next(os.walk(os. getcwd()+"/results_history/prob1"))[1]

dataScore = {"ambs": np.zeros((100, 5)),  "ga": np.zeros((100, 5)), "pso": np.zeros((100, 5)), "rs": np.zeros((100, 5)), "tpe": np.zeros((100, 5))}
dataModels = {"ambs": np.zeros((100, 5), dtype=int),  "ga": np.zeros((100, 5), dtype=int), "pso": np.zeros((100, 5), dtype=int), "rs": np.zeros((100, 5), dtype=int), "tpe": np.zeros((100, 5), dtype=int)}

for i in sims:
    files = next(os.walk(os. getcwd()+"/results_history/prob1"+ "/"+i))[2]

    loss = ""
    if i.split("_")[1] == "ambs":
        loss = "objective"
    else:
        loss = "loss"

    l = 0
    for j in files:
        if l >= 5:
            break
        data = pd.read_csv(os. getcwd()+"/results_history/prob1"+ "/"+i+"/"+j)
        dataScore[i.split("_")[1]][:, l] = data[loss]

        k = 0
        for n in data["model"]:
            if n =="randomForestRegressor":
                dataModels[i.split("_")[1]][k, l] = int(0)
            elif n =="kneighborsRegressor":
                dataModels[i.split("_")[1]][k, l] = int(1)
            else:
                dataModels[i.split("_")[1]][k, l] = int(2)

            k = k + 1



        l = l + 1

"""
================================================================================
"""
n=0
randomForestRegressorRun = list()
kneighborsRegressorRun= list()
nnRegressorRun = list()

while n < 5:
    randomForestRegressor = list()
    kneighborsRegressor= list()
    nnRegressor = list()

    l=0
    for i in dataModels["ambs"][:,n]:
        if i == 0:
            randomForestRegressor.append( True)
            kneighborsRegressor.append(False)
            nnRegressor.append(False)

        elif i == 1:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(True)
            nnRegressor.append(False)

        else:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(False)
            nnRegressor.append(True)

    randomForestRegressorRun.append(randomForestRegressor)
    kneighborsRegressorRun.append(kneighborsRegressor)
    nnRegressorRun.append(nnRegressor)

    n =  n + 1


x2=np.arange(100)

n = 0
while n < 5:
    plt.scatter(x2[randomForestRegressorRun[n]], (dataScore["ambs"][:,n])[randomForestRegressorRun[n]],s=5, label='randomForestRegressor'if n == 0 else "", c='b')
    plt.scatter(x2[kneighborsRegressorRun[n]], (dataScore["ambs"][:,n])[kneighborsRegressorRun[n]],s=5, label='kneighborsRegressor'if n == 0 else "", c='r')
    plt.scatter(x2[nnRegressorRun[n]], (dataScore["ambs"][:,n])[nnRegressorRun[n]],s=5 ,label='nnRegressor'if n == 0 else "", c='g')
    n = n + 1
    #plt.plot(x2,dataScore["ambs"][:,n-1], "black", linewidth=0.5)


plt.legend()
plt.title("Ambs Validation Score through evaluations")
plt.xlabel("Evaluations")
plt.ylabel("Validation Score")
plt.savefig('ScatterClassPlotAmbsAll.png')
plt.clf()
"""
================================================================================
"""
n=0
randomForestRegressorRun = list()
kneighborsRegressorRun= list()
nnRegressorRun = list()

while n < 5:
    randomForestRegressor = list()
    kneighborsRegressor= list()
    nnRegressor = list()

    l=0
    for i in dataModels["ga"][:,n]:
        if i == 0:
            randomForestRegressor.append( True)
            kneighborsRegressor.append(False)
            nnRegressor.append(False)

        elif i == 1:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(True)
            nnRegressor.append(False)

        else:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(False)
            nnRegressor.append(True)

    randomForestRegressorRun.append(randomForestRegressor)
    kneighborsRegressorRun.append(kneighborsRegressor)
    nnRegressorRun.append(nnRegressor)

    n =  n + 1


x2=np.arange(100)

n = 0
while n < 5:
    plt.scatter(x2[randomForestRegressorRun[n]], (dataScore["ga"][:,n])[randomForestRegressorRun[n]],s=5, label='randomForestRegressor'if n == 0 else "", c='b')
    plt.scatter(x2[kneighborsRegressorRun[n]], (dataScore["ga"][:,n])[kneighborsRegressorRun[n]],s=5, label='kneighborsRegressor'if n == 0 else "", c='r')
    plt.scatter(x2[nnRegressorRun[n]], (dataScore["ga"][:,n])[nnRegressorRun[n]],s=5 ,label='nnRegressor'if n == 0 else "", c='g')
    n = n + 1
    #plt.plot(x2,dataScore["ga"][:,n-1], "black", linewidth=0.5)



plt.legend()
plt.title("Ga Validation Score through evaluations")
plt.xlabel("Evaluations")
plt.ylabel("Validation Score")

plt.savefig('ScatterClassPlotGaAll.png')
plt.clf()
"""
================================================================================
"""
n=0
randomForestRegressorRun = list()
kneighborsRegressorRun= list()
nnRegressorRun = list()

while n < 5:
    randomForestRegressor = list()
    kneighborsRegressor= list()
    nnRegressor = list()

    l=0
    for i in dataModels["pso"][:,n]:
        if i == 0:
            randomForestRegressor.append( True)
            kneighborsRegressor.append(False)
            nnRegressor.append(False)

        elif i == 1:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(True)
            nnRegressor.append(False)

        else:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(False)
            nnRegressor.append(True)

    randomForestRegressorRun.append(randomForestRegressor)
    kneighborsRegressorRun.append(kneighborsRegressor)
    nnRegressorRun.append(nnRegressor)

    n =  n + 1


x2=np.arange(100)

n = 0
while n < 5:
    plt.scatter(x2[randomForestRegressorRun[n]], (dataScore["pso"][:,n])[randomForestRegressorRun[n]],s=5, label='randomForestRegressor'if n == 0 else "", c='b')
    plt.scatter(x2[kneighborsRegressorRun[n]], (dataScore["pso"][:,n])[kneighborsRegressorRun[n]],s=5, label='kneighborsRegressor'if n == 0 else "", c='r')
    plt.scatter(x2[nnRegressorRun[n]], (dataScore["pso"][:,n])[nnRegressorRun[n]],s=5 ,label='nnRegressor'if n == 0 else "", c='g')
    n = n + 1
    #plt.plot(x2,dataScore["pso"][:,n-1], "black", linewidth=0.5)



plt.legend()
plt.title("Pso Validation Score through evaluations")
plt.xlabel("Evaluations")
plt.ylabel("Validation Score")

plt.savefig('ScatterClassPlotPsoAll.png')
plt.clf()

"""
================================================================================
"""
n=0
randomForestRegressorRun = list()
kneighborsRegressorRun= list()
nnRegressorRun = list()

while n < 5:
    randomForestRegressor = list()
    kneighborsRegressor= list()
    nnRegressor = list()

    l=0
    for i in dataModels["rs"][:,n]:
        if i == 0:
            randomForestRegressor.append( True)
            kneighborsRegressor.append(False)
            nnRegressor.append(False)

        elif i == 1:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(True)
            nnRegressor.append(False)

        else:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(False)
            nnRegressor.append(True)

    randomForestRegressorRun.append(randomForestRegressor)
    kneighborsRegressorRun.append(kneighborsRegressor)
    nnRegressorRun.append(nnRegressor)

    n =  n + 1


x2=np.arange(100)

n = 0
while n < 5:
    plt.scatter(x2[randomForestRegressorRun[n]], (dataScore["rs"][:,n])[randomForestRegressorRun[n]],s=5, label='randomForestRegressor'if n == 0 else "", c='b')
    plt.scatter(x2[kneighborsRegressorRun[n]], (dataScore["rs"][:,n])[kneighborsRegressorRun[n]],s=5, label='kneighborsRegressor'if n == 0 else "", c='r')
    plt.scatter(x2[nnRegressorRun[n]], (dataScore["rs"][:,n])[nnRegressorRun[n]],s=5 ,label='nnRegressor'if n == 0 else "", c='g')
    n = n + 1
    #plt.plot(x2,dataScore["rs"][:,n-1], "black", linewidth=0.5)



plt.legend()
plt.title("Rs Validation Score through evaluations")
plt.xlabel("Evaluations")
plt.ylabel("Validation Score")

plt.savefig('ScatterClassPlotRsAll.png')
plt.clf()

"""
================================================================================
"""
n=0
randomForestRegressorRun = list()
kneighborsRegressorRun= list()
nnRegressorRun = list()

while n < 5:
    randomForestRegressor = list()
    kneighborsRegressor= list()
    nnRegressor = list()

    l=0
    for i in dataModels["tpe"][:,n]:
        if i == 0:
            randomForestRegressor.append( True)
            kneighborsRegressor.append(False)
            nnRegressor.append(False)

        elif i == 1:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(True)
            nnRegressor.append(False)

        else:
            randomForestRegressor.append(False)
            kneighborsRegressor.append(False)
            nnRegressor.append(True)

    randomForestRegressorRun.append(randomForestRegressor)
    kneighborsRegressorRun.append(kneighborsRegressor)
    nnRegressorRun.append(nnRegressor)

    n =  n + 1


x2=np.arange(100)

n = 0
while n < 5:
    plt.scatter(x2[randomForestRegressorRun[n]], (dataScore["tpe"][:,n])[randomForestRegressorRun[n]],s=5, label='randomForestRegressor'if n == 0 else "", c='b')
    plt.scatter(x2[kneighborsRegressorRun[n]], (dataScore["tpe"][:,n])[kneighborsRegressorRun[n]],s=5, label='kneighborsRegressor'if n == 0 else "", c='r')
    plt.scatter(x2[nnRegressorRun[n]], (dataScore["tpe"][:,n])[nnRegressorRun[n]],s=5 ,label='nnRegressor'if n == 0 else "", c='g')
    n = n + 1
    #plt.plot(x2,dataScore["tpe"][:,n-1], "black", linewidth=0.5)




plt.legend()
plt.title("Tpe Validation Score through evaluations")
plt.xlabel("Evaluations")
plt.ylabel("Validation Score")

plt.savefig('ScatterClassPlotTpeAll.png')
plt.clf()
