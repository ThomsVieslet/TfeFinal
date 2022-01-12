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
import colorsys


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

sims = next(os.walk(os. getcwd()+"/results_history/probR"))[1]

dataScore = {"ambs": np.zeros((100, 5)),  "ga": np.zeros((100, 5)), "pso": np.zeros((100, 5)), "rs": np.zeros((100, 5)), "tpe": np.zeros((100, 5))}
dataFeats = {"ambs": np.zeros((100, 5), dtype=int),  "ga": np.zeros((100, 5), dtype=int), "pso": np.zeros((100, 5), dtype=int), "rs": np.zeros((100, 5), dtype=int), "tpe": np.zeros((100, 5), dtype=int)}

for i in sims:
    files = next(os.walk(os. getcwd()+"/results_history/probR"+ "/"+i))[2]

    loss = ""
    if i.split("_")[1] == "ambs":
        loss = "objective"
    else:
        loss = "loss"


    l = 0
    for j in  files:
        if l >= 5:
            break
        data = pd.read_csv(os. getcwd()+"/results_history/probR"+ "/"+i+"/"+j)
        dataScore[i.split("_")[1]][:,l]= data[loss]

        k = 0
        for n in data["model"]:
            cntFeat = 0
            cntFeatTaken = 0
            for m in data.columns:

                if "feature_" in m:

                    if (data[m])[k] == 0.0:
                        cntFeat = cntFeat + 1
                    elif (data[m])[k] == 1.0:
                        cntFeat = cntFeat + 1
                        cntFeatTaken = cntFeatTaken + 1

            ratio = cntFeatTaken/cntFeat
            if ratio < 0.1:
                dataFeats[i.split("_")[1]][k, l] = int(0)
            elif 0.1 <= ratio < 0.2:
                dataFeats[i.split("_")[1]][k, l] = int(1)
            elif 0.2 <= ratio < 0.3:
                dataFeats[i.split("_")[1]][k, l] = int(2)
            elif 0.3 <= ratio < 0.4:
                dataFeats[i.split("_")[1]][k, l] = int(3)
            elif 0.4 <= ratio < 0.5:
                dataFeats[i.split("_")[1]][k, l] = int(4)
            elif 0.5 <= ratio < 0.6:
                dataFeats[i.split("_")[1]][k, l] = int(5)
            elif 0.6 <= ratio < 0.7:
                dataFeats[i.split("_")[1]][k, l] = int(6)
            elif 0.7 <= ratio < 0.8:
                dataFeats[i.split("_")[1]][k, l] = int(7)
            elif 0.8 <= ratio < 0.9:
                dataFeats[i.split("_")[1]][k, l] = int(8)
            else:
                dataFeats[i.split("_")[1]][k, l] = int(9)


            k = k + 1



        l = l + 1



feat0Run = list()
feat1Run = list()
feat2Run = list()
feat3Run = list()
feat4Run = list()
feat5Run = list()
feat6Run = list()
feat7Run = list()
feat8Run = list()
feat9Run = list()
n = 0

while n < 5:

    feat0 = list()
    feat1 = list()
    feat2 = list()
    feat3 = list()
    feat4 = list()
    feat5 = list()
    feat6 = list()
    feat7 = list()
    feat8 = list()
    feat9 = list()


    l=0
    for i in dataFeats["ambs"][:, n]:
        if i == 0:
            feat0.append(True)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)

        elif i == 1:
            feat0.append(False)
            feat1.append(True)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)

        elif i == 2:
            feat0.append(False)
            feat1.append(False)
            feat2.append(True)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)

        elif i == 3:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(True)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)
        elif i == 4:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(True)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)

        elif i == 5:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(True)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)

        elif i == 6:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(True)
            feat7.append(False)
            feat8.append(False)
            feat9.append(False)

        elif i == 7:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(True)
            feat8.append(False)
            feat9.append(False)

        elif i == 8:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(True)
            feat9.append(False)

        else:
            feat0.append(False)
            feat1.append(False)
            feat2.append(False)
            feat3.append(False)
            feat4.append(False)
            feat5.append(False)
            feat6.append(False)
            feat7.append(False)
            feat8.append(False)
            feat9.append(True)

    feat0Run.append(feat0)
    feat1Run.append(feat1)
    feat2Run.append(feat2)
    feat3Run.append(feat3)
    feat4Run.append(feat4)
    feat5Run.append(feat5)
    feat6Run.append(feat6)
    feat7Run.append(feat7)
    feat8Run.append(feat8)
    feat9Run.append(feat9)

    n = n + 1

x2=np.arange(100)


colors = ["thistle","mediumorchid", "darkviolet", "blueviolet", "mediumpurple", "mediumslateblue", "slateblue", "blue", "mediumblue", "midnightblue"]

print(len((dataScore["ambs"][feat3Run[0]])))
print(feat2Run[0])
n = 0
while n == 0:
    plt.scatter(x2[feat0Run[n]], (dataScore["ambs"][:, n])[feat0Run[n]],s=15, label='ratio < 0.1'if n == 0 else "", c=colors[0])
    plt.scatter(x2[feat1Run[n]], (dataScore["ambs"][:, n])[feat1Run[n]],s=15, label='0.1 <= ratio < 0.2'if n == 0 else "", c=colors[1])
    plt.scatter(x2[feat2Run[n]], (dataScore["ambs"][:, n])[feat2Run[n]],s=15, label='0.2 <= ratio < 0.3'if n == 0 else "", c=colors[2])
    plt.scatter(x2[feat3Run[n]], (dataScore["ambs"][:, n])[feat3Run[n]],s=15, label='0.3 <= ratio < 0.4'if n == 0 else "", c=colors[3])
    plt.scatter(x2[feat4Run[n]], (dataScore["ambs"][:, n])[feat4Run[n]],s=15, label='0.4 <= ratio < 0.5'if n == 0 else "", c=colors[4])
    plt.scatter(x2[feat5Run[n]], (dataScore["ambs"][:, n])[feat5Run[n]],s=15, label='0.5 <= ratio < 0.6'if n == 0 else "", c=colors[5])
    plt.scatter(x2[feat6Run[n]], (dataScore["ambs"][:, n])[feat6Run[n]],s=15, label='0.6 <= ratio < 0.7'if n == 0 else "", c=colors[6])
    plt.scatter(x2[feat7Run[n]], (dataScore["ambs"][:, n])[feat7Run[n]],s=15, label='0.7 <= ratio < 0.8'if n == 0 else "", c=colors[7])
    plt.scatter(x2[feat8Run[n]], (dataScore["ambs"][:, n])[feat8Run[n]],s=15, label='0.8 <= ratio < 0.9'if n == 0 else "", c=colors[8])
    plt.scatter(x2[feat9Run[n]], (dataScore["ambs"][:, n])[feat9Run[n]],s=15, label='0.9 <= ratio <= 1'if n == 0 else "", c=colors[9])
    plt.plot(x2,dataScore["ambs"][:, n], "black", linewidth=0.5)


    n = n + 1
    #plt.plot(x2,dataScore["ambs"], "black", linewidth=0.5)


plt.legend()
plt.title("Ambs Validation Score through evaluations")
plt.xlabel("Evaluations")
plt.ylabel("Validation Score")
plt.savefig('ScatterClassPlotAmbsAllLines.png')
plt.clf()
