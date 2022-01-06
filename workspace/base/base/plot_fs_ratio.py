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

sims = next(os.walk(os. getcwd()+"/results_history/prob1"))[1]

dataScore = {"ambs": np.zeros((100, 5)),  "ga": np.zeros((100, 5)), "pso": np.zeros((100, 5)), "rs": np.zeros((100, 5)), "tpe": np.zeros((100, 5))}
dataFeats = {"ambs": np.zeros((100, 5)),  "ga": np.zeros((100, 5)), "pso": np.zeros((100, 5)), "rs": np.zeros((100, 5)), "tpe": np.zeros((100, 5))}

for i in sims:
    files = next(os.walk(os. getcwd()+"/results_history/prob1"+ "/"+i))[2]

    loss = ""
    if i.split("_")[1] == "ambs":
        loss = "objective"
    else:
        loss = "loss"


    l = 0
    for j in  files:
        if l >= 5:
            break
        data = pd.read_csv(os. getcwd()+"/results_history/prob1"+ "/"+i+"/"+j)
        dataScore[i.split("_")[1]][:,l]= data[loss]

        k = 0
        for n in data["model"]:
            cntFeat = 0
            cntFeatTaken = 0
            for m in data.columns:

                if "feature_" in m:

                    if "feature_" in m:
                        cntFeat = cntFeat + 1
                        if (data[m])[k] == 1.0:
                            cntFeatTaken = cntFeatTaken + 1

            ratio = float(cntFeatTaken/cntFeat)
            dataFeats[i.split("_")[1]][k, l] = ratio

            k = k + 1



        l = l + 1



x2=np.arange(100)

plt.plot(x2, dataFeats["ambs"][: , 0], linewidth=0.5)
plt.title("Ambs Selected Feature Ratio through evaluation")
plt.ylabel("Ratio")
plt.xlabel("Evaluations")
plt.savefig('ratioAmbs.png')
plt.clf()


plt.plot(x2, dataFeats["ga"][: , 0], linewidth=0.5)
plt.title("Ga Selected Feature Ratio through evaluation")
plt.ylabel("Ratio")
plt.xlabel("Evaluations")

plt.savefig('ratioGa.png')
plt.clf()


plt.plot(x2, dataFeats["pso"][: , 0], linewidth=0.5)
plt.title("Pso Selected Feature Ratio through evaluation")
plt.ylabel("Ratio")
plt.xlabel("Evaluations")

plt.savefig('ratioPso.png')
plt.clf()


plt.plot(x2, dataFeats["rs"][: , 0], linewidth=0.5)
plt.title("Rs Selected Feature Ratio through evaluation")
plt.ylabel("Ratio")
plt.xlabel("Evaluations")

plt.savefig('ratioRs.png')
plt.clf()


plt.plot(x2, dataFeats["tpe"][: , 0], linewidth=0.5)
plt.title("Tpe Selected Feature Ratio through evaluation")
plt.ylabel("Ratio")
plt.xlabel("Evaluations")

plt.savefig('ratioTpe.png')
plt.clf()
