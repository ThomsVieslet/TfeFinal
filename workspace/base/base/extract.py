import numpy as np
import importlib
import pandas as pd
import os
import  subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("SummaryCl_500_RS.csv")

results = {"ambs": np.zeros(25), "ga": np.zeros(25), "pso": np.zeros(25), "rs": np.zeros(25), "tpe": np.zeros(25)}
i = 0

while i < len(data["Problem"]):
    tmp = (data["ambs"][i]).split(" ")[1]
    test = ""
    results["ambs"][i] = float(test.join(list(tmp)[1:4]))


    tmp = (data["ga"][i]).split(" ")[1]
    test = ""
    results["ga"][i] = float(test.join(list(tmp)[1:4]))


    tmp = (data["pso"][i]).split(" ")[1]
    test = ""
    results["pso"][i] = float(test.join(list(tmp)[1:4]))


    tmp = (data["rs"][i]).split(" ")[1]
    test = ""
    results["rs"][i] = float(test.join(list(tmp)[1:4]))

    tmp = (data["tpe"][i]).split(" ")[1]
    test = ""
    results["tpe"][i] = float(test.join(list(tmp)[1:4]))

    i = i + 1



for j in results:
    print(j + ": ")
    print(np.mean(results[j]))
    print(np.std(results[j]))
