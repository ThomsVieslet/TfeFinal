import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd



def visualize_losses_time(toVisualize):


    fileList = os.listdir(toVisualize)
    nbFiles = len(fileList)
    viridis = cm.get_cmap('viridis', nbFiles)
    listColor = viridis(range(nbFiles))



    dataframes = []

    for i in fileList:
        data = pd.read_csv(toVisualize +"/"+ i)
        dataframes.append(data)


    n = 0


    index_best = 0
    for i in dataframes:
        time_step = 9000/len(i["loss_best"])
        best_loss = np.zeros(len(i["loss_best"]))
        curr_best = float('-inf')
        k = 0
        time_axis = np.zeros(len(i["loss_best"]), dtype=int)
        time = 0
        for j in i["loss_best"]:
            if j > curr_best:
                curr_best = j
            best_loss[k] = curr_best
            time_axis[k] = int(time)
            time = time + time_step
            k =  k + 1


        x = range(len(i["loss_best"]))
        #index_best = np.argmax(i["loss"])
        #plt.plot(i["elapsed_sec"], i["loss_best"], label = "loss tested" , color=listColor[n], linestyle="dotted")
        plt.plot(x, i["loss_best"], label = "best_loss" , color=listColor[n])
        #plt.plot(x, i["loss_best"], label = "loss tested" , color=listColor[j], linestyle="dotted")

        n = n + 1

    plt.title("Score of pso through time ", loc='right')
    plt.xlabel("Evals")
    plt.legend()
    plt.savefig(toVisualize +"/"+ toVisualize + ".png")





if __name__ == "__main__":
    visualize_losses_time("FS_cash_pso_evals_180")
