import csv
import os


"""
********************************************************************************
    * 2 functions used to deal with csv files
********************************************************************************
"""

def nameIt(path, name):
    path2 = path.split(".")[0] + "_"+ name + "." + path.split(".")[1]
    os.rename(path,path2)



def reNameAndSort(problem, step):

    pathDir = problem + "/results"
    fileList = os.listdir(pathDir)
    nbFiles = len(fileList)

    for i in fileList:
        if len(i.split("_")) == 2:
            nameIt(pathDir + "/"+ i, str(nbFiles-1))
