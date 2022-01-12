import multiprocessing as mp
import sys
import time
import  subprocess
import threading
import os

"""
First script to be run. It launch is simulation in a different process.
We have one simulation for each task. For each simulation, parser.py is called
with the following args:
    * the problem (the task forlder)
    * the step (a specific step or CASH)
    * the optimization method
    * the budget 
"""

def simulation(p, file, prob, task, meth, budget):
    script_descriptor = open(file)

    a_script = script_descriptor.read()

    largs = [p, file, prob, task, meth, budget]

    sys.argv= largs
    sys.argv.pop(0)


    exec(a_script)

    if meth != "ambs":

        files = next(os.walk(prob + "/results/" + task + "_" + meth + "_"+ budget))[2]


        bashCmd2 = ["mv", prob + "/results/results.csv", prob + "/results/"+ str(len(files)+1) + "_results.csv"]

        process2 = subprocess.Popen(bashCmd2, stdout=subprocess.PIPE)

        output2, error2 = process2.communicate()

        bashCmd2 = ["mv", prob + "/results/"+ str(len(files)+1) + "_results.csv", prob + "/results/" + task + "_" + meth + "_"+ budget]

        process2 = subprocess.Popen(bashCmd2, stdout=subprocess.PIPE)

        output2, error2 = process2.communicate()

    else:
        files = next(os.walk(prob + "/results/" + task + "_" + meth + "_"+ budget))[2]

        bashCmd2 = ["mv", "results_" + str(prob) + ".csv", prob + "/results/"+ str(len(files)+1) + "_results.csv"]

        process2 = subprocess.Popen(bashCmd2, stdout=subprocess.PIPE)

        output2, error2 = process2.communicate()

        bashCmd2 = ["mv", prob + "/results/"+ str(len(files)+1) + "_results.csv", prob + "/results/" + task + "_" + meth + "_"+ budget]

        process2 = subprocess.Popen(bashCmd2, stdout=subprocess.PIPE)

        output2, error2 = process2.communicate()



    script_descriptor.close()


pool = mp.Pool(32)


arg1 = ["python", "parser.py", "prob1", "CASH", "ga", "100"]

results = pool.apply_async(simulation, arg1)


arg2 = ["python", "parser.py", "prob2", "CASH", "ga", "100"]

results = pool.apply_async(simulation, arg2)


arg3 = ["python", "parser.py", "prob3", "CASH", "ga", "100"]

results = pool.apply_async(simulation, arg3)


arg4 = ["python", "parser.py", "prob4", "CASH", "ga", "100"]

results = pool.apply_async(simulation, arg4)

arg5 = ["python", "parser.py", "prob5", "CASH", "ga", "100"]

results = pool.apply_async(simulation, arg5)

argA = ["python", "parser.py", "probA", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argA)

argB = ["python", "parser.py", "probB", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argB)

argC = ["python", "parser.py", "probC", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argC)

argD = ["python", "parser.py", "probD", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argD)

argE = ["python", "parser.py", "probE", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argE)

argF = ["python", "parser.py", "probF", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argF)

argG = ["python", "parser.py", "probG", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argG)

argH = ["python", "parser.py", "probH", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argH)

argI = ["python", "parser.py", "probI", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argI)

argJ = ["python", "parser.py", "probJ", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argJ)

argK = ["python", "parser.py", "probK", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argK)

argL = ["python", "parser.py", "probL", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argL)

argM = ["python", "parser.py", "probM", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argM)

argN = ["python", "parser.py", "probN", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argN)

argO = ["python", "parser.py", "probO", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argO)

argP = ["python", "parser.py", "probP", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argP)

argQ = ["python", "parser.py", "probQ", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argQ)

argR = ["python", "parser.py", "probR", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argR)

argS = ["python", "parser.py", "probS", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argS)

argT = ["python", "parser.py", "probT", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argT)

argU = ["python", "parser.py", "probU", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argU)

argV = ["python", "parser.py", "probV", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argV)

argW = ["python", "parser.py", "probW", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argW)

argX = ["python", "parser.py", "probX", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argX)

argY = ["python", "parser.py", "probY", "CASH", "ga", "100"]

results = pool.apply_async(simulation, argY)












pool.close()
pool.join()
