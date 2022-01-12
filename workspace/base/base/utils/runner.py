import sys
import  subprocess
import utils.csvUtils as cU
import importlib

import time
import multiprocessing
import os
import psutil

"""
********************************************************************************
run_FS:
    * Run Feature Selection that can be followed by other steps with appropriate
        args.
********************************************************************************
"""
def run_FS(problem, method, budget, argv):
    meth = importlib.import_module("FS." + method , package=None)
    algo = meth.algorithm(problem, budget, argv)
    algo.run()

"""
********************************************************************************
run_MS:
    * Run Model Selection that can be followed by other steps with approriate
        args.
********************************************************************************
"""
def run_MS(problem, method, budget, argv):
    meth = importlib.import_module("MS." + method , package=None)
    algo = meth.algorithm(problem, budget, argv)
    algo.run()



"""
********************************************************************************
run_HPT:
    * Run Hyperparameter Tuning (optimization) that can be followed by other
     steps with approriate args.
********************************************************************************
"""
def run_HPT(problem, method, budget, argv):
    meth = importlib.import_module("HPT." + method , package=None)
    algo = meth.algorithm(problem, budget, argv)
    algo.run()


"""
********************************************************************************
run_CASH:
    * Run the CASH problem with Features with Feature Selection with the
        approriate args.
********************************************************************************
"""
def run_CASH(problem, method, budget, argv):
    meth = importlib.import_module("FS.cash"  , package=None)
    algo = meth.algorithm(problem, budget, argv)

    start = ""
    if method == "ambs":
        algo.run_ambs()
        start = start + algo.start

    else:
        algo.run()
        start = start + algo.start

    meth = importlib.import_module("MS.cash"  , package=None)
    algo = meth.algorithm(problem, budget, argv)

    if method == "ambs":
        file = problem +"/prob_space.py"
        algo.run_ambs()
        start = start + algo.start
        start = start + ", model=\"randomForestRegressor\")"
        lines = open(file, 'r').readlines()
        initial_len = len(lines)
        count = len(lines)
        lines.insert(count+1, start)
        """
        out = open(file, 'w')
        out.writelines(lines)
        out.close()
        """



    else:
        algo.run()
        start = start + algo.start
        start = start + ", model=\"randomForestRegressor\")"
        prob  = importlib.import_module(algo.prob +".prob_space", package=None)
        Problem = prob.Problem

        #exec(start)


    run_HPT(problem, method, budget, argv)

    file = problem +"/prob_space.py"


    lines = open(file, 'r').readlines()
    new_lines = []
    for j in lines:
        if "Problem = HpProblem()"  in j:
            new_lines.append(j)
        elif j =="\n":
            new_lines.append(j)
        elif "import" in j:
            new_lines.append(j)




    out = open(file, 'w')
    out.writelines(new_lines)
    out.close()
