import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import numpy as np
import csv
import importlib
import  subprocess
import os

import utils.csvUtils as cU
from threading import Timer
import deephyper
import sys

"""
********************************************************************************
    * Only method not inheriting from HPT_algo because built before
    * Run the DeepHyper command with the appropriate args
********************************************************************************
"""
class algorithm:
    def __init__(self, problem, max_evals, argv):

        self.problem = problem
        self.problem_path = "base." + problem + ".prob_space.Problem"
        self.model_run_path = "base." + problem + ".model_run.run"
        #self.mod_run = importlib.import_module(problem + ".model_run", package=None)
        if "s" in max_evals:
            self.time = int(max_evals.replace("s",""))
        else:
            self.evals = int(max_evals)

    def run(self):
        if hasattr(self, 'evals'):


            bashCmd = ["deephyper", "hps", "ambs", "--problem", self.problem_path, "--run", self.model_run_path,
                "--max-evals", str(self.evals), "--evaluator" ,"subprocess", "--n-jobs", "-1"]

            process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)

            output, error = process.communicate()
            out0 = ""

            for i in output:
                if i == 10:
                    i = chr(10) + chr(32)
                    out0 = out0 + i
                else:
                    out0 = out0 + chr(i)
            print(out0)
        else:

            kill = lambda process: process.kill()
            bashCmd = ["deephyper", "hps", "ambs", "--problem", self.problem_path, "--run", self.model_run_path,
                "--max-evals", "100000", "--evaluator" ,"subprocess", "--n-jobs", "-1"]

            process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)


            my_timer = Timer(self.time, kill, [process])
            my_timer.start()
            output, error = process.communicate()
