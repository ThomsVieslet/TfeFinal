import numpy as np
import struct
from utils.problemUtils import from_problem_to_hyperopt
from operator import attrgetter
import random
from HPT.baseHPT import HPT_algo

import csv
import nevergrad as ng
import math
import time
import random
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os


"""
********************************************************************************
algorithm:
    * inherits from HPT_algo
    * implements the run method which use hyperopt to performs a search with TPE
********************************************************************************
"""

tic = 0

class algorithm(HPT_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)
        self.results =  np.array((self.evals, self.dim_space))
        self.max_evals = max_evals

    def monitor_callback(self, params, score):
        print("incall")
        #self.results[self.max_evals-self.evals, :] =


    def objective(self, params):

        feat = []
        for i in params:
            if "feature_" in i:
                feat.append(params[i])

        only_nul =  True
        for j in feat:
            if j != '0':
                only_nul = False
                break
        if only_nul:
            self.monitor_callback(params, 0)
            return 0
        else:
            params2 = {}
            for i in params:
                if i != "model":
                    params2[i] = params[i]
                else:
                    k=0
                    for j in params["model"]:

                        if j == "model_name":
                            params2["model"] = (params["model"])[j]
                        else:
                            if str(type((params["model"])[j])) == "<class 'float'>":
                                params2[j] = int((params["model"])[j])
                            else:
                                params2[j] = (params["model"])[j]
                        k = k + 1

            params2["loss"] = -self.mod_run.run(params2)
            params2["status"] = STATUS_OK
            params2["time"] = time.perf_counter() - tic

            #elf.monitor_callback(params2, score)
            return params2


    def run(self):
        global tic
        tic = time.perf_counter()
        sp = from_problem_to_hyperopt(self.problemConfig)
        trials = Trials()
        best = None
        if not hasattr(self, "time"):
            best = fmin(self.objective,
                space=sp,
                algo=tpe.suggest,
                max_evals=self.evals,
                trials=trials)
        else:
            best = fmin(self.objective,
                space=sp,
                algo=tpe.suggest,
                max_evals=10000,
                trials=trials)


        names = self.problemConfig._space.get_hyperparameter_names()
        results = pd.DataFrame(columns = names)


        for i in trials.results:
            results = results.append(i, ignore_index=True)


        results["loss"] = -results["loss"]
        results.fillna("None", inplace=True)
        results.to_csv(self.prob + '/results/results.csv')
