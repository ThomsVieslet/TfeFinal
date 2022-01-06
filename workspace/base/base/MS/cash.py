import numpy as np
import importlib
from MS.baseMS import  MS_algo
import textwrap
import  subprocess

import re




class algorithm(MS_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)



    def iterate(self):
        prob  = importlib.import_module(self.prob +".prob_space", package=None)
        problem = prob.Problem

        file = self.prob +"/prob_space.py"


        new_models = []
        for i in self.models:
            pattern = r'[\n]'
            mod1_str2 = re.sub(pattern, '', i)
            new_models.append(mod1_str2)


        self.models = new_models

        model = problem.add_hyperparameter(self.models, "model" )

        j = 0
        for i in self.models:

            mod_add = importlib.import_module("MS.model_utils.add_"+ i, package=None)


            self.start = self.start + mod_add.run(problem, model)


        self.evals =  0

    def run_ambs(self):

        file = self.prob +"/prob_space.py"

        new_models = []
        for i in self.models:
            pattern = r'[\n]'
            mod1_str2 = re.sub(pattern, '', i)
            new_models.append(mod1_str2)

        self.models = new_models



        lines = open(file, 'r').readlines()
        initial_len = len(lines)
        count = len(lines) + 1

        lines.insert(count, "model = Problem.add_hyperparameter("+str(self.models)+", \"model\" )\n")
        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        for i in self.models:
            mod_add = importlib.import_module("MS.model_utils.ambs_write", package=None)
            self.start = self.start + mod_add.run(i, file)
