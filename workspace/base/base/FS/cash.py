import numpy as np
import importlib
from FS.baseFS import  FS_algo
import textwrap
import  subprocess




class algorithm(FS_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)




    def iterate(self):
        prob  = importlib.import_module(self.prob +".prob_space", package=None)
        problem = prob.Problem

        file = self.prob +"/prob_space.py"

        lines = open(file, 'r').readlines()
        initial_len = len(lines)
        count = len(lines)
        start = "Problem.add_starting_point("

        for j in range(self.train_X.shape[1]):
            problem.add_hyperparameter(["1", "0"], "feature_" +str(j) )
            if count == initial_len:
                start = start + " feature_" +str(j)+"=\"1\""
            else:
                start = start + ", feature_" +str(j)+"=\"1\""
            count = count + 1

        self.trans_train_X = self.train_X
        self.trans_valid_X = self.valid_X
        self.evals =  0
        
        self.start = start

    def run_ambs(self):
        prob  = importlib.import_module(self.prob +".prob_space", package=None)
        problem = prob.Problem

        file = self.prob +"/prob_space.py"


        lines = open(file, 'r').readlines()
        initial_len = len(lines)
        count = len(lines)
        start = "Problem.add_starting_point("
        for j in range(self.train_X.shape[1]):
            lines.insert(count, "Problem.add_hyperparameter([\"1\", \"0\"], \"feature_" +str(j) + "\") \n")
            if count == initial_len:
                start = start + " feature_" +str(j)+"=\"1\""
            else:
                start = start + ", feature_" +str(j)+"=\"1\""
            count = count + 1


        out = open(file, 'w')
        out.writelines(lines)
        out.close()


        self.trans_train_X = self.train_X
        self.trans_valid_X = self.valid_X
        self.evals =  0

        self.start = start
