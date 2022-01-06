import numpy as np
import utils.runner as runner
from abc import ABCMeta, abstractmethod
import importlib


class HPT_algo:
    def __init__(self, problem, max_evals, argv):

        prob = importlib.import_module(problem + ".prob_space", package=None)
        self.problemConfig = prob.Problem
        self.mod_run = importlib.import_module(problem + ".model_run", package=None)
        self.dim_space = len(self.problemConfig._space.get_hyperparameters())


        if "s" in max_evals:
            self.time = int(max_evals.replace("s",""))
            self.evals =  100000
        else:
            self.evals = int(max_evals)
        self.train_X = np.load(problem + "/datasets/custom/train_X.npy", allow_pickle=True)
        self.train_y = np.load(problem + "/datasets/custom/train_y.npy", allow_pickle=True)
        self.valid_X = np.load(problem + "/datasets/custom/valid_X.npy", allow_pickle=True)
        self.valid_y = np.load(problem + "/datasets/custom/valid_y.npy", allow_pickle=True)
        self.feat_names = np.load(problem + "/datasets/custom/names.npy", allow_pickle=True)

        self.prob = problem

        self.argv = argv

        """
        self.current_model = None
        self.next_step_name = argv.pop(0)
        self.next_step_run = getattr(runner,"run_" + self.next_step_name)
        """
