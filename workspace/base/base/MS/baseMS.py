import numpy as np
import utils.runner as runner
from abc import ABCMeta, abstractmethod

"""
********************************************************************************
MS_algo:
    * the generic class for Model Selection methods
    * implement generic run method
********************************************************************************
"""

class MS_algo:
    def __init__(self, problem, max_evals, argv):
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

        self.trans_train_X = None
        self.trans_valid_X = None
        self.new_features_names = None


        self.prob = problem
        self.current_model = None
        self.argv = None

        self.start = ""

        if  bool(argv):
            self.next_step_name = argv.pop(0)
            self.argv = argv
            self.next_step_run = getattr(runner,"run_" + self.next_step_name)


        self.models = []
        with open(self.prob + '/content/models.txt') as f:
            lines = f.readlines()

            for i in lines:
                self.models.append(i)

    def run(self):
        while self.evals > 0:
            self.iterate()


            self.evals = self.evals - 1




    @abstractmethod
    def iterate(self):
        raise NotImplementedError("Must override a iterate method ")
