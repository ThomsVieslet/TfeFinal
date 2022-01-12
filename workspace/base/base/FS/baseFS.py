import numpy as np
import utils.runner as runner
from abc import ABCMeta, abstractmethod


"""
********************************************************************************
FS_algo:
    * the generic class for Features Selection methods
    * implement generic run method
********************************************************************************
"""

class FS_algo:
    def __init__(self, problem, max_evals, argv):
        if "s" in max_evals:
            self.time = int(max_evals.replace("s",""))
            self.evals = 100000
        else:
            self.evals = int(max_evals)

        self.train_X = np.load(problem + "/datasets/original/train_X.npy", allow_pickle=True)
        self.train_y = np.load(problem + "/datasets/original/train_y.npy", allow_pickle=True)
        self.valid_X = np.load(problem + "/datasets/original/valid_X.npy", allow_pickle=True)
        self.valid_y = np.load(problem + "/datasets/original/valid_y.npy", allow_pickle=True)
        self.feat_names = np.load(problem + "/datasets/original/names.npy", allow_pickle=True)

        self.trans_train_X = None
        self.trans_valid_X = None
        self.new_features_names = None

        self.start = ""


        self.prob = problem
        self.current_model = None
        self.argv = None

        if  bool(argv):
            self.next_step_name = argv.pop(0)
            self.argv = argv
            self.next_step_run = getattr(runner,"run_" + self.next_step_name)


    """
    Attention time budget si not implemented for Feature Selection !!!
    """
    def run(self):
        while self.evals > 0:
            self.iterate()
            self.save()
            if bool(self.argv):
                self.next_step_run(self.prob, self.argv[0], self.argv[1], self.argv[2:])

            self.evals = self.evals - 1



    def save(self):
        np.save(self.prob + "/datasets/custom/train_X", self.trans_train_X)
        np.save(self.prob + "/datasets/custom/train_y", self.train_y)
        np.save(self.prob + "/datasets/custom/valid_X", self.trans_valid_X)
        np.save(self.prob + "/datasets/custom/valid_y", self.valid_y)


        np.save(self.prob + "/datasets/custom/names", self.new_features_names)

    @abstractmethod
    def iterate(self):
        raise NotImplementedError("Must override a iterate method ")
