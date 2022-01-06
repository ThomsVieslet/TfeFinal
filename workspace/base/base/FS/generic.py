from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import GenericUnivariateSelect
from FS.baseFS import  FS_algo
import numpy as np
import joblib

class algorithm(FS_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)



    def iterate(self):

        trans = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=50)
        self.trans_train_X = trans.fit_transform(self.train_X, self.train_y)


        ind_rem = []
        if self.feat_names is not None:
            j = 0
            for i in self.feat_names:
                if trans.get_support()[j]:
                    print(i)
                else:
                    ind_rem.append(j)
                j= j + 1


        self.trans_valid_X = np.delete(self.valid_X, ind_rem, axis=1)
        self.new_features_names = np.delete(self.feat_names, ind_rem)

        print("We started with {0} features but retained only {1} of them!".format(self.train_X.shape[1], self.trans_train_X.shape[1]))
