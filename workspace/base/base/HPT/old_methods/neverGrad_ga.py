import numpy as np

from utils.problemUtils import from_problem_to_sklearnSpace
from utils.problemUtils import from_problem_to_ngParam
import importlib
from operator import attrgetter
import random

from HPT.baseHPT import HPT_algo

import csv
import nevergrad as ng



class algorithm(HPT_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)

    def loss(self, value):

        evaluation = self.mod_run.run((value[0])[0])

        return -evaluation


    def run(self):
        with open(self.prob + '/results/results_HPT.csv', 'w') as f:
            writer = csv.writer(f)
            names = np.append(self.problemConfig._space.get_hyperparameter_names(), "loss_Tested")
            names = np.append(names, "loss_Best")
            writer.writerow(names)


            paramgrid, param = from_problem_to_ngParam(self.problemConfig)


            instru = ng.p.Instrumentation(paramgrid)



            opt1 = ng.families.DifferentialEvolution()


            opt = opt1.__call__(instru, self.evals)

            for _ in range(opt.budget):
                x = opt.ask()
                loss_tested = self.loss(x.value)
                opt.tell(x, loss_tested)


                point = opt.provide_recommendation()
                loss_best = self.loss(point.value)


                writer = csv.writer(f)

                values = list((x.value[0])[0].values())




                values.append(-loss_tested)
                values.append(-loss_best)

                writer = csv.writer(f)

                writer.writerow(values)
