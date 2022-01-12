import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import numpy as np
import csv
import importlib
from HPT.baseHPT import HPT_algo
import time


"""
********************************************************************************
algorithm:
    * inherits from HPT_algo
    * implements the run method which performs a random search
********************************************************************************
"""
class algorithm(HPT_algo):


    def run(self):

        configs = []
        if not hasattr(self, "time"):
            configs = self.problemConfig._space.sample_configuration(self.evals)
            if not isinstance(configs, list):
                configs = [configs]

            param_names = list(configs[0].get_dictionary().keys())
            param_names.append("loss")
            with open(self.prob + '/results/results.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(param_names)


                for i in configs:
                    values = []
                    loss = self.mod_run.run(i.get_dictionary())



                    for j in i.get_dictionary().keys():
                        values.append(i.get_dictionary()[j])


                    values.append(loss)

                    writer.writerow(values)
        else:
            tic = time.perf_counter()
            configs = self.problemConfig._space.sample_configuration()
            if not isinstance(configs, list):
                configs = [configs]

            param_names = list(configs[0].get_dictionary().keys())
            param_names.append("loss")
            with open(self.prob + '/results/results.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(param_names)


                while True:
                    values = []
                    config = self.problemConfig._space.sample_configuration()
                    loss = self.mod_run.run(config.get_dictionary())



                    for j in config.get_dictionary().keys():
                        values.append(config.get_dictionary()[j])


                    values.append(loss)

                    writer.writerow(values)

                    if time.perf_counter() > tic + self.time:
                        break
