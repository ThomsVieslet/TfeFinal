import numpy as np

from utils.problemUtils import from_problem_to_sklearnSpace
import importlib
from operator import attrgetter
import random


import csv
from deap import base
from deap import creator
from deap import tools


def selRandom(individuals, k):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]




class algorithm:
    def __init__(self, problem, max_evals, argv):

        prob = importlib.import_module(problem + ".prob_space", package=None)
        self.problemConfig = prob.Problem

        self.evals = int(max_evals)
        self.train_X = np.load(problem + "/datasets/original/train_X.npy", allow_pickle=True)
        self.train_y = np.load(problem + "/datasets/original/train_y.npy", allow_pickle=True)
        self.valid_X = np.load(problem + "/datasets/original/valid_X.npy", allow_pickle=True)
        self.valid_y = np.load(problem + "/datasets/original/valid_y.npy", allow_pickle=True)
        self.feat_names = np.load(problem + "/datasets/original/names.npy", allow_pickle=True)




        self.mod_run = importlib.import_module(problem + ".model_run", package=None)

        self.prob = problem
        self.current_model = None


        self.argv = argv


    def eval_run(self, individual):

        point = individual.config.get_dictionary()


        evaluation = self.mod_run.run(point)

        individual.fitness_valid =  True


        return evaluation



    def run(self):

        toolbox = base.Toolbox()


        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            names = np.append(self.problemConfig._space.get_hyperparameter_names(), "loss")
            writer.writerow(names)




        toolbox.register("mod_run", self.mod_run.run )
        toolbox.register("attr_config", self.problemConfig._space.sample_configuration)
        toolbox.register("individual",  Individual, toolbox.attr_config, toolbox.mod_run)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.eval_run)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)


        pop = toolbox.population(n=20)
        CXPB, MUTPB = 0.9, 0.8

        fitnesses = list(map(toolbox.evaluate, pop))


        for ind, fit in zip(pop, fitnesses):
            ind.fitness = fit



        fits = [ind.fitness for ind in pop]



        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            names = np.append(self.problemConfig._space.get_hyperparameter_names(), "loss")
            writer.writerow(names)

            g = 0

            # Begin the evolution
            while g < 100:

                # A new generation

                g = g + 1
                print("-- Generation %i --" % g)




                # Select the next generation individuals
                offspring = toolbox.select(pop, len(pop),fit_attr="fitness")
                #offspring = pop
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))



                print("Offs:")


                for child1, child2 in zip(offspring[::2], offspring[1::2]):

                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)

                        child1.fitness_valid = False
                        child2.fitness_valid = False


                for mutant in offspring:
                    if random.random() < MUTPB:
                        print(mutant.configs)
                        toolbox.mutate(mutant)
                        print(mutant.configs)
                        mutant.fitness_valid = False

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness_valid]


                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness= fit


                pop[:] = offspring

                # Gather all the fitnesses in one list and print the stats
                fits = [ind.fitness for ind in pop]

                length = len(pop)
                mean = sum(fits) / length
                sum2 = sum(x*x for x in fits)
                std = abs(sum2 / length - mean**2)**0.5

                max_value = max(fits)


                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))
                print("  Avg %s" % mean)
                print("  Std %s" % std)





                iInd = fits.index(max_value)
                ind = pop[iInd]


                loss = max(fits)


                values = list(ind.configs.values())


                values.append(loss)

                writer = csv.writer(f)

                writer.writerow(values)




class Individual(list):
    def __init__(self, config,  mod_run):
        self.config = config(1)
        self.configs = self.config.get_dictionary()
        super(Individual, self).__init__(list(self.configs.values()))

        tmp = mod_run(self.config.get_dictionary())

        self.fitness = tmp
        self.fitness_valid = True
