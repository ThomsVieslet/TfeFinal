import numpy as np
import struct
from utils.problemUtils import from_problem_to_sklearnSpace
from utils.problemUtils import from_problem_to_ngParam
import importlib
from operator import attrgetter
import random
from HPT.baseHPT import HPT_algo

import csv
import nevergrad as ng
import math
import time
import random
import threading


def transpose_space(configSpace):
    limits = []
    map = []
    k = 1
    for i in configSpace.get_hyperparameters():
        if str(type(i)) == "<class \'ConfigSpace.hyperparameters.CategoricalHyperparameter\'>":
            for j in i.choices:
                limits.append((0, 1))
                map.append(k)
            k = k + 1
        elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformIntegerHyperparameter\'>":
            limits.append((i.lower, i.upper))
            map.append(0)

        elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformFloatHyperparameter\'>":
            limits.append((i.lower, i.upper))
            map.append(0)

    return limits, map


def from_conf_to_gene(config, configSpace):
    limits = []

    for i in config.keys():
        if str(type(config.get_dictionary()[i])) == "<class 'str'>":
            for j in configSpace.get_hyperparameter(i).choices:
                if config.get_dictionary()[i] == j:
                    limits.append(1)
                else:
                    limits.append(0)
        else:
            limits.append(config.get_dictionary()[i])
    return limits

def from_dict_to_gene(config, configSpace):
    limits = []

    for i in config.keys():
        if str(type(config[i])) == "<class 'str'>":
            for j in configSpace.get_hyperparameter(i).choices:
                if config[i] == j:
                    limits.append(1)
                else:
                    limits.append(0)
        else:
            limits.append(config[i])

    return limits

def from_gene_to_config(gene ,configSpace):

    new_conf = dict.fromkeys(configSpace.get_hyperparameter_names())
    k = 0

    for i in configSpace.get_hyperparameters():
        if str(type(i)) == "<class \'ConfigSpace.hyperparameters.CategoricalHyperparameter\'>":
            for j in i.choices:
                if gene[k]:
                    new_conf[i.name] = j
                k = k + 1

        elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformIntegerHyperparameter\'>":
            new_conf[i.name] = int(gene[k])
            k = k + 1

        elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformFloatHyperparameter\'>":
            new_conf[i.name] = float(gene[k])
            k = k + 1
    return new_conf


def formate_pos(new_position, bounds):
    if new_position > bounds[1]:
        return bounds[1]
    elif new_position < bounds[0]:
        return bounds[0]
    else:
        if isinstance(bounds[0], int):
            return round(new_position)
        else:
            return new_position



class Population():
    def __init__(self, problem_space, pop_size, nMate, mod_run):
        self.mod_run = mod_run
        self.problem_space =  problem_space
        self.nMate = nMate
        self.best_sol = None
        self.best_fitness = 0
        self.limits, self.map = transpose_space(problem_space._space)
        self.pop_size = pop_size
        self.pop = np.zeros((self.pop_size, len(self.limits)))


        for i in range(len(self.pop)):
            config = problem_space._space.sample_configuration()
            self.pop[i] = from_conf_to_gene(config, problem_space._space)


    def fitness(self, individual):
        config = from_gene_to_config(individual ,self.problem_space._space)
        feat = []
        for i in config:
            if "feature_" in i:
                feat.append(config[i])

        only_nul =  True
        for j in feat:
            if j != '0':
                only_nul = False
                break
        if only_nul:
            return 0
        else:
            fitness = self.mod_run.run(config)
            return fitness

    def select_mating_pool(self):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        fitness = np.empty(self.pop_size)
        values = np.empty((self.pop_size, len(self.pop[0])+1))
        k = 0
        for i in self.pop:
            fitness[k] = self.fitness(i)
            values[k, 0:len(self.pop[0])] = i
            values[k, len(self.pop[0])] = fitness[k]
            k = k + 1
        parents = np.empty((self.nMate, self.pop.shape[1]))
        for parent_num in range(self.nMate):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.pop[max_fitness_idx, :]
            if parent_num == 0:
                self.best_sol = parents[parent_num, :]
                self.best_fitness = fitness[ max_fitness_idx]
            fitness[max_fitness_idx] = -99999999999
        return parents, values


    def crossover(self, parents):
        offspring_size = self.pop_size - parents.shape[0]
        offspring = np.empty((offspring_size, parents.shape[1]) )
        # The point at which crossover takes place between two parents. Usually it is at the center.
        crossover_point = np.uint8(parents.shape[1]/2)
        bool = random.choice([True, False])
        if self.map[crossover_point-1] == self.map[crossover_point]:
            if bool:
                while self.map[crossover_point-1] == self.map[crossover_point]:
                    crossover_point = crossover_point + 1
            else:
                while self.map[crossover_point-1] == self.map[crossover_point]:
                    crossover_point = crossover_point - 1


        #print(crossover_point)

        for k in range(offspring_size):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover, configSpace):

        prob = 10
        for idx in range(offspring_crossover.shape[0]):
            new_conf = dict.fromkeys(configSpace.get_hyperparameter_names())
            k = 0

            for i in configSpace.get_hyperparameters():
                if str(type(i)) == "<class \'ConfigSpace.hyperparameters.CategoricalHyperparameter\'>":
                    if random.randint(1, prob) == prob:
                        new_conf[i.name] = random.choice(i.choices)

                        for j in i.choices:
                            k = k + 1
                    else:
                        l = 0
                        for j in i.choices:
                            l = l + 1
                            if offspring_crossover[idx, k]:
                                new_conf[i.name] = j
                            k = k + 1


                elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformIntegerHyperparameter\'>":
                    if random.randint(1, prob) == prob:
                        new_conf[i.name] = random.randint(i.lower, i.upper)
                    else:
                        new_conf[i.name] = int(offspring_crossover[idx, k])
                    k = k + 1

                elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformFloatHyperparameter\'>":
                    if random.randint(1, prob) == prob:
                        new_conf[i.name] = random.uniform(i.lower, i.upper)
                    else:
                        new_conf[i.name] = float(offspring_crossover[idx, k])
                    k = k + 1

            offspring_crossover[idx, :] = from_dict_to_gene(new_conf, configSpace)



        return offspring_crossover



class algorithm(HPT_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)


    def run(self):

        pop_size = 20
        nMate= 10
        self.evals = self.evals/pop_size
        tic = time.perf_counter()
        with open(self.prob + '/results/results.csv', 'w') as f:
            writer = csv.writer(f)
            names = np.append(self.problemConfig._space.get_hyperparameter_names(), "loss")

            writer.writerow(names)


            search_space = Population(self.problemConfig, pop_size, nMate, self.mod_run)

            while self.evals > 0:
                parents, val = search_space.select_mating_pool()


                offspring_crossover = search_space.crossover(parents)


                offspring_mutation = search_space.mutation(offspring_crossover, search_space.problem_space._space)

                search_space.pop[0:parents.shape[0], :] = parents
                search_space.pop[parents.shape[0]:, :] = offspring_mutation


                for i in range(len(val)):
                    print(val[i, 0:len(val[0,:])])
                    values = list(from_gene_to_config(val[i, 0:len(val[0,:])-1], search_space.problem_space._space).values())

                    values.append(val[i, len(val[0,:])-1])
                    writer.writerow(values)

                self.evals = self.evals - 1

                if hasattr(self, 'time'):
                    if time.perf_counter() > tic + self.time:
                        break
