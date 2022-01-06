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

def decimal_converter(num):
    while num > 1:
        num /= 10
    return num


def float_bin(number, places = 3):
    # split() seperates whole number and decimal
    # part and stores it in two seperate variables
    whole, dec = str(number).split(".")

    # Convert both whole number and decimal
    # part from string type to integer type
    whole = int(whole)
    dec = int (dec)

    # Convert the whole number part to it's
    # respective binary form and remove the
    # "0b" from it.
    res = bin(whole).lstrip("0b") + "."

    # Iterate the number of times, we want
    # the number of decimal places to be
    for x in range(places):

        # Multiply the decimal value by 2
        # and seperate the whole number part
        # and decimal part
        print((decimal_converter(dec)))
        whole, dec = str((decimal_converter(dec)) * 2).split(".")

        # Convert the decimal part
        # to integer again
        dec = int(dec)

        # Keep adding the integer parts
        # receive to the result variable
        res += whole

    return res



def from_gene_to_bin(gene, map):
    print("New")
    new_gene = []
    k=0
    for i in gene:
        if map[k] == 0 :
            print(i)
            print(float_bin(i))
            for j in float_bin(i):
                new_gene.append(int(j))
        else:
            new_gene.append(i)
        k = k + 1

    return np.array(new_gene)


def from_bin_to_gene(bin):
    st = ""
    for i in bin:
        st = st + str(int(i))
    int_number= int(st, 2)
    float_number= float(int_number)

    return float_number


class Population():
    def __init__(self, problem_space, pop_size, nMate, mod_run):
        self.mod_run = mod_run
        self.problem_space =  problem_space
        self.nMate = nMate
        self.search_space, self.map = transpose_space(problem_space._space)
        self.pop_size = pop_size
        self.pop = np.zeros((self.pop_size, len(self.search_space)))

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
        k = 0
        for i in self.pop:
            fitness[k] = self.fitness(i)
            k = k + 1
        parents = np.empty((self.nMate, self.pop.shape[1]))
        for parent_num in range(self.nMate):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents


    def crossover(self, parents):
        offspring_size = self.pop_size - parents.shape[0]
        offspring = np.empty((offspring_size, parents.shape[1]) )
        # The point at which crossover takes place between two parents. Usually it is at the center.
        crossover_point = np.uint8(offspring_size/2)

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

    def mutation(self, offspring_crossover):
        flip_prob = 100
        # Mutation changes a single gene in each offspring randomly.
        """
        for idx in range(offspring_crossover.shape[0]):
            for j in range(offspring_crossover[idx, :]):
                if offspring_crossover[idx, j] == 0 and random.randin(1, flip_prob) == flip_prob:
                    offspring_crossover[idx, j] = 1
                elif offspring_crossover[idx, j] == 1 and random.randin(1, flip_prob) == flip_prob:
                    offspring_crossover[idx, j] = 0
        """
        return offspring_crossover



class algorithm(HPT_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)


    def run(self):
        pop_size = 4
        nMate= 2
        self.evals = self.evals/pop_size
        tic = time.perf_counter()
        with open(self.prob + '/results/results_HPT.csv', 'w') as f:
            writer = csv.writer(f)
            names = np.append(self.problemConfig._space.get_hyperparameter_names(), "loss_best")

            writer.writerow(names)

            search_space = Population(self.problemConfig, pop_size, nMate, self.mod_run)

            while self.evals > 0:
                parents = search_space.select_mating_pool()

                gene_size = len(from_gene_to_bin(parents[0], search_space.map))
                new_parents = np.zeros((parents.shape[0], gene_size))
                k = 0
                for i in parents:
                    new_parents[k, :] = from_gene_to_bin(i, search_space.map)
                    k = k + 1
                break
                offspring_crossover = search_space.crossover(new_parents)


                children = np.zeros((offspring_crossover.shape[0], parents.shape[1]))
                for i in range(parents.shape[0]):
                    k = 0
                    for j in range(parents.shape[1]):
                        if search_space.map[j] == 0:
                            print(new_parents[i, k:k+32])
                            children[i, j] = from_bin_to_gene(new_parents[i, k:k+32] )
                            print(children[i, j])
                            k = k + 32
                        else:
                            children[i, j] = new_parents[i, k]
                            k = k + 1

                #print(children)


                break
