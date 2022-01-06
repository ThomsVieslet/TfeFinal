import numpy as np

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


def from_conf_to_pos(config, configSpace):
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

def from_pos_to_config(pos ,configSpace):

    new_conf = dict.fromkeys(configSpace.get_hyperparameter_names())
    k = 0

    for i in configSpace.get_hyperparameters():
        if str(type(i)) == "<class \'ConfigSpace.hyperparameters.CategoricalHyperparameter\'>":
            for j in i.choices:
                if pos[k] == 1:
                    new_conf[i.name] = j
                k = k + 1

        elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformIntegerHyperparameter\'>":
            new_conf[i.name] = pos[k]
            k = k + 1

        elif str(type(i)) == "<class \'ConfigSpace.hyperparameters.UniformFloatHyperparameter\'>":
            new_conf[i.name] = pos[k]
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








class Particle():
    def __init__(self, problem_space):
        self.problem_space = problem_space._space
        self.config = self.problem_space.sample_configuration()

        self.search_space, self.map = transpose_space(self.problem_space)
        self.position = from_conf_to_pos(self.config, self.problem_space)
        self.pbest_position = self.position
        self.pbest_value = float('-inf')
        self.velocity = np.zeros(len(self.position))
        self.fitness = 0

    def move(self):
        new_position = []
        prev = None
        count_choices = 0

        for i in range(len(self.position)):


            if self.map[i] == 0:
                if prev != 0 and prev is not None:

                    index = np.argmax(new_position[i-count_choices:i])
                    new_position[index + (i-count_choices)] = 1
                    for l in range(len(new_position[i-count_choices:i])):
                        if l != index :
                            new_position[l + (i-count_choices)] = 0

                    count_choices = 0
                    tmp = formate_pos(self.position[i] + self.velocity[i], self.search_space[i])
                    new_position.append(tmp)

                else:
                    tmp = formate_pos(self.position[i] + self.velocity[i], self.search_space[i])
                    new_position.append(tmp)
            else:
                if prev == 0 or prev is None or prev == self.map[i]:
                    tmp = formate_pos(self.position[i] + self.velocity[i], self.search_space[i])

                    new_position.append(tmp)
                    count_choices = count_choices + 1
                else:

                    index = np.argmax(new_position[i-count_choices:i])
                    new_position[index + (i-count_choices)] = 1


                    for l in range(len(new_position[i-count_choices:i])):
                        if l != index:
                            new_position[l+ (i-count_choices)] = 0


                    count_choices = 0
                    tmp = formate_pos(self.position[i] + self.velocity[i], self.search_space[i])
                    new_position.append(tmp)
                    count_choices = count_choices + 1
            if i == (len(self.position)-1) and count_choices != 0:
                new_position[np.argmax(new_position[i+1-count_choices::i+1]) + (i+1-count_choices)] = 1

            prev =  self.map[i]


        self.position = new_position








class Space():

    def __init__(self, problem_space, n_particles, mod_run):
        self.mod_run = mod_run
        self.n_particles = n_particles
        self.particles = []
        self.problem_space =  problem_space
        self.search_space, self.map  = transpose_space(self.problem_space._space)

        for i in range(n_particles):
            part = Particle(problem_space)
            self.particles.append(part)

        self.gbest_value = float('-inf')
        gbest_position = np.zeros(len(self.search_space))
        for i in range(len(self.search_space)):
            gbest_position[i] = (self.search_space[i])[0]

        self.gbest_position = gbest_position


    def fitness(self, particle):
        config = from_pos_to_config(particle.position ,self.problem_space._space)

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
            particle.fitness = 0
            return 0
        else:
            fitness = self.mod_run.run(config)
            particle.fitness = fitness
            return fitness

    def set_pbest(self):
        for particle in self.particles:
            fitness_candidate = self.fitness(particle)
            if(particle.pbest_value < fitness_candidate):
                particle.pbest_value = fitness_candidate
                particle.pbest_position = particle.position


    def set_gbest(self):
        for particle in self.particles:
            best_fitness_candidate = self.fitness(particle)
            if(self.gbest_value < best_fitness_candidate):
                self.gbest_value = best_fitness_candidate
                self.gbest_position = particle.position

    def move_particles(self):
        Vmax =  4
        W = 0.5
        c1 = 2
        c2 = 2
        for particle in self.particles:
            k = 0
            for i in particle.position:

                new_velocity = (W*particle.velocity[k]) + (c1*random.random()) * (particle.pbest_position[k] - i) + (random.random()*c2) * (self.gbest_position[k] - i)
                if abs(new_velocity) > Vmax:
                    new_velocity = Vmax*np.sign(new_velocity)
                if (particle.search_space[k])[0] == 0 and (particle.search_space[k])[1] == 1:
                    new_velocity = 1/(1 + math.exp(-new_velocity))


                particle.velocity[k] = new_velocity
                k = k + 1

            particle.move()




class algorithm(HPT_algo):
    def __init__(self, problem, max_evals, argv):
        super().__init__(problem, max_evals, argv)



    def run(self):
        nb_particles = 50
        self.evals = self.evals/nb_particles
        tic = time.perf_counter()
        with open(self.prob + '/results/results.csv', 'w') as f:
            writer = csv.writer(f)
            names = np.append(self.problemConfig._space.get_hyperparameter_names(), "loss")

            writer.writerow(names)

            search_space = Space(self.problemConfig, nb_particles, self.mod_run)


            while self.evals > 0:
                 search_space.set_pbest()
                 search_space.set_gbest()
                 search_space.move_particles()

                 """
                 values = list(from_pos_to_config(search_space.gbest_position, search_space.problem_space._space).values())

                 values.append(search_space.gbest_value)
                 writer.writerow(values)
                 """
                 for i in search_space.particles:
                     values = list(from_pos_to_config(i.position, search_space.problem_space._space).values())

                     values.append(i.fitness)
                     writer.writerow(values)


                 if hasattr(self, 'time'):
                     if time.perf_counter() > tic + self.time:
                         break

                 self.evals = self.evals - 1
