import numpy as np
import ConfigSpace as cs

"""
********************************************************************************
    * adds kneighborsClassifier to the search space
    * adds its corresponding hyperparameters
    * adds the conditions
********************************************************************************
"""

def run(problem, model):
    n_neighbors = problem.add_hyperparameter((1, 10), "n_neighbors")
    weights = problem.add_hyperparameter(["uniform", "distance"], "weights")
    algorithm = problem.add_hyperparameter(["auto", "ball_tree", "kd_tree", "brute"], "algorithm")
    leaf_size = problem.add_hyperparameter((2, 40), "leaf_size")


    problem.add_condition(cs.EqualsCondition(n_neighbors, model, "kneighborsClassifier"))
    problem.add_condition(cs.EqualsCondition(weights, model, "kneighborsClassifier"))
    problem.add_condition(cs.EqualsCondition(algorithm, model, "kneighborsClassifier"))
    problem.add_condition(cs.EqualsCondition(leaf_size, model, "kneighborsClassifier"))


    start = ", n_neighbors=5, weights=\"uniform\", algorithm=\"auto\", leaf_size=30 "

    return start
