import numpy as np
import ConfigSpace as cs

"""
********************************************************************************
    * adds randomForestClassifier to the search space
    * adds its corresponding hyperparameters
    * adds the conditions
********************************************************************************
"""

def run(problem, model):
    criterion = problem.add_hyperparameter(["gini", "entropy"], "criterion")
    max_features = problem.add_hyperparameter(["auto", "sqrt", "log2"], "max_features")
    max_depth = problem.add_hyperparameter((1, 100), "max_depth")
    min_samples_split = problem.add_hyperparameter((2, 40), "min_samples_split")
    min_samples_leaf = problem.add_hyperparameter((2, 20), "min_samples_leaf")





    problem.add_condition(cs.EqualsCondition(criterion, model, "randomForestClassifier"))
    problem.add_condition(cs.EqualsCondition(max_features, model, "randomForestClassifier"))
    problem.add_condition(cs.EqualsCondition(max_depth, model, "randomForestClassifier"))
    problem.add_condition(cs.EqualsCondition(min_samples_split, model, "randomForestClassifier"))
    problem.add_condition(cs.EqualsCondition(min_samples_leaf, model, "randomForestClassifier"))


    start = ", criterion=\"mse\", max_depth=1, min_samples_split=2, min_samples_leaf=1 , max_features=\"auto\""

    return start
