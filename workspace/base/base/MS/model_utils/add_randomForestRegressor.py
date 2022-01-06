import numpy as np
import ConfigSpace as cs


def run(problem, model):
    criterion = problem.add_hyperparameter(["mse", "mae", "poisson"], "criterion")
    max_features = problem.add_hyperparameter(["auto", "sqrt", "log2"], "max_features")
    max_depth = problem.add_hyperparameter((1, 100), "max_depth")
    min_samples_split = problem.add_hyperparameter((2, 40), "min_samples_split")
    min_samples_leaf = problem.add_hyperparameter((2, 20), "min_samples_leaf")




    
    problem.add_condition(cs.EqualsCondition(criterion, model, "randomForestRegressor"))
    problem.add_condition(cs.EqualsCondition(max_features, model, "randomForestRegressor"))
    problem.add_condition(cs.EqualsCondition(max_depth, model, "randomForestRegressor"))
    problem.add_condition(cs.EqualsCondition(min_samples_split, model, "randomForestRegressor"))
    problem.add_condition(cs.EqualsCondition(min_samples_leaf, model, "randomForestRegressor"))
    

    start = ", criterion=\"mse\", max_depth=1, min_samples_split=2, min_samples_leaf=1 , max_features=\"auto\""

    return start
