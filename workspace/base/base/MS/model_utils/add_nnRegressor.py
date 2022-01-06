import numpy as np
import ConfigSpace as cs

def run(problem, model):
    activation = problem.add_hyperparameter(['relu', 'tanh', 'identity', 'logistic'], 'activation')
    nLayers = problem.add_hyperparameter((1, 10), 'nLayers')
    lr = problem.add_hyperparameter(['constant', 'invscaling', 'adaptive'], 'lr')
    optimizer = problem.add_hyperparameter(['adam', 'sgd', 'lbfgs'], 'optimizer')
    batchSize = problem.add_hyperparameter((8, 32), 'batchSize')

    
    problem.add_condition(cs.EqualsCondition(activation, model, "nnRegressor"))
    problem.add_condition(cs.EqualsCondition(nLayers, model, "nnRegressor"))
    problem.add_condition(cs.EqualsCondition(lr, model, "nnRegressor"))
    problem.add_condition(cs.EqualsCondition(optimizer, model, "nnRegressor"))
    problem.add_condition(cs.EqualsCondition(batchSize, model, "nnRegressor"))
    

    start = ", activation=\'relu\', nLayers=3, lr=\'constant\', optimizer=\'Adam\', batchSize=16 "

    return start
