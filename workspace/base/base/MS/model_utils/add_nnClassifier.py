import numpy as np
import ConfigSpace as cs


"""
********************************************************************************
    * adds nnClassifier to the search space
    * adds its corresponding hyperparameters
    * adds the conditions
********************************************************************************
"""
def run(problem, model):
    activation = problem.add_hyperparameter(['relu', 'tanh', 'identity', 'logistic'], 'activation')
    nLayers = problem.add_hyperparameter((1, 10), 'nLayers')
    lr = problem.add_hyperparameter(['constant', 'invscaling', 'adaptive'], 'lr')
    optimizer = problem.add_hyperparameter(['adam', 'sgd', 'lbfgs'], 'optimizer')
    batchSize = problem.add_hyperparameter((8, 32), 'batchSize')


    problem.add_condition(cs.EqualsCondition(activation, model, "nnClassifier"))
    problem.add_condition(cs.EqualsCondition(nLayers, model, "nnClassifier"))
    problem.add_condition(cs.EqualsCondition(lr, model, "nnClassifier"))
    problem.add_condition(cs.EqualsCondition(optimizer, model, "nnClassifier"))
    problem.add_condition(cs.EqualsCondition(batchSize, model, "nnClassifier"))



    start = ", activation=\'relu\', nLayers=3, lr=\'constant\', optimizer=\'adam\', batchSize=16 "

    return start
