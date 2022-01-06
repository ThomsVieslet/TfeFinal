from deephyper.problem import HpProblem

from sklearn_genetic.space import Continuous, Categorical, Integer
import nevergrad as ng
from hyperopt import hp

def from_problem_to_hyperopt(Problem):
    confSpaceProb =  Problem._space
    paramGrid = {}
    for i in confSpaceProb.get_hyperparameters():
        if "feature_" in i.name:
            paramGrid[i.name] = hp.choice(i.name, ["0", "1"])
        if i.name == "model":
            lsModels = []
            for j in i.choices:
                modelParam = {"model_name": j}
                for l in confSpaceProb.get_conditions():
                    if j == l.value:
                        if str(type(l.child)) == "<class 'ConfigSpace.hyperparameters.UniformIntegerHyperparameter'>":
                            modelParam[l.child.name] = hp.quniform(l.child.name, l.child.lower, l.child.upper, 1)
                        elif str(type(l.child)) == "<class 'ConfigSpace.hyperparameters.UniformFloatHyperparameter'>":
                            modelParam[l.child.name] = hp.uniform(l.child.name, l.child.lower, l.child.upper)
                        elif str(type(l.child)) == "<class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>":
                            modelParam[l.child.name] = hp.choice(l.child.name, l.child.choices)
                        elif str(type(l.child)) == "<class 'ConfigSpace.hyperparameters.OrdinalHyperparameter'>":
                            modelParam[l.child.name] = hp.quniform(l.child.name, l.child.lower, l.child.upper, 1)
                        elif str(type(l.child)) == "<class 'ConfigSpace.hyperparameters.Constant'>":
                            modelParam[l.child.name] = hp.uniform(l.child.name, l.child.value, l.child.value)
                lsModels.append(modelParam)

            paramGrid[i.name] = hp.choice(i.name, lsModels)
            

    return paramGrid






def from_problem_to_sklearnSpace(Problem):
    confSpaceProb =  Problem._space

    paramGrid = {}

    for i in confSpaceProb.get_hyperparameters():
        if str(type(i)) == "<class 'ConfigSpace.hyperparameters.UniformIntegerHyperparameter'>":
            paramGrid[i.name] = Integer(i.lower, i.upper)
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.UniformFloatHyperparameter'>":
            paramGrid[i.name] = Continuous(i.lower, i.upper)
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>":
            paramGrid[i.name] = Categorical(list(i.choices))
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.OrdinalHyperparameter'>":
            paramGrid[i.name] = Integer(i.lower, i.upper)
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.Constant'>":
            paramGrid[i.name] = Continuous(i.value, i.value)

    return paramGrid

def constraint(param):
    rf = ["mse", "auto", 1, 2, 1]
    knn = [1, "uniform", "auto", 2]

    if param["model"] == "randomForestRegressor":
        j = 0
        for i in param.values():
            if i != rf[j]:
                return False
            j = j + 1
    else:
        j = 0
        for i in param.values():
            if i != knn[j]:
                return False
            j = j + 1

    return True





def from_problem_to_ngParam(Problem):
    confSpaceProb =  Problem._space

    paramGrid = ng.p.Dict()
    param = {}

    for i in confSpaceProb.get_hyperparameters():
        if str(type(i)) == "<class 'ConfigSpace.hyperparameters.UniformIntegerHyperparameter'>":
            paramGrid[i.name] = ng.p.TransitionChoice(range(i.lower, i.upper, 1) )
            param[i.name] = ng.p.TransitionChoice(range(i.lower, i.upper, 1) )
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.UniformFloatHyperparameter'>":
            paramGrid[i.name] = Array(shape=(1, )).set_bounds(i.lower, i.upper)
            param[i.name] = Array(shape=(1, )).set_bounds(i.lower, i.upper)
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.CategoricalHyperparameter'>":
            paramGrid[i.name] = ng.p.Choice(list(i.choices))
            param[i.name] = ng.p.Choice(list(i.choices))
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.OrdinalHyperparameter'>":
            paramGrid[i.name] = ng.p.TransitionChoice(range(i.lower, i.upper, 1) )
            param[i.name] = ng.p.TransitionChoice(range(i.lower, i.upper, 1) )
        elif str(type(i)) == "<class 'ConfigSpace.hyperparameters.Constant'>":
            paramGrid[i.name] = Array(shape=(1, )).set_bounds(i.value, i.value)
            param[i.name] = Array(shape=(1, )).set_bounds(i.value, i.value)


    return paramGrid, param
