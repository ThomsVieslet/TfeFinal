import numpy as np


def run(model, file):
    lines = open(file, 'r').readlines()
    initial_len = len(lines)
    count = len(lines) + 1


    if model == "randomForestRegressor":
        lines.insert(count, "criterion = Problem.add_hyperparameter([\"mse\", \"poisson\", \"mae\"], \"criterion\") \n")
        count = count + 1
        lines.insert(count, "max_features = Problem.add_hyperparameter([\"auto\", \"sqrt\", \"log2\"], \"max_features\") \n")
        count = count + 1
        lines.insert(count, "max_depth = Problem.add_hyperparameter((1, 100), \"max_depth\") \n")
        count = count + 1
        lines.insert(count, "min_samples_split = Problem.add_hyperparameter((2, 40), \"min_samples_split\") \n")
        count = count + 1
        lines.insert(count, "min_samples_leaf = Problem.add_hyperparameter((1, 20), \"min_samples_leaf\") \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(criterion, model, \"randomForestRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(max_features, model, \"randomForestRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(max_depth, model, \"randomForestRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(min_samples_split, model, \"randomForestRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(min_samples_leaf, model, \"randomForestRegressor\")) \n")
        count = count + 1


        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        return  ", criterion=\"mse\", max_depth=1, min_samples_split=2, min_samples_leaf=1 , max_features=\"auto\""

    elif model == "kneighborsRegressor":
        lines.insert(count, "n_neighbors = Problem.add_hyperparameter((1, 10), \"n_neighbors\") \n")
        count = count + 1
        lines.insert(count, "weights = Problem.add_hyperparameter([\"uniform\", \"distance\"], \"weights\") \n")
        count = count + 1
        lines.insert(count, "algorithm = Problem.add_hyperparameter([\"auto\", \"ball_tree\", \"kd_tree\", \"brute\"], \"algorithm\") \n")
        count = count + 1
        lines.insert(count, "leaf_size = Problem.add_hyperparameter((2, 40), \"leaf_size\") \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(n_neighbors, model, \"kneighborsRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(weights, model, \"kneighborsRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(algorithm, model, \"kneighborsRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(leaf_size, model, \"kneighborsRegressor\")) \n")

        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        return ", n_neighbors=5, weights=\"uniform\", algorithm=\"auto\", leaf_size=30 "

    elif model == "nnRegressor":
        lines.insert(count, "activation = Problem.add_hyperparameter([\'relu\', \'tanh\', \'identity\', \'logistic\'], \'activation\') \n")
        count = count + 1
        lines.insert(count, "nLayers = Problem.add_hyperparameter((1, 10), \'nLayers\') \n")
        count = count + 1
        lines.insert(count, "lr = Problem.add_hyperparameter([\'constant\', \'invscaling\', \'adaptive\'], \'lr\') \n")
        count = count + 1
        lines.insert(count, "optimizer = Problem.add_hyperparameter([\'adam\', \'sgd\', \'lbfgs\'], \'optimizer\') \n")
        count = count + 1
        lines.insert(count, "batchSize = Problem.add_hyperparameter((8, 32), \'batchSize\') \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(activation, model, \"nnRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(nLayers, model, \"nnRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(lr, model, \"nnRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(optimizer, model, \"nnRegressor\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(batchSize, model, \"nnRegressor\")) \n")

        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        return ", activation=\'relu\', nLayers=5, lr=0.01, optimizer=\'adam\', batchSize=16 "

    elif model == "kneighborsClassifier":
        lines.insert(count, "n_neighbors = Problem.add_hyperparameter((1, 10), \"n_neighbors\") \n")
        count = count + 1
        lines.insert(count, "weights = Problem.add_hyperparameter([\"uniform\", \"distance\"], \"weights\") \n")
        count = count + 1
        lines.insert(count, "algorithm = Problem.add_hyperparameter([\"auto\", \"ball_tree\", \"kd_tree\", \"brute\"], \"algorithm\") \n")
        count = count + 1
        lines.insert(count, "leaf_size = Problem.add_hyperparameter((2, 40), \"leaf_size\") \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(n_neighbors, model, \"kneighborsClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(weights, model, \"kneighborsClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(algorithm, model, \"kneighborsClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(leaf_size, model, \"kneighborsClassifier\")) \n")

        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        return ", n_neighbors=5, weights=\"uniform\", algorithm=\"auto\", leaf_size=30 "

    elif model == "randomForestClassifier":
        lines.insert(count, "criterion = Problem.add_hyperparameter([\"gini\", \"entropy\"], \"criterion\") \n")
        count = count + 1
        lines.insert(count, "max_features = Problem.add_hyperparameter([\"auto\", \"sqrt\", \"log2\"], \"max_features\") \n")
        count = count + 1
        lines.insert(count, "max_depth = Problem.add_hyperparameter((1, 100), \"max_depth\") \n")
        count = count + 1
        lines.insert(count, "min_samples_split = Problem.add_hyperparameter((2, 40), \"min_samples_split\") \n")
        count = count + 1
        lines.insert(count, "min_samples_leaf = Problem.add_hyperparameter((1, 20), \"min_samples_leaf\") \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(criterion, model, \"randomForestClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(max_features, model, \"randomForestClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(max_depth, model, \"randomForestClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(min_samples_split, model, \"randomForestClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(min_samples_leaf, model, \"randomForestClassifier\")) \n")
        count = count + 1


        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        return  ", criterion=\"mse\", max_depth=1, min_samples_split=2, min_samples_leaf=1 , max_features=\"auto\""

    else:
        lines.insert(count, "activation = Problem.add_hyperparameter([\'relu\', \'tanh\', \'identity\', \'logistic\'], \'activation\') \n")
        count = count + 1
        lines.insert(count, "nLayers = Problem.add_hyperparameter((1, 10), \'nLayers\') \n")
        count = count + 1
        lines.insert(count, "lr = Problem.add_hyperparameter([\'constant\', \'invscaling\', \'adaptive\'], \'lr\') \n")
        count = count + 1
        lines.insert(count, "optimizer = Problem.add_hyperparameter([\'adam\', \'sgd\', \'lbfgs\'], \'optimizer\') \n")
        count = count + 1
        lines.insert(count, "batchSize = Problem.add_hyperparameter((8, 32), \'batchSize\') \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(activation, model, \"nnClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(nLayers, model, \"nnClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(lr, model, \"nnClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(optimizer, model, \"nnClassifier\")) \n")
        count = count + 1
        lines.insert(count, "Problem.add_condition(cs.EqualsCondition(batchSize, model, \"nnClassifier\")) \n")

        out = open(file, 'w')
        out.writelines(lines)
        out.close()

        return ", activation=\'relu\',  nLayers=5, lr=\'constant\', optimizer=\'adam\', batchSize=16 "
