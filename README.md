# Optimization Techniques for AutoML systems
## Code's components
The code is located in "workspace/base/base". It is composed of the following parts (the files that are not used are in _italic_):
  * A script lauching all simulations in parallel: bench_test_clus.py
  * A parser.py file that parse the commands launched in each process and call the appropriate run function in utils/runner.py file
  * The utils folder: 
    * csvUtils.py which contains useful function to deal with csv files
    * problemUtils.py which contains functions used to map search space representations between them.
    * runner.py contains the run functions of all the steps called by parser.py
    * _visual.py_ 
  * The FS folder:
    * baseFS.py which implements a generic class for feature selection algorithms
    * cash.py which is used in our simulations to add the features to the search space
    * _from_model.py_
    * _generic.py_
  * The MS folder:
    * baseMS.py which implements a generic class for model selection algorithms
    * cash.py which is used in our simulations to add the models and the constraints to the search space
    * The model_utils folder:
      * add_kneighborsClassifier.py adds the kneighborsClassifier model to the search space together with its hyperparameters and the constraints
      * add_kneighborsRegressor.py adds the kneighborsRegressor model to the search space together with its hyperparameters and the constraints
      * add_nnClassifier.py adds the nnClassifier model to the search space together with its hyperparameters and the constraints
      * add_nnRegressor.py adds the nnRegressor model to the search space together with its hyperparameters and the constraints
      * add_randomForestClassifier.py adds the randomForestClassifier model to the search space together with its hyperparameters and the constraints
      * add_randomForestRegressor.py adds the randomForestRegressor model to the search space together with its hyperparameters and the constraints
      * ambs_write.py adds the corresponding input model to the search sapce together with its hyperparameters and the constraints by writting inside the       prob_space.py file of the task
   * The HPT folder:
     * baseHPT.py which implements a generic class for hyperparameter tunning algorithms
     * ambs.py the ambs method that can be applied on the previously defined search space
     * ga.py the ga method that can be applied on the previously defined search space
     * pso.py the pso method that can be applied on the previously defined search space
     * rs.py the rs method that can be applied on the previously defined search space
     * tpe.py the tpe method that can be applied on the previously defined search space
     * (these method can easily adapted to be used for FS of MS only)
                    
  

## Code's working process

## Running experiments
