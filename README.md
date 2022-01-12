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
      * add_kneighborsClassifier.py
                    
  

## Code's working process

## Running experiments
