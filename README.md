# Optimization Techniques for AutoML systems
## Code's components
The code is located in "workspace/base/base". It is composed of the following parts (the files that are not used are in _italic_):
  * A script lauching all simulations in parallel: bench_test_clus.py
  * A parser.py file that parse the commands launched in each process and call the appropriate run function in utils/runner.py file
  * The utils folder: 
    * csvUtils.py which contains useful function to deal with csv files
    * problemUtils.py which contains functions used to map search space representations between them.
    * runner.py contains the run functions of all the steps called by parser.py
    * _visual.py_ implements visuals representations used during system construction
                    
  

## Code's working process

## Running experiments
