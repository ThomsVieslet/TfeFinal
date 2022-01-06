import sys
import  subprocess
import utils.runner as runner
import importlib
import os

probs = next(os.walk(os. getcwd()))[1]

for problem in probs:
    if "prob" in problem:
        print(problem)
        load = importlib.import_module(problem+".load", package=None)
        load.load()
