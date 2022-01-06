import sys
import  subprocess
import utils.runner as runner
import importlib



if __name__ == '__main__':
    sys.argv.pop(0)

    problem = sys.argv.pop(0)

    bashCmd1 = ["python", problem + "/prob_space.py"]

    process1 = subprocess.Popen(bashCmd1, stdout=subprocess.PIPE)

    output1, error1 = process1.communicate()


    load = importlib.import_module(problem+".load", package=None)
    load.load()


    if sys.argv[0] == "FS":
        runner.run_FS(problem, sys.argv[1], sys.argv[2], sys.argv[3:])
    elif sys.argv[0] == "HPT":
        runner.run_HPT(problem, sys.argv[1], sys.argv[2], sys.argv[3:])
    elif sys.argv[0] == "CASH":
        runner.run_CASH(problem, sys.argv[1], sys.argv[2], sys.argv[3:])
