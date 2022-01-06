import  subprocess
import utils.csvUtils as cU


# Keras equivalent but with lr, activation and nbr of units



print("--------------------Keras----------------------------------------------")

bashCmd0 = ["python", "parser.py", "prob1", "CASH", "ga", "500"]

process0 = subprocess.Popen(bashCmd0, stdout=subprocess.PIPE)

output0, error0 = process0.communicate()


out0 = ""

for i in output0:
    if i == 10:
        i = chr(10) + chr(32)
        out0 = out0 + i
    else:
        out0 = out0 + chr(i)

print(out0)

if bashCmd0[4] != "ambs":

    bashCmd1 = ["mkdir", bashCmd0[2] + "/results/" + bashCmd0[3] + "_" + bashCmd0[4] + "_"+ bashCmd0[5]]

    process1 = subprocess.Popen(bashCmd1, stdout=subprocess.PIPE)

    output1, error1 = process1.communicate()

    bashCmd2 = ["mv", bashCmd0[2] + "/results/results.csv", bashCmd0[2] + "/results/" + bashCmd0[3] + "_" + bashCmd0[4] + "_"+ bashCmd0[5]]

    process2 = subprocess.Popen(bashCmd2, stdout=subprocess.PIPE)

    output2, error2 = process2.communicate()

else:
    bashCmd1 = ["mkdir", bashCmd0[2] + "/results/" + bashCmd0[3] + "_" + bashCmd0[4] + "_"+ bashCmd0[5]]

    process1 = subprocess.Popen(bashCmd1, stdout=subprocess.PIPE)

    output1, error1 = process1.communicate()

    bashCmd2 = ["mv", "results.csv", bashCmd0[2] + "/results/" + bashCmd0[3] + "_" + bashCmd0[4] + "_"+ bashCmd0[5]]

    process2 = subprocess.Popen(bashCmd2, stdout=subprocess.PIPE)

    output2, error2 = process2.communicate()
