import openml
import pandas as pd
import random




print("Loading...")
openml_list = openml.datasets.list_datasets()
datalist = pd.DataFrame.from_dict(openml_list, orient="index")

new_datalist = datalist[datalist.NumberOfInstancesWithMissingValues == 0.0 ].sort_values(["NumberOfInstances"])
new_datalist = new_datalist[new_datalist.NumberOfInstances > 5000]
new_datalist = new_datalist[15000 > new_datalist.NumberOfInstances ]

i = random.choice(new_datalist["did"].tolist())

j = openml.datasets.get_dataset(i, download_data=True)


print(j)

print(i)
