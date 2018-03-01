import pandas as pd
import numpy as np

data = pd.read_csv('test1.csv')
print(data.columns)

test_array = data.columns

print(test_array[0])
column1 = test_array['Column1']

print(column1.values)
print(column1.mean)
