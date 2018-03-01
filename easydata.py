import pandas as pd
import numpy as np

data = pd.read_csv('test1.csv')

print(data.columns)

features = data.columns


user = []
n_columns = int(input('Enter number of columns: '))
for i in range(0,n_columns):
	column_title = input('Enter exact column title: ')
	user.append(column_title)

column1_input = user[0]
column2_input = user[1]


# Using pandas to get columns
column1 = data[column1_input]
column2 = data[column2_input]

print("column1 ",column1.values)
print("column2 ",column2.values)

column1_mean = column1.mean()
column2_mean = column2.mean()

print("mean: ", column1_mean)
print("mean: ", column2_mean)
