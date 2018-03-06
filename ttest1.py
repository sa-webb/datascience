# /enter/directory 
"""
Submission requirement:

1) Python 3.6.4 |Anaconda, Inc.| [GCC 7.2.0] on linux
2) Pandas, numpy, and scipy.stats
3) For both questions a and b, provide the solution steps, commented code, and
intermediate and final computation results in a MS word file, and upload the
file to UTCLearn.
"""
__author__ = "Austin Webb"
__contact__ = "tsm792@mocs.utc.edu"
__date__ = "2018/02/19"

import pandas as pd 
import numpy as np 

from scipy import stats 

data = pd.read_csv('Heights.csv')

print("rows, columns", data.shape)
print(data.columns)

# Individualizing male and female columns. 
male_data = data['Height of Males: in Inches']
male_mean = male_data.mean()
male_std = male_data.std()

# excluding male nan's 
male_range = list(range(0,22))
male_array = male_data.iloc[male_range].values

female_data = data['Height of Females: in Inches']
female_mean = female_data.mean()
female_std = female_data.std()
female_array = female_data.values 

print("Male mean: ", male_mean)
print("Female mean: ",female_mean)
print("Male std: ", male_std)
print("Female std: ", female_std)
print("")

## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(male_array,female_array)
print("t value " + str(t2))
print("p value " + str(2*p2))

print("Thus ", t2, " times different")
print("")
print("Permutating...")

perm_male = np.random.permutation(male_array)
print("male permutation",perm_male)

perm_female = np.random.permutation(female_array)
print("female perm: ", perm_female)
print("")
tp, pp = stats.ttest_ind(perm_male,perm_female)
print("t value ", + tp)
print("p value ", + (2*pp))
