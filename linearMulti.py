import pandas as pd 
import seaborn as sns 

import statsmodels.formula.api as smf 

tips = sns.load_dataset('tips')

model = smf.ols(formula='tip ~ total_bill + size', data=tips).fit()

print(model.summary())

# Insight to our data types 
print(tips.info())

# See our categories of the sex feature 
print(tips.sex.unique())

# Using multiple categories 
model = smf.ols(formula='tip ~ total_bill + size + sex + smoker + day + time', data=tips).fit()

print(model.summary())

print(tips.day.unique())

from sklearn import linear_model

lr = linear_model.LinearRegression()

predicted = lr.fit(X=tips[['total_bill', 'size']], y=tips['tip'])

print(predicted.coef_)

print(predicted.intercept_)

# Using categorical variables with sklearn 
# converting categorical variables into indicators with pd.dummies
tips_dummy = pd.get_dummies(tips[['total_bill', 'size', 'sex', 'smoker', 'day', 'time']])

print(tips_dummy.head())

lr = linear_model.LinearRegression()

x_tips_dummy_ref = pd.get_dummies(tips[['total_bill', 'size', 'sex', 'smoker', 'day', 'time']], drop_first=True)

predicted = lr.fit(X=x_tips_dummy_ref, y=tips['tip'])

# manually store the labels and append coefficients to them 

import numpy as np 

# fit the model
lr = linear_model.LinearRegression()
predicted = lr.fit(X=x_tips_dummy_ref, y=tips['tip'])

# get the intercept and other coefficients 
values = np.append(predicted.intercept_, predicted.coef_)

# get the names of the values 
names = np.append('intercept', x_tips_dummy_ref.columns)

# put everything into dataframe 
results = pd.DataFrame(values, index = names, columns=['coef'])

print(results)