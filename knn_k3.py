import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Read the data into a pandas dataframe
dataset = pd.read_csv('iris.csv')

# Group all 3 classes 
dataset.groupby('class').size()

# sepal_length,sepal_width,petal_length,petal_width,class
feature_columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
X = dataset[feature_columns].values
y = dataset['class'].values

# Using a label encoder to handle strings
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the data using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)

# k = 9
classifier = KNeighborsClassifier(n_neighbors=3)

# fitting the model
classifier.fit(X_train, y_train)

# predict 
y_pred = classifier.predict(X_test)

# Using pandas confusion matrix function
conf_m = ConfusionMatrix(y_test, y_pred)
print(conf_m)

# Reveal the results 
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
