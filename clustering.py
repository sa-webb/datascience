import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

df = pd.read_csv('Height-Weight.csv')

male_height_data = df['Height of Males: in Inches']
male_weight_data = df['Weight of Males: in pounds']

male_height_range = list(range(0,22))
male_weight_range = list(range(0,22))

male_height_array = male_height_data.iloc[male_height_range].values
male_weight_array = male_weight_data.iloc[male_weight_range].values

f1 = df['Height of Females: in Inches'].values
f2 = df['Weight of Females: in pounds'].values
f3 = male_height_array
f4 = male_weight_array

X = np.array(list(zip(f1,f2,f3,f4)))

print(X)
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)

labels = kmeans.predict(X)
print(labels)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], c='red')
ax.scatter(X[:, 2], X[:, 3], c='green')

plt.show()