import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Capturing the data into a Pandas dataframe
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
# Simple display of feature columns 
print(df.columns)

# Feature array
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Defining the target column 
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# Fitting PCA 3 compononent
pca3 = PCA(n_components=3)
pComponents = pca3.fit_transform(x)
pDf = pd.DataFrame(data = pComponents, 
	columns = ['principal component1', 'principal component 2', 'principal component 3'])
finalDff = pd.concat([pDf, df [['target']]], axis = 1)
print(finalDff.head())

# Fitting PCA 2 component
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf.head())

# Plotting PCA 2 component 
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
# Variance Ratio's 
print("Variance ratio 3 ", pca3.explained_variance_ratio_)
print("Variance ratio 2 ", pca.explained_variance_ratio_)
