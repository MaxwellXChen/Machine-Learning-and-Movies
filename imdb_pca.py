import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

logisticRegr = LogisticRegression(solver = 'lbfgs')

df = pd.read_csv('https://raw.githubusercontent.com/MaxwellXChen/Machine-Learning-and-Movies/main/movieData.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999999, inplace=True)
arr = df.values

#need to convert to all numerical data!
X = arr[:,[2,3,4,5,6,8,9,10,11,13,14]]
avg_vote = arr[:,7]
avg_metascore = arr[:,12]

features = ["genre","duration","country","language","votes","budget","usa_gross_income","worlwide_gross_income","reviews_from_users","reviews_from_critics"]
x = df.loc[:, features].values
avg_vote = df.loc[:,['avg_vote']].values
metascore = df.loc[:,['metascore']].values

year = x[:,0]
genre = x[:,1]

X_train, X_test, y_train, y_test = train_test_split(x, metascore, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components = 2)
pca.fit(X_train)
X_train = pca.transform(X_train)
print(X_train.shape)
X_test = pca.transform(X_test)

# Scatterplot
plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0],X_train[:,1], c = y_train,cmap = 'rainbow')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

explained_variance = pca.explained_variance_ratio_

cum_sum_eigenvalues = np.cumsum(explained_variance)

# expected variance chart
plt.bar(range(0,len(explained_variance)), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()