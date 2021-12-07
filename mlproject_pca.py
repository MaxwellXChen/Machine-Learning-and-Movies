import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
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

features = ["genre","duration","country","language","avg_vote","votes","budget","metascore","reviews_from_users","reviews_from_critics"]
x = df.loc[:, features].values
avg_vote = df.loc[:,['avg_vote']].values
metascore = df.loc[:,['metascore']].values
income = df.loc[:,['worlwide_gross_income']].values

year = x[:,0]
genre = x[:,1]

X_train, X_test, y_train, y_test = train_test_split(x, income, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape)
print(X_test.shape)

pca = PCA(0.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
#print(X_train)
print(X_train.shape)
X_test = pca.transform(X_test)

# Scatterplot
# plt.figure(figsize=(8,6))
# plt.scatter(X_train[:,0],X_train[:,1], c = y_train,cmap = 'rainbow')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.show()

explained_variance = pca.explained_variance_ratio_

cum_sum_eigenvalues = np.cumsum(explained_variance)
print(explained_variance)

# expected variance chart
# plt.bar(range(0,len(explained_variance)), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
# plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

regr = LinearRegression()
mse = []

X_reduced_train = pca.fit_transform(scale(X_train))
n = len(X_reduced_train)

## 10-fold CV
kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)

mse = []

score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score)

for i in np.arange(1, 20):
    score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)

# plt.plot(np.array(mse), '-v')
# plt.xlabel('Number of principal components in regression')
# plt.ylabel('MSE')
# plt.title('Income')
# plt.xlim(xmin=-1)
#plt.show()

## Report: we cam also see that this is the case by looking at the MSE

X_reduced_test = pca.transform(scale(X_test))[:,:10]

#Train Regression Model on Training data
regr = LinearRegression()
regr.fit(X_reduced_train[:,:10], y_train)

# Prediction with test data
pred = regr.predict(X_reduced_test)
mean_squared_error(y_test, pred)

plt.scatter(X_reduced_test[:,0], y_test, color = "black", label = "Actual")
plt.scatter(X_reduced_test[:,0], pred, color = "blue", label = "Predicted")
plt.title("Principal Component Regression Model")
plt.xlabel("First Principal Component")
plt.ylabel("Income")
plt.legend(loc='best')
plt.show()

explained_variance_reg = explained_variance_score(y_test, pred)
print(mean_squared_error(y_test, pred))
print(explained_variance_reg)