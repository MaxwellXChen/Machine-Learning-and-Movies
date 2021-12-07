import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from matplotlib import pyplot as plt

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
income = df.loc[:,['worlwide_gross_income']].values

# print(df.head())
x = np.array(x)
# print(x[0,:])
income = np.array(income)
income = income.reshape((income.shape[0],))
# print(income.shape)

year = x[:,0]
genre = x[:,1]
avg_vote = x[:,4]

X_train, X_test, y_train, y_test = train_test_split(x, income, test_size=0.2, random_state=42)

# print('Training Features Shape:', X_train.shape)
# print('Training Labels Shape:', y_train.shape)
# print('Testing Features Shape:', X_test.shape)
# print('Testing Labels Shape:', y_test.shape)

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
mse = mean_squared_error(y_test, predictions)

errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

plt.scatter(X_test[:,6], y_test, color = "black", label = "Actual")
plt.scatter(X_test[:,6], predictions, color = "blue", label = "Predicted")
plt.title("Random Forest Regression Model")
plt.xlabel("Budget")
plt.ylabel("Income")
plt.legend(loc='best')
plt.show()

explained_variance_reg = explained_variance_score(y_test, predictions)
print(explained_variance_reg)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]