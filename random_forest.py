import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class RandomForest(object):
    def __init__(self, n_estimators=8, max_depth=3, max_features=0.9):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy') for i in range(n_estimators)]
        
    def _bootstrapping(self, num_training, num_features, random_seed = None):
        """
        TODO: 
        - Randomly select a sample dataset of size num_training with replacement from the original dataset. 
        - Randomly select certain number of features (num_features denotes the total number of features in X, 
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.
        
        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        
        Hint: Consider using np.random.choice.
        """
        np.random.seed(seed = 6)
        row_idx = np.random.choice(num_training, num_training, replace = True)
        col_idx = np.random.choice(num_features, int(self.max_features * num_features), replace = False)

        return row_idx, col_idx
        #raise NotImplementedError
            
    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)
            
    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        
        X: NxD numpy array, where N is number 
           of instances and D is the dimensionality of each 
           instance
        y: Nx1 numpy array, the predicted labels

        Returns: 
            None. Calling this function should train the decision trees held in self.decision_trees
        """
        N, D = X.shape
        self.bootstrapping(N,D)
        for i in range(self.n_estimators):
            row_idx = self.bootstraps_row_indices[i]
            col_idx = self.feature_indices[i]
            x_train = X[row_idx][:,col_idx]
            y_train = y[row_idx]
            self.decision_trees[i].fit(x_train,y_train)
        #raise NotImplementedError   
        return None
     
    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(self.decision_trees[t].predict(np.reshape(X[i][self.feature_indices[t]], (1,-1)))[0])
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    
    def plot_feature_importance(self, data_train):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in 
        at least one decision tree from the tuned random__forest from Q3.2.
        
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        for i in self.decision_trees: 
            plt.figure()
            importances = i.feature_importances_
            idx = np.argsort(importances)[::-1]
            plt.bar(data_train.columns[idx], importances[idx], align = "center")
            plt.show()
        #raise NotImplementedError