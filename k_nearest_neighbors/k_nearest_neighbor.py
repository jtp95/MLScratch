import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k, p=2, task="classification"):
        self.k = k
        self.p = p
        self.task = task
        
    def __MinkowskiDistance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=self.p)
    
    def __accuracy(self, y, yhat):
        return np.mean(y == yhat)
    
    def __mean_squared_error(self, y, yhat):
        return np.mean((y - yhat) ** 2)
    
    def fit(self, X, y):
        """Stores the training samples for K-Nearest Neighbors estimator

        Args:
            X (matrix of shape (n_samples, n_features)): training samples
            y (array of shape (n_samples,)): target values

        Returns:
            KNearestNeighbors object: trained estimator
        """
        
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.n_samples, self.n_features = X.shape
        
        y = np.asarray(y)
        
        self.X = X
        self.y = y
        
        return self
    
    def predict(self, X):
        """Predicts the labels/values given samples using trained samples.

        Args:
            X (matrix of shape (n_samples, n_features)): samples for prediction

        Returns:
            array of shape (n_samples): predicted labels/values
        """
        
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        yhat = []

        for x in X:
            distances = []
            for i in range(self.n_samples):
                d = self.__MinkowskiDistance(self.X[i], x)
                distances.append((d, self.y[i]))
            
            distances.sort(key=lambda d: d[0])
            klabels = [label for _, label in distances[:self.k]]
            
            if self.task == "classification":
                yhat.append(Counter(klabels).most_common(1)[0][0])
            elif self.task == "regression":
                yhat.append(np.mean(klabels))
                
        return np.array(yhat)
    
    def score(self, X, y):
        """Computes score metric to evaluate model performance. The score metric depends on the task of the model.

        Args:
            X (matrix of shape (n_samples, n_features)): test samples
            y (array of shape (n_samples,)): true labels/values

        Returns:
            float: score metric (accuracy/MSE)
        """
        
        yhat = self.predict(X)
        
        if self.task == "classification":
            return self.__accuracy(y, yhat)
        elif self.task == "regression":
            return self.__mean_squared_error(y, yhat)