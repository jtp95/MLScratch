import numpy as np
import pandas as pd

class TreeNode():
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        
    def __str__(self):
        if self.is_leaf:
            return f"Leaf node with prediction {self.prediction}"
        else:
            return f"Internal node with feature index {self.feature_index} and threshold {self.threshold}; Left: {self.left}; Right: {self.right}"

class DecisionTree():
    def __init__(self, task="classification", criterion="gini", max_depth=None, min_sample=2):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample = min_sample
    
    def _input_processing(self, X, y=None):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        if y is not None:
            self.n_samples, self.n_features = X.shape
            y = np.asarray(y)
        
        return X, y
    
    def _best_split(self, X, y):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        imp_metric = self._get_metric()
                
        imp_s = imp_metric(y)
        len_s = len(y)
        
        for feature_index in range(self.n_features):
            thresholds = np.unique(X[:,feature_index])
            for threshold in thresholds:
                left_sample = X[:,feature_index] < threshold
                right_sample = ~left_sample
                
                len_l = sum(left_sample)
                len_r = sum(right_sample)
                
                if len_l == 0 or len_r == 0:
                    continue
                
                imp_l = imp_metric(y[left_sample])
                imp_r = imp_metric(y[right_sample])
                
                gain = imp_s - (len_l / len_s) * imp_l - (len_r / len_s) * imp_r
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _create_node(self, X, y, depth=0):
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_sample:
            return self._create_leaf(y)
        
        best_feature, best_threshold = self._best_split(X, y)
        
        left_sample = X[:, best_feature] < best_threshold
        right_sample = ~left_sample
        
        left = self._create_node(X[left_sample], y[left_sample], depth+1)
        right = self._create_node(X[right_sample], y[right_sample], depth+1)
        
        return TreeNode(
            is_leaf=False, 
            feature_index=best_feature, 
            threshold=best_threshold, 
            left=left, 
            right=right
        )
            
    def _create_leaf(self, y):
        if self.task == "classification":
            unique, count = np.unique(y, return_counts=True)
            pred = unique[np.argmax(count)]
        elif self.task == "regression":
            pred = np.mean(y)
        return TreeNode(is_leaf=True, prediction=pred)
    
    def _get_metric(self):
        if self.task == "classification":
            if self.criterion == "gini":
                return self._gini
            elif self.criterion == "entropy":
                return self._entropy
        elif self.task == "regression":
            if self.criterion == "mse":
                return self._mse
        
    def _gini(self, y):
        probs = np.bincount(y) / len(y)
        gini = 1 - np.sum(probs ** 2)
        return gini
    
    def _entropy(self, y):
        probs = np.bincount(y) / len(y)
        entropy = - np.sum(p * np.log2(p) for p in probs if p>0)
        return entropy
    
    def _mse(self, y):
        mean = np.mean(y)
        mse = np.mean((y-mean) ** 2)
        return mse
    
    def fit(self, X, y):
        X, y = self._input_processing(X, y)
        self.root = self._create_node(X, y)
        return self
    
    
    
    def predict(self, X):
        X, _ = self._input_processing(X)
        y_hat = [self._traverse_tree(x) for x in X]
        return np.array(y_hat)
    
    def _traverse_tree(self, x, node=None):
        if node is None:
            node = self.root
            
        if node.is_leaf:
            return node.prediction
        elif x[node.feature_index] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    
    
    def _accuracy(self, y, yhat):
        return np.mean(y == yhat)
    
    def _mean_squared_error(self, y, yhat):
        return np.mean((y - yhat) ** 2)
    
    def score(self, X, y):
        y_hat = self.predict(X)
        
        if self.task == "classification":
            return self._accuracy(y, y_hat)
        elif self.task == "regression":
            return self._mean_squared_error(y, y_hat)