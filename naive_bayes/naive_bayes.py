import numpy as np
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self):
        self.priors = dict()
        
    def _input_processing(self, X, y=None):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        if y is not None:
            self.n_samples, self.n_features = X.shape
            y = np.asarray(y)
        
        return X, y
    
    def __accuracy(self, y, yhat):
        return np.mean(y == yhat)
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return self.__accuracy(y, y_hat)
    
class MultinomialNB(NaiveBayes):
    def fit(self, X, y):
        X, y = self._input_processing(X, y)
        
        self.classes = np.unique(y)
        
        # Calculate priors (P(y))
        label_counter = Counter(y)
        for label, freq in label_counter.items():
            self.priors[label] = np.log(freq / self.n_samples)
        
        # Calculate liklihoods (P(xi|y))
        self.likelihoods = defaultdict(lambda: defaultdict(float))
        v = self.n_features
        
        # Count word per label and frequency of word for each label
        Nxy, Ny = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for x, label in zip(X, y):
            Ny[label] += np.sum(x)
            for feature_idx, count in enumerate(x):
                Nxy[label][feature_idx] += count
        
        # calculate likelihood of word given label
        self.unseen_likelihood = dict()
        for label in self.classes:
            self.unseen_likelihood[label] = np.log(1 / (Ny[label] + v))
            for feature_idx in range(v):
                self.likelihoods[label][feature_idx] = np.log((Nxy[label][feature_idx] + 1) / (Ny[label] + v))
            
    def predict(self, X):
        X, _ = self._input_processing(X)
        y_hat = []
        
        for x in X:
            log_probs = dict()
            
            for label in self.classes:
                log_p = self.priors[label]
                for feature_idx, count in enumerate(x):
                    log_p += count * self.likelihoods[label].get(feature_idx, self.unseen_likelihood[label])
                log_probs[label] = log_p
                
            y_hat.append(max(log_probs, key=log_probs.get))
        
        return np.asarray(y_hat)

class GaussianNB(NaiveBayes):
    def fit(self, X, y):
        X, y = self._input_processing(X, y)
        
        self.classes = np.unique(y)
        
        # Calculate priors (P(y))
        label_counter = Counter(y)
        for label, freq in label_counter.items():
            self.priors[label] = np.log(freq / self.n_samples)
        
        # Calculate means and variances for each feature and label for future liklihood calculations
        self.means = defaultdict(lambda: np.zeros(X.shape[1]))
        self.variances = defaultdict(lambda: np.ones(X.shape[1]))
        
        for label in self.classes:
            X_label = X[y == label]
            self.means[label] = np.mean(X_label, axis=0)
            self.variances[label] = np.var(X_label, axis=0)
    
    def __gaussian_likelihood(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var + 1e-9)) * np.exp(- ((x - mean) ** 2) / (2 * var + 1e-9))
        
    def predict(self, X):
        X, _ = self._input_processing(X)
        y_hat = []
        
        for x in X:
            log_probs = dict()
            
            for label in self.classes:
                log_p = self.priors[label]
                
                likelihood = self.__gaussian_likelihood(x, self.means[label], self.variances[label])
                log_l = np.log(likelihood)
                log_p += np.sum(log_l)
                
                log_probs[label] = log_p
                
            y_hat.append(max(log_probs, key=log_probs.get))
        
        return np.asarray(y_hat)

class BernoulliNB(NaiveBayes):
    pass