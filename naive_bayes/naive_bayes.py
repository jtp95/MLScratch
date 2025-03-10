import numpy as np
from collections import Counter, defaultdict

class NaiveBayes:
    def __init__(self):
        self.priors = dict()
        
    def __input_processing(self, X, y=None):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        if y is not None:
            self.n_samples, self.n_features = X.shape
            y = np.asarray(y)
        
        return X, y
    
    def __accuracy(self, y, yhat):
        return np.mean(y == yhat)
        
    def fit(self, X, y):
        X, y = self.__input_processing(X, y)
        
        self.classes = np.unique(y)
        
        # Calculate priors (P(y))
        label_counter = Counter(y)
        for label, freq in label_counter.items():
            self.priors[label] = freq / self.n_samples
            
        return self

    def predict(self, X):
        pass
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return self.__accuracy(y, y_hat)
    
class MultinomialNB(NaiveBayes):
    def fit(self, X, y):
        X, y = self.__input_processing(X, y)
        super().fit(X, y)
        
        self.likelihoods = defaultdict(lambda: defaultdict(float))
        v = 0
        
        # Count word per label and frequency of word for each label
        Nxy, Ny = defaultdict(lambda: defaultdict(int)), defaultdict(int)
        for x, label in zip(X, y):
            v += len(np.unique(x))
            Ny[label] += len(x)
            for xi in x:
                Nxy[label][xi] += 1
        
        # calculate likelihood of word given label
        self.unseen_likelihood = dict()
        for label in self.classes:
            self.unseen_likelihood[label] = (1 / (Ny[label] + v)) if (Ny[label] + v) > 0 else 1e-9
            for xi in Nxy[label]:
                self.likelihoods[label][xi] = (Nxy[label][xi] + 1) / (Ny[label] + v)
            
    def predict(self, X):
        X, _ = self.__input_processing(X)
        y_hat = []
        
        for x in X:
            prob = dict()
            
            for label in self.classes:
                val = self.priors[label]
                for xi in x:
                    val *= self.likelihoods[label].get(xi, self.unseen_likelihood[label])
                prob[label] = val
                
            y_hat.append(max(prob, key=prob.get))
        
        return np.asarray(y_hat)

class GaussianNB(NaiveBayes):
    pass

class BernoulliNB(NaiveBayes):
    pass