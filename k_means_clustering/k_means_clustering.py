import numpy as np

class KMeansCluster():
    def __init__(self, k=8, max_iter=300, tolerance=1e-4, init="k-means++"):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init = init  
        
    def _input_processing(self, X):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.n_samples, self.n_features = X.shape
        return X
        
    def _distance(self, x, c):
        return np.linalg.norm(x - c)
    
    def _init_centroid(self, X):
        if self.init == "random":
            self.centroids = self._random_init(X)
        elif self.init == "k-means++":
            self.centroids = self._kmeanspp(X)
    
    def _random_init(self, X):
        idx = np.random.choice(len(X), size=self.k, replace=False)
        centroids = X[idx]
        return centroids
    
    def _kmeanspp(self, X):
        centroids = []
        idx = np.random.choice(len(X))
        centroids.append(X[idx])
        
        for j in range(1,self.k):
            dist_sq = [float('inf')] * self.n_samples
            for i, x in enumerate(X):
                for cj in centroids[:j]:
                    dist_sq[i] = min(dist_sq[i], self._distance(x, cj) ** 2)
                    
            prob = np.array(dist_sq) / np.sum(dist_sq)
            idx = np.random.choice(len(X), p=prob)
            centroids.append(X[idx])
        
        return centroids
        
    def _update_centroid(self, X):
        new_centroids = []
        for i in range(self.k):
            mask = self.label == i
            if np.any(mask):
                new_centroids.append(np.mean(X[mask], axis=0))
            else:
                new_centroids.append(self.centroids[i])
        return np.array(new_centroids)
    
    def _assign_cluster(self, X):
        label = []
        for x in X:
            dists = [self._distance(x,c) for c in self.centroids]
            label.append(np.argmin(dists))
        return np.array(label)
    
    def _convergence(self, new):
        for i in range(self.k):
            if self._distance(self.centroids[i], new[i]) > self.tolerance:
                return False
        return True
    
    def fit(self, X):
        X = self._input_processing(X)
        
        self._init_centroid(X)
        self.label = np.array([-1] * self.n_samples)
        
        for _ in range(self.max_iter):
            self.label = self._assign_cluster(X)
            new_centroids = self._update_centroid(X)
            if self._convergence(new_centroids):
                break
            else:
                self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        X = self._input_processing(X)
        label = self._assign_cluster(X)
        return label
            
    def score(self, X):
        X = self._input_processing(X)
        labels = self.predict(X)
        return np.sum([self._distance(x, self.centroids[labels[i]])**2 for i, x in enumerate(X)])