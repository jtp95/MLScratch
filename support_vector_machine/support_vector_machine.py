import numpy as np
from cvxopt import matrix, solvers

class LinearSVM:
    def __init__(self, learning_rate=0.01, C=1.0, max_iters=1000):
        self.lr = learning_rate
        self.C = C
        self.max_iters = max_iters
        
    def _input_processing(self, X, y=None):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        if y is not None:
            self.n_samples, self.n_features = X.shape
            y = np.asarray(y)
        
        return X, y
    
    def _init_param(self):
        self.w = np.random.randn(self.n_features) * 0.01
        self.b = 0.0
        return
    
    def _hinge_loss(self, x, y):
        return max(0, 1 - y * (self.w @ x + self.b))

    def fit(self, X, y):
        X, y = self._input_processing(X, y)
        
        self._init_param()
        
        for _ in range(self.max_iters):
            for xi, yi in zip(X,y):
                self.w *= (1 - self.lr)
                
                if self._hinge_loss(xi,yi) > 0:
                    self.w += self.lr * self.C * yi * xi
                    self.b += self.lr * self.C * yi
        
        return self

    def predict(self, X):
        X, _ = self._input_processing(X)
        return np.sign(X @ self.w + self.b)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    
class Kernel:
    @staticmethod
    def linear(x, y):
        return x @ y.T

    @staticmethod
    def polynomial(x, y, degree=3, c=1):
        return (x @ y.T + c) ** degree

    @staticmethod
    def rbf(x, y, gamma=1):
        x_norm = (x ** 2).sum(axis=1).reshape(-1, 1)
        y_norm = (y ** 2).sum(axis=1).reshape(1, -1)
        dist = x_norm + y_norm - 2 * x @ y.T
        return np.exp(-gamma * dist)
    
    @staticmethod
    def sigmoid(x, y, alpha=0.01, c=0):
        return np.tanh(alpha * (x @ y.T) + c)

    @staticmethod
    def laplacian(x, y, gamma=1):
        return np.exp(-gamma * np.sum(np.abs(x[:, np.newaxis] - y), axis=2))
    


class KernelSVM:
    def __init__(self, C=1.0, kernel=Kernel.linear, kernel_params={}):
        self.C = C
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None
        
    def _input_processing(self, X, y=None):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        if y is not None:
            self.n_samples, self.n_features = X.shape
            y = np.asarray(y)
        
        return X, y
        
    def _get_kernel_matrix(self, X):
        self.K = self.kernel(X, X, **self.kernel_params)
    
    def _get_alpha(self, y):
        y = y.astype(float).reshape(-1, 1)
        Y = y @ y.T
        
        Q = matrix(Y * self.K)
        p = matrix(-np.ones((self.n_samples, 1)))
        
        G = matrix(np.vstack([
            -np.eye(self.n_samples),  # -alpha <= 0
            np.eye(self.n_samples)   #  alpha <= C
        ]))
        h = matrix(np.vstack([
            np.zeros((self.n_samples, 1)),
            np.full((self.n_samples, 1), self.C)
        ]))
        
        A = matrix(y.T)
        b = matrix(np.zeros(1))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)
        self.alpha = np.ravel(sol['x'])
        
    def _get_sv(self, X, y):
        mask = (self.alpha > 1e-5)
        self.support_vectors = X[mask]
        self.support_labels = y[mask]
        self.support_alpha = self.alpha[mask]

    def _get_bias(self, X, y):
        margin_mask = (self.alpha > 1e-5) & (self.alpha < self.C)
        margin_vectors = X[margin_mask]
        margin_labels = y[margin_mask]

        b_list = []

        K_margin = self.kernel(margin_vectors, self.support_vectors, **self.kernel_params)
        for i in range(len(margin_vectors)):
            b_i = margin_labels[i] - np.sum(self.support_alpha * self.support_labels * K_margin[i])
            b_list.append(b_i)

        self.b = np.mean(b_list) if b_list else 0.0

    def fit(self, X, y):
        X, y = self._input_processing(X, y)
        
        self._get_kernel_matrix(X)
        
        self._get_alpha(y)
        
        self._get_sv(X, y)
        
        self._get_bias(X, y)
    
    def project(self, X):
        X, _ = self._input_processing(X)
        K = self.kernel(X, self.support_vectors, **self.kernel_params)
        return K @ (self.support_alpha * self.support_labels) + self.b
                
    def predict(self, X):
        return np.sign(self.project(X))
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)