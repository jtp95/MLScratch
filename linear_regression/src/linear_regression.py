import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100, verbose=False, fit_intercept=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.w = None
        self.b = None
        
    def init_parameters(self):
        """Initialize parameters (weights and bias).
        """
        self.w = np.random.randn(self.n_features) * 0.01
        self.b = 0.0 if self.fit_intercept else None
        return
        
    def __forward(self, X):
        if self.b is None:
            return X @ self.w
        else:
            return X @ self.w + self.b
    
    def __loss(self, y_hat, y):
        loss = np.mean((y_hat - y) ** 2)
        return loss
    
    def __backward(self, X, y, y_hat):
        gw = (2 / self.n_samples) * (X.T @ (y_hat - y))
        gb = (2 / self.n_samples) * np.sum(y_hat - y)
        
        self.w -= self.learning_rate * gw
        self.b -= self.learning_rate * gb
        return gw, gb
        
    def fit(self, X, y):
        """Trains the estimator with given samples using linear regression.

        Args:
            X (matrix of shape (n_samples, n_features)): training samples
            y (array of shape (n_samples,)): target values
            
        Returns:
            LinearRegression object: trained estimator
        """
        
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.n_samples, self.n_features = X.shape
        
        self.init_parameters()
        
        for iter in range(self.epochs):
            y_hat = self.__forward(X)
            self.__backward(X, y, y_hat)
            
            if self.verbose and (iter % 10 == 0 or iter == self.epochs-1):
                loss = self.__loss(y_hat,y)
                print(f"Iteration: {iter}, Loss: {loss:.4f}")
            
        return self
    
    def predict(self, X):
        """Predicts the values given samples using linear model with trained parameters.

        Args:
            X (matrix of shape (n_samples, n_features)): samples for prediction

        Returns:
            array of shape (n_samples): predicted values
        """
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        y_hat = X @ self.w + self.b
        return y_hat
    
    def score(self, X, y):
        """Computes R squared score to evaluate model performance.

        Args:
            X (matrix of shape (n_samples, n_features)): test samples
            y (array of shape (n_samples,)): true values

        Returns:
            float: R squared score
        """
        
        y_hat = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_hat) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
        