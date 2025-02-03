import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.w = None
        self.b = None
    
    def init_parameters(self):
        """Initialize parameters (weights and bias).
        """
        self.w = np.random.randn(self.n_features) * 0.01
        self.b = 0.0
        return
    
    def __input_adjustment(self, X):
        X = np.asarray(X)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return X
        
    def __forward(self, X):
        z = X @ self.w + self.b
        z = np.clip(z, -500, 500) # prevent overflowing
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat
    
    def __loss(self, y, y_hat):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1-epsilon) # prevent division by 0
        return - np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    
    def __backward(self, X, y, y_hat):
        gw = (X.T @ (y_hat - y)) / (self.n_samples)
        gb = (np.sum(y_hat - y)) / (self.n_samples)
        self.w -= self.learning_rate * gw
        self.b -= self.learning_rate * gb
    
    def fit(self, X, y, tol=1e-4):
        """Trains the estimator with given samples using logistic regression

        Args:
            X (matrix of size (n_samples, n_shapes)): featuers of samples
            y (arrary of size (n_samples)): labels of samples, must be 0 or 1
            tol (float): tolerance for early stopping

        Returns:
            LogisticRegression: trained model object
        """

        X = self.__input_adjustment(X)
        self.n_samples, self.n_features = X.shape
        
        self.init_parameters()
        prev_loss = float('inf')
        
        for iter in range(self.epochs):
            y_hat = self.__forward(X)
            self.__backward(X, y, y_hat)
            
            loss = self.__loss(y, y_hat)
            
            if abs(prev_loss - loss) < tol:
                print(f"Stopping early at iteration {iter}, loss: {loss:.4f}")
                break
            
            prev_loss = loss
            
            if self.verbose and (iter % 10 == 0 or iter == self.epochs-1):
                print(f"Iteration: {iter}, Loss: {loss:.4f}")
                
        return self
    
    
    def predict(self, X):
        """Predicts the class given features of samples with trained logistic regression model

        Args:
            X (matrix of shape (n_samples, n_features)): samples for prediction

        Returns:
            array of shape (n_samples): predicted class, 0 or 1
        """
        
        X = self.__input_adjustment(X)
        y_hat = self.__forward(X)
        return (y_hat >= 0.5).astype(int)
    
    def score(self, X, y):
        """Compute accuracy as a score of the model

        Args:
            X (matrix of size (n_samples, n_shapes)): featuers of samples
            y (arrary of size (n_samples)): classes of samples, must be 0 or 1

        Returns:
            float: score
        """
        y_hat = self.predict(X)
        return np.mean(y_hat == y)