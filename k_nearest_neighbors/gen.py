import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

def generate_classification_data(n_samples=200, n_features=5, n_classes=2, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['label'] = y
    df.to_csv('classification_data.csv', index=False)
    print("Classification dataset saved as classification_data.csv")

def generate_regression_data(n_samples=200, n_features=5, noise=0.1, random_state=42):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    df.to_csv('regression_data.csv', index=False)
    print("Regression dataset saved as regression_data.csv")

if __name__ == "__main__":
    generate_classification_data()
    generate_regression_data()