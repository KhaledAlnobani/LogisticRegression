import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(y_pred, y):
    return np.sum(y == y_pred) / len(y)

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mu) / std

    return X_norm

def train_test_split(X, y, random_state=None, test_size=0.2):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * test_size)

    train_indices = indices[split:]
    test_indices = indices[:split]

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    return X_train, y_train, X_test, y_test

np.seterr(divide='ignore')

class LogisticRegression:
    def __init__(self, iteration=1000, learning_rate=0.001, alpha=0):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.error = []
        self.alpha = alpha

    def fit(self, X, y):
        num_records, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        for _ in range(self.iteration):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

            dw = np.dot(X.T, (y_pred - y)) / num_records
            db = np.sum(y_pred - y) / num_records

            # Update weights with L2 regularization term
            self.weights = self.weights  * (1 - self.alpha / num_records) - self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = -(np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
                     + (self.alpha / (2 * len(y))) * np.sum(self.weights ** 2))


            self.error.append(cost / num_records)

    def predict(self, X):
        prediction = sigmoid(np.dot(X, self.weights) + self.bias)
        return [1 if pred >= 0.5 else 0 for pred in prediction]




