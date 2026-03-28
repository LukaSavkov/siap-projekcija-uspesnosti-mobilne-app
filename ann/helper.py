import numpy as np

def one_hot(y, num_classes):
    m = len(y)
    Y = np.zeros((m, num_classes))
    Y[np.arange(m), y] = 1.0
    return Y

def standardize(X, mu=None, std=None):
    if mu  is None: mu  = X.mean(axis=0)
    if std is None: std = X.std(axis=0) + 1e-8
    return (X - mu) / std, mu, std

def accuracy(y_pred, y_true_int):
    return np.mean(np.argmax(y_pred, axis=1) == y_true_int)