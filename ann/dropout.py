import numpy as np

class Dropout:

    def __init__(self, p=0.5):
        if not (0 < p <= 1):
            raise ValueError(f"Keep probability p must be in (0, 1], got {p}")

        self.p    = p
        self.mask = None  

    def forward(self, X, mode='train'):
        if mode == 'train':
            self.mask = (np.random.rand(*X.shape) < self.p) / self.p
            return X * self.mask
        else:
            self.mask = None
            return X

    def backward(self, dA):
        return dA * self.mask


    def update(self, learning_rate, m, t=1):
        pass

    def __repr__(self):
        return f"Dropout(p={self.p})"
