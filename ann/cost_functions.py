import numpy as np

def mse(a, y):
    return (1/2) * np.sum((np.linalg.norm(a - y, axis=1))**2)

def derivative_mse(a, y):
    return a - y

def cross_entropy(a, y, epsilon=1e-12):
    a = np.clip(a, epsilon, 1. - epsilon)
    return -np.sum(y * np.log(a))

def derivative_cross_entropy(a, y, epsilon=1e-12):
    a = np.clip(a, epsilon, 1. - epsilon)
    return -y / a

COSTS = {
    'mse'           : mse,
    'cross-entropy' : cross_entropy,
}

DERIVATIVE_COSTS = {
    'mse'           : derivative_mse,
    'cross-entropy' : derivative_cross_entropy,
}


def compute_loss(a, y, cost_type):
    if cost_type not in COSTS:
        raise ValueError(f"Unknown cost '{cost_type}'. Choose from: {list(COSTS)}")
    return COSTS[cost_type](a, y)


def compute_derivative_loss(a, y, cost_type):
    if cost_type not in DERIVATIVE_COSTS:
        raise ValueError(f"Unknown cost '{cost_type}'. Choose from: {list(DERIVATIVE_COSTS)}")
    return DERIVATIVE_COSTS[cost_type](a, y)
