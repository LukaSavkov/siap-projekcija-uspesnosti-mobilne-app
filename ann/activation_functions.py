import numpy as np

def linear(x):
    return x

def derivative_linear(x):
    return np.ones(x.shape)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative_tanh(x):
    return 1 - (tanh(x))**2

def relu(x):
    return x * (x > 0)

def derivative_relu(x):
    return (x>0)*np.ones(x.shape)

def parametric_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def derivative_parametric_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha) 

def softmax(x):
    z = np.array(x) - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator

def derivative_softmax(x):
    if len(x.shape)==1:
        x = np.array(x).reshape(1,-1)
    else:
        x = np.array(x)
    m, d = x.shape
    a = softmax(x)
    tensor1 = np.einsum('ij,ik->ijk', a, a)
    tensor2 = np.einsum('ij,jk->ijk', a, np.eye(d, d))
    return tensor2 - tensor1