import numpy as np

from ann.activation_functions import (
    linear,   derivative_linear,
    sigmoid,  derivative_sigmoid,
    tanh,     derivative_tanh,
    relu,     derivative_relu,
    parametric_relu, derivative_parametric_relu,
    softmax,  derivative_softmax,
)

from ann.optimizer import Optimizer


ACTIVATIONS = {
    'linear'         : linear,
    'sigmoid'        : sigmoid,
    'tanh'           : tanh,
    'relu'           : relu,
    'parametric_relu': parametric_relu,
    'softmax'        : softmax,
}

DERIVATIVE_ACTIVATIONS = {
    'linear'         : derivative_linear,
    'sigmoid'        : derivative_sigmoid,
    'tanh'           : derivative_tanh,
    'relu'           : derivative_relu,
    'parametric_relu': derivative_parametric_relu,
    'softmax'        : derivative_softmax,
}


class Dense:

    def __init__(self, input_size, output_size, activation='linear', optimizer_type='adam'):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {list(ACTIVATIONS)}"
            )

        self.input_size  = input_size
        self.output_size = output_size
        self.activation  = activation
        

        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros((1, output_size))

        self.X  = None
        self.Z  = None
        self.dW = None
        self.db = None

        self.opt = Optimizer(optimizer_type, shape_W=self.W.shape, shape_b=self.b.shape)

    def forward(self, X):
        self.X = X 
        self.Z = X @ self.W + self.b
        A = ACTIVATIONS[self.activation](self.Z)
        return A

    def backward(self, dA):
        f_prime = DERIVATIVE_ACTIVATIONS[self.activation](self.Z)

        if self.activation == 'softmax':
            dZ = np.einsum('ijk,ik->ij', f_prime, dA)
        else:
            dZ = dA * f_prime                        

        self.dW = self.X.T @ dZ                        
        self.db = np.sum(dZ, axis=0, keepdims=True)    
        dX = dZ @ self.W.T                             
        return dX

    def update(self, learning_rate, m, t=1):
        step_W, step_b = self.opt.get_optimization(self.dW, self.db, t)

        self.W -= (learning_rate / m) * step_W
        self.b -= (learning_rate / m) * step_b

    def __repr__(self):
        return (f"Dense(input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"activation='{self.activation}')")
