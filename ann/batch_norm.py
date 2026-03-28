import numpy as np


class BatchNorm:

    def __init__(self, momentum=0.9, epsilon=1e-6):
        self.momentum = momentum
        self.epsilon  = epsilon

        self.gamma = None
        self.beta  = None

        self.running_mean = None
        self.running_var  = None

        self.m      = None
        self.Z_bar  = None   
        self.Z_hat  = None   
        self.ivar   = None   

        # gradients
        self.dgamma = None
        self.dbeta  = None

    def _init_params(self, d):
        self.gamma        = np.ones(d)
        self.beta         = np.zeros(d)
        self.running_mean = np.zeros(d)
        self.running_var  = np.zeros(d)

    def forward(self, Z, mode='train'):

        if self.gamma is None:
            self._init_params(Z.shape[1])

        if mode == 'train':
            self.m = Z.shape[0]

            mu  = np.mean(Z, axis=0)        
            var = np.var(Z,  axis=0)       

            self.Z_bar = Z - mu          
            self.ivar  = 1.0 / np.sqrt(var + self.epsilon) 
            self.Z_hat = self.Z_bar * self.ivar             

            q = self.gamma * self.Z_hat + self.beta          

            self.running_mean = (self.momentum * self.running_mean
                                 + (1 - self.momentum) * mu)
            self.running_var  = (self.momentum * self.running_var
                                 + (1 - self.momentum) * var)

        elif mode == 'test':
            Z_hat = ((Z - self.running_mean)
                     / np.sqrt(self.running_var + self.epsilon))
            q = self.gamma * Z_hat + self.beta

        else:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

        return q

    def backward(self, dq):
        self.dgamma = np.sum(dq * self.Z_hat, axis=0)  
        self.dbeta  = np.sum(dq,              axis=0) 

        dZ_hat = dq * self.gamma                      

        dvar = np.sum(
            dZ_hat * self.Z_bar * (-0.5) * (self.ivar ** 3),
            axis=0
        )                                              

        dmu = np.sum(dZ_hat * (-self.ivar), axis=0)     

        dZ = (dZ_hat * self.ivar
              + dvar * (2.0 / self.m) * self.Z_bar
              + dmu / self.m)                          

        return dZ

    def update(self, learning_rate, m, t=1):
        self.gamma -= (learning_rate / m) * self.dgamma
        self.beta  -= (learning_rate / m) * self.dbeta


    def __repr__(self):
        return f"BatchNorm(momentum={self.momentum}, epsilon={self.epsilon})"
