import numpy as np

'''Linear Layer: y = W^T h + b'''
class LinearLayer:
    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(in_size, out_size)
        self.bias = np.zeros((1,out_size))
        self.input = None
        self.dW = None
        self.db = None
        
    def forward(self, h):
        self.input = h
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, dY, lr = 0.01):
        dX = np.dot(dY, self.weights.T)
        self.dW = np.dot(self.input.T, dY)
        self.db = np.sum(dY, axis = 0, keepdims = True)        
        self.weights -= lr* self.dW
        self.bias -= lr* self.db        
        return dX
    
        
'''Sigmoid:'''
class Sigmoid:       
    def __init__(self):
        self.input = None    
        self.output = None
        
    def forward(self, h):
        self.input = h
        self.output = 1.0/(1.0 + np.exp(-self.input))
        return self.output
    
    def backward(self, dY):
        dX = dY* self.output* (1 - self.output)
        return dX
        
      
'''ReLU'''
class ReLU:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, h):
        self.input = h
        self.output = np.maximum(0, self.input)
        return self.output
    
    def backward(self,dY):
        dX =  dY * np.where(self.output > 0, 1, 0)
        return dX
    
    