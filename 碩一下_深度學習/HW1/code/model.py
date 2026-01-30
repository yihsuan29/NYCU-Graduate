import numpy as np
from layer import * 

class NN:
    def __init__(self, in_size, h1_size, h2_size, out_size,a1, a2, opt , lr):
        self.h1_layer = LinearLayer(in_size, h1_size) 
        if a1 =="sigmoid":
            self.h1_act = Sigmoid()
        else:
            self.h1_act = ReLU()
        
        self.h2_layer  = LinearLayer(h1_size, h2_size) 
        if a2 =="sigmoid":
            self.h2_act = Sigmoid()
        else:
            self.h2_act = ReLU()
            
        self.out_layer = LinearLayer(h2_size, out_size) 
        self.out_act = Sigmoid()
        
        if opt == 1:
            self.opt1 = Adagrad(lr)
            self.opt2 = Adagrad(lr)
            self.opt3 = Adagrad(lr)
        
    def forward(self, X):
        self.x1 = self.h1_layer.forward(X)
        self.a1 = self.h1_act.forward(self.x1)
        self.x2 = self.h2_layer.forward(self.a1)
        self.a2 = self.h2_act.forward(self.x2)
        self.x3 = self.out_layer.forward(self.a2)
        self.output = self.out_act.forward(self.x3)
        return self.output
    
    def backward(self, dY, opt, lr = 0.01):
        dA3 = self.out_act.backward(dY)
        if opt ==1:
            lr = self.opt3.getlr(dA3)
        dX3 = self.out_layer.backward(dA3, lr)
        dA2 = self.h2_act.backward(dX3)
        if opt ==1:
            lr = self.opt2.getlr(dA2)
        dX2 = self.h2_layer.backward(dA2, lr)
        dA1 = self.h1_act.backward(dX2)
        if opt ==1:
            lr = self.opt1.getlr(dA1)
        dX1 = self.h1_layer.backward(dA1, lr)
        return dX1
    
class NN_wo_act:
    def __init__(self, in_size, h1_size, h2_size, out_size):
        self.h1_layer = LinearLayer(in_size, h1_size) 
        self.h2_layer = LinearLayer(h1_size, h2_size) 
        self.out_layer = LinearLayer(h2_size, out_size) 
        self.out_act = Sigmoid() 
        
    def forward(self, X):
        self.x1 = self.h1_layer.forward(X)  
        self.x2 = self.h2_layer.forward(self.x1)  
        self.x3 = self.out_layer.forward(self.x2)  
        self.output = self.out_act.forward(self.x3)
        return self.output
    
    def backward(self, dY, lr = 0.01):
        dA3 = self.out_act.backward(dY)
        dX3 = self.out_layer.backward(dA3, lr)
        dX2 = self.h2_layer.backward(dX3, lr)
        dX1 = self.h1_layer.backward(dX2, lr)
        return dX1

    
def cross_entropy_loss(y, y_pred):
    loss = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    return loss

def d_cross_entropy(y, y_pred):
    dL = (-y/y_pred) + ((1-y)/(1-y_pred))
    return dL  

class Adagrad():
    def __init__(self, lr=0.01):
        self.lr = lr
        self.G = None
    def getlr(self, dY):
        if self.G is None:
            self.G = 1
        self.G += np.mean(dY**2)
        return self.lr / np.sqrt(self.G + 1e-7)

