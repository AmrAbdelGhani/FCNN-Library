import numpy as np



class SGD(object):
    
    def __init__(self, lr = 0.01, lmbda = 0.01, decay_rate = 0.1):
        self.lr = lr
        self.lmbda = lmbda
        self.decay_rate = decay_rate
    def preset(self, ws, bs):
        pass
    def update(self, batch_size, weights, biases, nabla_w, nabla_b, n):
        ws = [(1 - self.lr * (self.lmbda / n))* w - ( self.lr / batch_size ) * nw
                        for w, nw in zip(weights, nabla_w)]
        bs = [b-( self.lr / batch_size) * nb
                       for b, nb in zip(biases, nabla_b)]
        return ws, bs

    def decay(self):
        self.lr -= self.lr*(self.decay_rate)
    
    
    
    
class adam(object):
    
    def __init__(self, lr = 0.001, lmbda = 0.01, decay_rate = 0.1):
        self.lr = lr
        self.lmbda = lmbda
        self.decay_rate = decay_rate
        
    def preset(self, weights, biases):
        self.acc = [np.zeros(w.shape) for w in weights]
        self.m = [np.zeros(w.shape) for w in weights] 
        self.accb = [np.zeros(b.shape) for b in biases]
        self.mb = [np.zeros(b.shape) for b in biases]
        
        #Beta1 = 0.9, Beta2 = 0.999
    def update(self, batch_size, weights, biases, nabla_w, nabla_b, n):          
        self.m = [0.9*mo + (1-0.9)*w
                        for w, mo in zip(nabla_w, self.m)]
        
        self.acc = [0.999*acc + (1-0.999) * nw * nw
                        for nw, acc in zip(nabla_w, self.acc)]
        
        ws = [(1 - self.lr *( self.lmbda / n) ) * w -  self.lr * mo / ( np.sqrt(acc) + 1e-8)
                        for w, mo, acc in zip(weights, self.m, self.acc)]
        
        self.mb = [0.9*mo + (1-0.9)*b
                        for b, mo in zip(nabla_b, self.mb)]
        
        self.accb = [0.999*acc + (1-0.999) * nb * nb
                        for nb, acc in zip(nabla_b, self.accb)]
        
        bs = [ b -  self.lr * mo / ( np.sqrt(acc) + 1e-8)
                        for b, mo, acc in zip(biases, self.mb, self.accb)]
        #print(np.max(ws))
        return ws, bs

    def decay(self):
        self.lr -= self.lr*(self.decay_rate)
    
        
    
