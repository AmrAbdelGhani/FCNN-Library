import numpy as np

class QuadraticCost(object):
    def __init__(self,act_prime):
        self.act_prime = act_prime
    
    def fn(self, a, y):
        return 0.5*np.linalg.norm(a-y)**2

    
    def delta(self,z, a, y):
        return (a-y)*self.act_prime(z)

class NLL(object):

    
    def __init__(self,act_prime):
        self.act_prime = act_prime

    def fn(self,a, y):
        return (-np.sum(y*np.log(a))) 


    def delta(self,z, a, y):
        return ( a - y )
