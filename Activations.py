import numpy as np



class softmax(object):
    @staticmethod
    def activation(z):
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()
    def activation_prime(z):
        J = - z[..., None] * z[:, None, :] # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = z * (1. - z) # diagonal
        return J.sum(axis=1)
    
class sigmoid(object):
    @staticmethod
    def activation(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def activation_prime(z):
        return sigmoid.activation(z)*(1-sigmoid.activation(z))
    
class lrelu(object):
    @staticmethod
    def activation(z):
        return (z<0)*0.01*z + z * (z > 0) 
    @staticmethod
    def activation_prime(z):
        return (z<0)*0.01 + (z > 0)
    
class relu(object):
    @staticmethod
    def activation(z):
        return z * (z > 0) 
    @staticmethod
    def activation_prime(z):
        return (z > 0)
    
class elu(object):
    @staticmethod
    def activation(z):
        return (z<0)*(1.0*np.exp(z) -1) + z * (z > 0) 
    @staticmethod
    def activation_prime(z):
        return (z<0)*1.0*np.exp(z) + (z > 0)
    
class tanh(object):
    @staticmethod
    def activation(z):
        return np.tanh(z)
    @staticmethod
    def activation_prime(z):
        return 1.0 - np.tanh(z)**2
