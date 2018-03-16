import _pickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def tuplize(X_train, y_train, X_test, y_test, validation_split = 0.0):
    dim = X_train.shape[1]
    size = 50000
    s_r = int(size-size*validation_split)
    
    tr_d = (X_train[:s_r],y_train[:s_r])
    tv_d = (X_train[s_r:size],y_train[s_r:size])
    te_d = (X_test,y_test)
    
    print("Training: ", tr_d[0].shape)
    print("Validation: ", tv_d[0].shape)
    training_inputs = [np.reshape(x, (dim, 1)) for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    
    test_inputs = [np.reshape(x, (dim, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    
    validation_inputs = [np.reshape(x, (dim, 1)) for x in tv_d[0]]
    validation_data = zip(validation_inputs, tv_d[1])
    
    return (tr_d, training_data, validation_data, test_data)
def vectorize(i):
    ar = np.zeros((10, 1))
    ar[i] = 1.0
    return ar

def preproc(X_train, X_test, reduce =-1):
    X_train /= 255
    X_test /= 255
    
    x_mean = np.mean(X_train, axis = 0)
    x_std = np.std(X_train, axis = 0)
    X_train -= x_mean
    X_test -= x_mean
    
    if(reduce != -1):
        X_train, X_test = pca(X_train, X_test, reduce)
    else:
        X_train /= x_std
        X_test /= x_std
    return X_train, X_test
def load_CIFAR():
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    return X_train, y_train, X_test, y_test

def pca(X_train, X_test, dim = 200):
    cov = np.dot(X_train.T, X_train) / X_train.shape[0] # get the data covariance matrix
    U,S,V = np.linalg.svd(cov)
    X_train = np.dot(X_train, U[:,:dim]) #Dimensionality Reduction
    X_test = np.dot(X_test, U[:,:dim])
    return X_train, X_test
