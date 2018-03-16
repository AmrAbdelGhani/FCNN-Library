'''
**************************************
Fully Connected Neural Network with added features

By: Amr Abdel Ghani Ali 
Course: Practical Machine Deep Learning
Instructor: Dr. Mohamed Moustafa
The American University in Cairo

Reference Text: Neural Networks and Deep Learning by Michael Nielsen
'''

import json
import random
import time
import numpy as np
import Activations, Cost, Optimizers
from data_utils import vectorize

class Network(object):

    def __init__(self, dimensions, dropout = 0, cost=Cost.NLL, act = Activations.elu, last_act = Activations.softmax):
        self.activation_type = act
        self.activation = act.activation
        self.activation_prime = act.activation_prime
        
        self.last_activation = last_act.activation
        self.last_activation_prime = last_act.activation_prime
        
        self.num_layers = len(dimensions)
        self.dimensions = dimensions
        self.xavier()
        self.cost=cost(self.last_activation_prime)
        self.dropout = dropout
        
        self.evaluation_cost, self.evaluation_accuracy = [], []
        self.training_cost, self.training_accuracy = [], []
        
    def xavier(self):
        self.biases = [np.random.randn(y, 1) for y in self.dimensions[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x/2)
                        for x, y in zip(self.dimensions[:-1], self.dimensions[1:])]

        
    def forwardpass(self, a):
        for b, w in zip(self.biases, self.weights):
            l = a
            a = self.activation(np.dot(w,a)+b)*( 1 -  self.dropout)
        a = self.last_activation(np.dot(self.weights[-1],l)+self.biases[-1])
        return a

    
    
    def fit(self, training_data, epochs, mini_batch_size, optimizer = Optimizers.SGD,
            evaluation_data=None,
            verbose = 0):
        
        self.optimizer = optimizer
        self.optimizer.preset(self.weights,self.biases)

        self.lmbda = self.optimizer.lmbda
        training_data = list(training_data)
        nb_examples = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_valdata = len(evaluation_data)
        
        for j in range(epochs):
            if(j+1 % 150 == 0):
                self.optimizer.decay()
            self.start_time = time.time()
            random.shuffle(training_data)
            
            mini_batches = [
                training_data[x:x+mini_batch_size] for x in range(0, nb_examples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch( mini_batch, nb_examples )
                
            print("Epoch {}/{} ".format(j, epochs))
            self.display_feedback(verbose, training_data, evaluation_data, nb_examples, n_valdata)

            
        return self.evaluation_cost, self.evaluation_accuracy, \
            self.training_cost, self.training_accuracy

    def update_mini_batch(self, mini_batch, n):
        
        dC_db_total = [np.zeros(b.shape) for b in self.biases]
        dC_dw_total = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            dC_db, dC_dw = self.backprop(x, y)
            dC_db_total = [dbt+db for dbt, db in zip(dC_db_total, dC_db)]
            dC_dw_total = [dwt+dw for dwt, dw in zip(dC_dw_total, dC_dw)]
        
        self.weights, self.biases = self.optimizer.update(len(mini_batch),
                                                   self.weights, self.biases, 
                                                   dC_dw_total, dC_db_total,
                                                   n)
        

        
    def backprop(self, x, y):
        dC_db = [np.zeros(b.shape) for b in self.biases]
        dC_dw = [np.zeros(w.shape) for w in self.weights]
        # forwardpass
        activation = x
        activation_list = [x] # Layer Oriented List of Activations
        z_list = [] # Layer Oriented List of Inputs
        c = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_list.append(z)
            activation = self.activation(z)
            if(c==0): #Dropout Implementation: Masking Forward Pass
                drop_matrix = [np.random.binomial(1, (1-self.dropout),
                                                  size = activation.shape )]
                c=1
            drop_array = np.random.binomial(1, (1-self.dropout),
                                            size = activation.shape )
            drop_matrix.append(drop_array)
            activation *= drop_array
            activation_list.append(activation)
        # backward pass
        
        #if(self.activation_type != Activations.softmax)
        #dC_dzf = (self.cost).delta(z_list[-1], self.last_activation(z_list[-1]), y) #Used if Activation is not softmax
        #Get the Error from the last layer
        dC_dzf =  self.last_activation(z_list[-1]) - y   #Algebrically Simplified version for Softmax Activation
        dC_db[-1] = dC_dzf
        dC_dw[-1] = np.dot(dC_dzf, activation_list[-2].transpose())

        #Backpropagate Error
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sp = self.activation_prime(z) * drop_matrix[-l] #Dropout Implementation: Masking Backward Pass
            dC_dzf = np.dot(self.weights[-l+1].transpose(), dC_dzf) * sp
            dC_db[-l] = dC_dzf
            dC_dw[-l] = np.dot(dC_dzf, activation_list[-l-1].transpose())
        return (dC_db, dC_dw)

    
    
    def accuracy(self, dataset, isTrain=False, final = False):
        
        if isTrain:
            results = [(np.argmax(self.forwardpass(img)), np.argmax(labels))
                       for (img, labels) in dataset]
        else:
            results = [(np.argmax(self.forwardpass(img)), label)
                            for (img, label) in dataset]
        if final: #if we want CCR for each class
            self.ea_class = np.zeros((10))
            for (x,y) in results:
                self.ea_class[y] += (x==y)
            print(self.ea_class)
                
        num_correct = sum(int(x == y) for (x, y) in results) 
        
        return num_correct

    def calculate_cost(self, dataset, isTest=False):
        
        cost = 0.0
        for x, y in dataset:
            a = self.forwardpass(x)
            if isTest: y = vectorize(y)
            cost += self.cost.fn(a, y)/len(dataset)\
            + 0.5*(self.lmbda/len(dataset))*sum(np.linalg.norm(ws)**2 for ws in self.weights)
        return cost

    #Save Network Object with the name of the file
    def save(self, filename):
        data = {"dimensions": self.dimensions,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "dropout": self.dropout,
                "evaluation_cost": self.evaluation_cost,
                "training_cost": self.training_cost}
        
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    def display_feedback(self, verbose, training_data, evaluation_data, n, n_data):
        
        if (verbose==1):
            cost = self.calculate_cost(training_data)
            self.training_cost.append(cost)
            accuracy = self.accuracy(training_data, isTrain=True)
            self.training_accuracy.append(accuracy)
            print("{}s - loss: {:.4f} - acc: {:.4f}".format(int(time.time() - self.start_time),
                                                            cost,
                                                            accuracy / n))
        if (verbose == 0):
            cost = self.calculate_cost(evaluation_data, isTest=True)
            self.evaluation_cost.append(cost)
            accuracy = self.accuracy(evaluation_data)
            self.evaluation_accuracy.append(accuracy)
            print("{}s - val_loss: {:.4f} - val_acc: {:.4f}".format(int(time.time() - self.start_time),
                                                                    cost,
                                                                    accuracy / n_data))
        if (verbose==2):
            tr_cost = self.calculate_cost(training_data)
            self.training_cost.append(tr_cost)

            val_cost = self.calculate_cost(evaluation_data, isTest=True)
            self.evaluation_cost.append(val_cost)

            val_accuracy = self.accuracy(evaluation_data)
            self.evaluation_accuracy.append(val_accuracy)

            print("{}s - loss: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}"\
                  .format(int(time.time() - self.start_time),
                          tr_cost,
                          val_cost, val_accuracy/n_data))
        if (verbose==3):
            tr_cost = self.calculate_cost(training_data)
            self.training_cost.append(tr_cost)

            tr_accuracy = self.accuracy(training_data, isTrain=True)
            self.training_accuracy.append(tr_accuracy)

            val_cost = self.calculate_cost(evaluation_data, isTest=True)
            self.evaluation_cost.append(val_cost)

            val_accuracy = self.accuracy(evaluation_data)
            self.evaluation_accuracy.append(val_accuracy)

            print("{}s - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}"\
                  .format(int(time.time() - self.start_time),
                          tr_cost, tr_accuracy / n,
                          val_cost, val_accuracy/n_data))

                
#Load Entire network 
def load(filename):

    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = Cost.NLL
    net = Network(dimensions = data["dimensions"], dropout=data["dropout"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    net.evaluation_cost = data["evaluation_cost"]
    net.training_cost = data["training_cost"]
    return net


