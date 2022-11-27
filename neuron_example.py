import numpy as np

def sigmoid(x):
    # activation function (logisitic sigmoid function)
    return 1 / (1 + np.exp(-x))

class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights #array 
        self.bias = bias


    def feedforward(self, inputs):
        #weight inputs and add the bias then user activation function
        total = np.dot(self.weights, inputs) + self.bias #using dot product
        return sigmoid(total)

weights = np.array([0, 1]) # w1, w2
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3]) #x1, x2