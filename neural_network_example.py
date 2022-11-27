from neuron_example import Neuron
import numpy as np

class NeuralNetwork:

    '''

    Will be a neural network with 3 layers:

        - 2 inputs
        - one hidden layer with neurons (h1, h2)
        - one output neuron (o1)    
    
    Weight and bias will be the same across all neurons

    weight = [0, 1]
    bias = 0

    example from https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
    
    '''

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0


        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        h1_output = self.h1.feedforward(x)
        h2_output = self.h2.feedforward(x)

        

        o1_output = self.o1.feedforward(np.array([h1_output, h2_output]))

        return o1_output

network = NeuralNetwork()
x = np.array([2, 3])

print(network.feedforward(x)) # 0.7216325609518421 is the result