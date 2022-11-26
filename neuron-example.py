import numpy as np

def sigmoid(x):
    # activation function (logisitic sigmoid function)
    return 1 / (1 + np.exp(-x))