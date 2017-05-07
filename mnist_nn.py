# Simple NN for MNIST

import numpy as np
from numpy.random import *

# Download MNIST Data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='./mnist_data')

# Load MNIST Data
TRAIN_SAMPLE = 60000
x_train, x_test = mnist.data[:TRAIN_SAMPLE],   mnist.data[TRAIN_SAMPLE:]   # X 28x28
t_train, t_test = mnist.target[:TRAIN_SAMPLE], mnist.target[TRAIN_SAMPLE:] # Text

# Construction of NN
# Input(28x28) x 1st 50 x  2nd 100 x Output 10
L0 = len(x_train[0])  # 28*28
L1 = 50
L2 = 100
L3 = 10

bias = 1
W10 = np.array( [[rand() for x in range(0, L1)] for xx in range(0, L0 + bias)] )
W21 = np.array( [[rand() for x in range(0, L2)] for xx in range(0, L1 + bias)] )
W32 = np.array( [[rand() for x in range(0, L3)] for xx in range(0, L2 + bias)] )

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculation
z0 = np.append(x_train[0], 1)
a1 = np.dot(z0, W10)
z1 = np.append(sigmoid(a1), 1)
a2 = np.dot(z1, W21)
z2 = np.append(sigmoid(a2), 1)
a3 = np.dot(z2, W32)

print(a3) # Result

# [EOF]
