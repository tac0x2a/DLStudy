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

# By Random #
# bias = 1
# W10 = np.array( [[rand() for x in range(0, L1)] for xx in range(0, L0 + bias)] )
# W21 = np.array( [[rand() for x in range(0, L2)] for xx in range(0, L1 + bias)] )
# W32 = np.array( [[rand() for x in range(0, L3)] for xx in range(0, L2 + bias)] )

# By sample Weights
path = "./mnist_data/sample_weight.pkl"
import os.path
if not os.path.exists(path):
    import urllib
    url = "https://github.com/oreilly-japan/deep-learning-from-scratch/raw/master/ch03/sample_weight.pkl"
    urllib.request.urlretrieve(url, path)

with open(path, "rb") as f:
    import pickle
    sample_weight = pickle.load(f)

W10 = np.concatenate((sample_weight['W1'], [sample_weight['b1']]), axis=0)
W21 = np.concatenate((sample_weight['W2'], [sample_weight['b2']]), axis=0)
W32 = np.concatenate((sample_weight['W3'], [sample_weight['b3']]), axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculation
i = int(rand() * TRAIN_SAMPLE)

z0 = np.append(x_train[i], 1)
a1 = np.dot(z0, W10)
z1 = np.append(sigmoid(a1), 1)
a2 = np.dot(z1, W21)
z2 = np.append(sigmoid(a2), 1)
a3 = np.dot(z2, W32)

print("NN Output:", a3) # Result
print("NN Output",  np.argmax(a3)) # Result
print("Expected:",  int(t_train[i]))

# [EOF]
