import numpy as np

def load(data_home='./mnist_data'):
    # Download MNIST Data
    from sklearn.datasets import fetch_mldata
    return fetch_mldata('MNIST original', data_home)
