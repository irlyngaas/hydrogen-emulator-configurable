"""
Minimal data loader for the boston-housing UCI benchmark -- the one dataset
this first port is being validated against (same raw data file UQnet ships
in datasets/UCI_datasets/boston-housing/, copied here so this port doesn't
depend on UQnet's repo location persisting).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_boston():
    path = os.path.join(HERE, 'data', 'boston-housing', 'boston_housing.txt')
    raw = np.loadtxt(path)
    X = raw[:, :-1]
    Y = raw[:, -1]
    return X, Y
