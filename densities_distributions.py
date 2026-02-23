import numpy as np


def p(x):
    if x >= 0:
        return np.exp(-x)
    else:
        return 0

def R(x):
    a = 0
    b = 30
    if x > a and x < b:
        return 1/(b - a)
    else:
        return 0;

