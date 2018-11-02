import numpy as np
from math import log10, floor

def print_mean_std(x, y):
    digits = -int(floor(log10(abs(y))))
    return str(round(x, digits)) + "±" + str(round(y, 1+digits))

class normal(np.ndarray):
    def __new__(cls, sigma):
        return np.ndarray.__new__(cls, (0,))

    def __init__(self, sigma):
        self.sigma = sigma

    def __add__(self, other):
        try:
            return other + np.random.normal(0, self.sigma, other.shape)
        except AttributeError:
            return other + np.random.normal(0, self.sigma)

    def __radd__(self, other):
        return self.__add__(other)
    