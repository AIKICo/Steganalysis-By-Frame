import numpy as np
from pyeeg import hfd, pfd
from math import log10


def katz(data, n):
    L = np.hypot(np.diff(data), 1).sum() # Sum of distances
    d = np.hypot(data - data[0], np.arange(len(data))).max() # furthest distance from first point
    return log10(n) / (log10(d/L) + log10(n))


if __name__ == '__main__':
    x = np.random.randint(1, 500, 25)
    print(pfd(x, np.diff(x+5)))

