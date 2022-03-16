import numpy as np

def initial_params(scale, dense, rnd = np.random.RandomState(99)):
    return [(scale * rnd.randn(m, n), np.zeros(n)) for m, n in zip(dense[:-1], dense[1:])]

