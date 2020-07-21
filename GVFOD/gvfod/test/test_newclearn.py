import gvfod
import numpy as np


def test():
    for _ in range(1000):
        res = gvfod.newclearn(
            phi=np.arange(10, dtype=np.uintp).reshape(5, 2),
            y=np.arange(5, dtype=np.double),
            tde=np.arange(5, dtype=np.double),
            w=np.arange(20, dtype=np.double),
            z=np.arange(20, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1)
