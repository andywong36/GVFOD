import gvfod
import numpy as np
from tqdm import trange


def test():
    for _ in trange(1000):
        res = gvfod.flearn(
            phi=np.arange(10, dtype=np.uintp).reshape(5, 2),
            y=np.arange(5, dtype=np.double),
            tde=np.arange(5, dtype=np.double),
            w=np.arange(20, dtype=np.double),
            z=np.arange(20, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1)


if __name__ == "__main__":
    test()
