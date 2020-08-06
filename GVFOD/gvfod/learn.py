from collections import deque

import numpy as np
from tqdm import trange


def learn(phi, y, tde, w, z, gamma, lambda_, alpha):
    if alpha:
        z.fill(0)
        tde.fill(0)
        for t in trange(y.shape[0] - 1):
            delta = (y[t + 1]
                     + gamma * np.sum(w[phi[t + 1, :]])
                     - np.sum(w[phi[t, :]]))
            tde[t] = delta
            np.multiply(z, gamma*lambda_, out=z)
            for i in phi[t, :]:
                z[i] += 1
            np.add(w, alpha * delta * z, out=w)

    else:
        z.fill(0) # Not used
        tde.fill(0)

        v = np.zeros_like(y)
        for i, row in enumerate(phi):
            v[i] = np.sum(w[row])

        v_rolled = np.roll(v, -1)
        y_rolled = np.roll(y, -1)

        np.add(y_rolled, gamma * v_rolled - v, out=tde)
        tde[-1] = 0






