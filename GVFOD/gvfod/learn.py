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
            np.multiply(z, gamma * lambda_, out=z)
            z[phi[t, :]] += 1
            np.add(w, alpha * delta * z, out=w)

    else:
        z.fill(0)  # Not used
        tde.fill(0)

        v = np.zeros_like(y)
        for i, row in enumerate(phi):
            v[i] = np.sum(w[row])

        v_rolled = np.roll(v, -1)
        y_rolled = np.roll(y, -1)

        np.add(y_rolled, gamma * v_rolled - v, out=tde)
        tde[-1] = 0


def learn_ude_naive(phi, y, tde, w, z, gamma, lambda_, alpha, ude, beta):
    z.fill(0)
    tde.fill(0)
    ude.fill(0)

    for n in trange(y.shape[0] - 1):
        delta = (y[n + 1]
                 + gamma * np.sum(w[phi[n + 1, :]])
                 - np.sum(w[phi[n, :]]))
        tde[n] = delta
        np.multiply(z, gamma * lambda_, out=z)
        z[phi[n, :]] += 1
        deltaw = alpha * delta * z
        np.add(w, deltaw, out=w)

        if n > 1:
            td_history = np.zeros(n + 1)
            for i in range(n + 1):
                td_history[i] = y[i + 1] + gamma * np.sum(w[phi[i + 1, :]]) - np.sum(w[phi[i, :]])
            td_ma = np.sum(td_history[-beta:]) / beta
            td_std = np.std(td_history, ddof=1) + np.finfo(np.float).eps
            ude[n] = abs(td_ma / td_std)


def learn_ude(phi, y, tde, w, z, gamma, lambda_, alpha, ude, beta):
    z.fill(0)
    tde.fill(0)
    ude.fill(0)

    # 3 traces are needed:
    # p, h, and H
    # p is the history of states, used to calculate TD error
    # p(0) = gamma * phi(1) - phi(0)
    # p(n) = gamma * phi(n+1) - phi(n) + p(n-1)
    # h is the history of states multiplied by cumulants
    # h(0) = 2 * C(1) * (gamma * phi(1) - phi(0))
    # h(n) = 2 * C(n+1) * (gamma * phi(n+1) - phi(n)) + h(n)
    # H is the sum of outer products of previous states
    # H(0) = np.outer(gamma * phi(1) - phi(0), gamma * phi(1) - phi(0))
    # H(n) = np.outer(gamma * phi(n+1) - phi(n), gamma * phi(n+1) - phi(n)) + H(0)

    # A buffer of the last beta states is also needed
    p = np.zeros_like(w)
    h = np.zeros_like(w)
    H = np.zeros((len(w), len(w)))
    _H_update_vec = np.zeros_like(w)  # Used to build H
    _H_update_mat = np.zeros_like(H)
    ns = 0  # the sum of squares from the sample average
    ndbn = 0
    dbn = 0

    state_diff_buffer = deque(maxlen=beta)
    state_diff_buffer_arr = np.zeros((beta, len(w)))
    c_buffer = deque(maxlen=beta)
    c_buffer_arr = np.zeros(beta)

    for n in trange(y.shape[0] - 1):
        delta = (y[n + 1]
                 + gamma * np.sum(w[phi[n + 1, :]])
                 - np.sum(w[phi[n, :]]))
        tde[n] = delta
        np.multiply(z, gamma * lambda_, out=z)
        z[phi[n, :]] += 1
        deltaw = alpha * delta * z
        np.add(w, deltaw, out=w)

        #  Calculate the average of the last td errors:
        #  Update the previous TD errors with the new weights
        # c_buffer.append(y[n + 1])
        c_buffer_arr[n % beta] = y[n + 1]
        # state_diff_buffer.append(gamma * _expand(phi[n + 1, :], len(w)) - _expand(phi[n, :], len(w)))
        state_diff_buffer_arr[n % beta, :] = gamma * _expand(phi[n + 1, :], len(w)) - _expand(phi[n, :], len(w))

        _sum_tde_n = np.sum(c_buffer_arr)
        # for state_diff in state_diff_buffer:
        #     _sum_tde_n += np.dot(w, state_diff)
        _sum_tde_n += np.sum(np.dot(state_diff_buffer_arr, w))

        avg_tde = _sum_tde_n / beta

        # Calculate the historical variance of the TD errors:
        # deltaprime - the TD error using the updated set of weights
        deltaprime = y[n + 1] + gamma * np.sum(w[phi[n + 1]]) - np.sum(w[phi[n]])

        # Update the mean estimates
        ndbn += deltaprime + np.dot(deltaw, p)

        dbn_last = dbn
        dbn = ndbn / (n + 1)

        # Update the complicated trace item, A
        A = np.inner(deltaw, h) + np.dot(np.dot(H, deltaw), (2 * w - deltaw))
        ns += A + deltaprime ** 2 - dbn ** 2 - n * (dbn ** 2 - dbn_last ** 2)

        # update the ude
        if n < 2:
            ude[n] = 0
        else:
            std = np.sqrt(max(ns / n, 0)) + np.finfo(np.float).eps
            ude[n] = abs(avg_tde / std)

        # update p(n)
        p[phi[n + 1]] += gamma
        p[phi[n]] -= 1

        # update h(n)
        h[phi[n + 1]] += 2 * y[n + 1] * gamma
        h[phi[n]] -= 2 * y[n + 1]

        # update H(n)
        _H_update_vec.fill(0)
        _H_update_mat.fill(0)
        _H_update_vec[phi[n + 1]] += gamma
        _H_update_vec[phi[n]] -= 1
        np.outer(_H_update_vec, _H_update_vec, out=_H_update_mat)
        np.add(H, _H_update_mat, out=H)

    return tde, ude


def _expand(phi, state_size):
    state = np.zeros(state_size)
    state[phi] = 1
    return state
