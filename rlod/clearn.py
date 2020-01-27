""" Tests the clearn.dll """

import ctypes
import logging

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)


# def _fast_learn(x, y, td_err, w,1
#                 z,
#                 gamma, lambda_, alpha,
#                 learn=True):
#     """ TD(lambda) learning algorithm implemented with numba"""
#     zscale = gamma * lambda_
#     for t in range(y.shape[0] - 1):
#         delta = (y[t + 1]
#                  + gamma * np.sum(w[x[t + 1, :]])
#                  - np.sum(w[x[t, :]]))
#         td_err[t] = delta
#
#         # self.z = self.z * self.gamma * self.lambda_ # replacement
#         if learn:
#             z *= zscale
#             for i in x[t, :]:
#                 z[i] += 1
#
#             w += z * alpha * delta
#     return td_err, w

def learn(x, y,
          tde, w, z, gamma,
          lambda_, alpha):
    """ wraps clearn.dll (from clearn.c) to learn on x, y

    Be careful with this function - it often fails silently.
    """

    xtype = np.uintp
    ytype = np.float64
    tdetype = np.float64
    wtype = np.float64
    ztype = np.float64
    gammatype = float
    lambdatype = float
    alphatype = float

    # Preprocessing
    if (not x.flags['C_CONTIGUOUS']) or (x.dtype != xtype):
        x = np.ascontiguousarray(x, dtype=xtype)
    nobs = x.shape[0]
    ntilings = x.shape[1]
    assert x.ndim == 2

    if (not y.flags['C_CONTIGUOUS']) or (y.dtype != ytype):
        y = np.ascontiguousarray(y, dtype=ytype)
    assert y.size == nobs

    if (not tde.flags['C_CONTIGUOUS']) or (tde.dtype != tdetype):
        tde = np.ascontiguousarray(tde, dtype=tdetype)
    assert tde.size == nobs

    if (not w.flags['C_CONTIGUOUS']) or (w.dtype != wtype):
        w = np.ascontiguousarray(w, dtype=wtype)
    nweights = w.size

    if (not z.flags['C_CONTIGUOUS']) or (z.dtype != ztype):
        z = np.ascontiguousarray(z, dtype=ztype)
    assert z.size == nweights

    if not isinstance(gamma, gammatype):
        gamma = gammatype(gamma)
    if not isinstance(lambda_, lambdatype):
        lambda_ = lambdatype(lambda_)
    if not isinstance(alpha, alphatype):
        alpha = alphatype(alpha)

    _indpp = ndpointer(dtype=np.uintp, ndim=1, flags="C")
    _doublep = ndpointer(dtype=np.float64, ndim=1, flags="C")
    logger.info("Loading DLL")
    _lib = ctypes.WinDLL(
        "rlod/clearn.dll")
    _flearn = _lib.learn
    _flearn.argtypes = [_indpp, _doublep,
                        _doublep, _doublep, _doublep,
                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.c_double, ctypes.c_double, ctypes.c_double,
                        ]
    _flearn.restype = None

    logger.info("Preparing data")

    # xpp = np.ascontiguousarray(
    #     (x.ctypes.data
    #      + np.arange(x.shape[0]) * x.strides[0]).astype(np.uintp))
    xpp = (x.ctypes.data
           + np.arange(0,
                       x.shape[0] * x.strides[0],
                       x.strides[0], dtype=np.uintp))
    # yp = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # tdep = tde.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # wp = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # zp = z.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    nobs_arg = ctypes.c_size_t(nobs)
    ntilings_arg = ctypes.c_size_t(ntilings)
    nweights_arg = ctypes.c_size_t(nweights)
    gamma_arg = ctypes.c_double(gamma)
    lambda_arg = ctypes.c_double(lambda_)
    alpha_arg = ctypes.c_double(alpha)

    logger.info("Entering clearn.dll")

    _flearn(
        xpp, y,
        tde, w, z,
        nobs_arg, ntilings_arg, nweights_arg,
        gamma_arg, lambda_arg, alpha_arg
    )


def test():
    """ Tests to see if clearn is working as expected """
    from rlod.surprise import TDLambdaGVF

    np.random.seed(1234)
    w = np.random.randint(10, size=5)
    x = np.vstack(list(
        (np.random.choice(5, size=2, replace=False) for _ in range(10))))
    n = np.random.normal(size=10)
    y = np.zeros(10)
    for i in range(10 - 1):
        y[i + 1] = np.sum(w[x[i, :]]) + n[i]

    agent1 = TDLambdaGVF(state_size=5, discount_rate=0.0, learn_rate=0.1, lamda=0.0, beta=3)
    agent2 = TDLambdaGVF(5, 0.0, 0.1, 0.0, 3)

    agent1.learn(x, y)

    agent2.tderrors = np.zeros_like(agent1.tderrors)

    learn(x, y,
          agent2.tderrors, agent2.w, agent2.z, agent2.gamma, agent2.lamda,
          agent2.alpha)

    print("The original TD errors are:")
    print(agent1.tderrors)
    print("The C TD errors are :")
    print(agent2.tderrors)


if __name__ == "__main__":
    test()
