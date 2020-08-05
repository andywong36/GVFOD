
import ctypes
import logging

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)


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
    x = np.ascontiguousarray(x, dtype=xtype)
    y = np.ascontiguousarray(y, dtype=ytype)
    tde = np.ascontiguousarray(tde, dtype=tdetype)
    w = np.ascontiguousarray(w, dtype=wtype)
    z = np.ascontiguousarray(z, dtype=ztype)
    gamma = gammatype(gamma)
    lambda_ = lambdatype(lambda_)
    alpha = alphatype(alpha)

    nobs = x.shape[0]
    ntilings = x.shape[1]
    nweights = w.size

    assert x.ndim == 2
    assert y.size == nobs
    assert tde.size == nobs
    assert z.size == nweights

    _indpp = ndpointer(dtype=np.uintp, ndim=1, flags="C")
    _doublep = ndpointer(dtype=np.float64, ndim=1, flags="C")
    logger.info("Loading DLL")
    _lib = ctypes.WinDLL(__file__.replace("py", "dll"))
    _flearn = _lib.learn
    _flearn.argtypes = [_indpp, _doublep,
                        _doublep, _doublep, _doublep,
                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.c_double, ctypes.c_double, ctypes.c_double,
                        ]
    _flearn.restype = None

    logger.info("Preparing data")

    xpp = (x.ctypes.data
           + np.arange(0,
                       x.shape[0] * x.strides[0],
                       x.strides[0], dtype=np.uintp))

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
