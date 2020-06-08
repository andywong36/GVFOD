""" Tests the clearn.dll """

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
    _lib = ctypes.WinDLL(__file__.replace("py", "dll"))
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



