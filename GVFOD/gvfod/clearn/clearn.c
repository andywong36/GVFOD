#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "utils.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

PyDoc_STRVAR(clearn_doc,
             "Built in, cross-platform clearn method.\n"
             "\n"
             "All the input arrays need to be C Arrays in Numpy. \n"
             "\n"
             "kwargs:\n"
             "   phi: np.ndarray of shape (nobs, ntilings), of dtype np.uintp \n"
             "   y: np.ndarray of shape (nobs, ), dtype np.double \n"
             "   tde: np.ndarray of shape (nobs, ), dtype np.double. This array will be overwritten. \n"
             "   w: np.ndarray of shape (nweights, ), dtype np.double. This array will be overwritten if alpha > 0. \n"
             "   z: np.ndarray of shape (nweights, ), dtype np.double. This array will be overwritten if alpha > 0. \n"
             "   gamma: float, the discount rate \n"
             "   lambda_: float, the trace decay parameter \n"
             "   alpha: float, the step-size \n"
             "Returns:\n"
             "   0 on success.");

static PyObject *
clearn(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static const char *keywordList[] = {"phi", "y", "tde", "w", "z", "gamma", "lambda_", "alpha", NULL};
    PyArrayObject *phi = NULL, *y = NULL, *tde = NULL, *w = NULL, *z = NULL;
    double gamma, lambda, alpha;

    npy_uintp nobs, ntilings, nweights;

    npy_uintp *cphi = NULL;
    npy_double *cy = NULL, *ctde = NULL, *cw = NULL, *cz = NULL;

    PyObject *ret;

    goto try_;
try_:
    assert(!PyErr_Occurred());
    assert(args || kwargs);
    Py_INCREF(args);
    Py_XINCREF(kwargs);

    /* obj_a = ...; */
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!O!O!O!ddd", keywordList,
            &PyArray_Type, &phi,
            &PyArray_Type, &y,
            &PyArray_Type, &tde,
            &PyArray_Type, &w,
            &PyArray_Type, &z,
            &gamma, &lambda, &alpha))
        return NULL;

    if (!phi || !y || !tde || !w || !z)
    {
        PyErr_SetString(PyExc_ValueError, "Could not parse inputs.");
        goto except;
    }
    /* Only do this if obj_a is a borrowed reference. */
    PyArray_INCREF(phi);
    PyArray_INCREF(y);
    PyArray_INCREF(tde);
    PyArray_INCREF(w);
    PyArray_INCREF(z);

    /* Check datatypes */
    if (PyArray_TYPE(phi) != NPY_UINTP)
    {
        PyErr_SetString(PyExc_ValueError, "Phi not of correct type");
        goto except;
    }
    if ((PyArray_TYPE(y) != NPY_DOUBLE) ||
        (PyArray_TYPE(tde) != NPY_DOUBLE) ||
        (PyArray_TYPE(w) != NPY_DOUBLE) ||
        (PyArray_TYPE(z) != NPY_DOUBLE))
    {
        PyErr_SetString(PyExc_ValueError, "y, tde, w, and/or z not of correct type");
        goto except;
    }
//    if (PyObject_Print(phi, stdout, Py_PRINT_RAW) == -1)
//    {
//        PyErr_SetString(PyExc_IOError, "Could not print phi");
//        goto except;
//    }
    if (!PyArray_ISCARRAY(phi))
    {
        PyErr_SetString(PyExc_ValueError, "phi is not NPY_CARRAY");
        goto except;
    }
    if (!PyArray_ISCARRAY(y))
    {
        PyErr_SetString(PyExc_ValueError, "y is not NPY_CARRAY");
        goto except;
    }
    if (!PyArray_ISCARRAY(tde))
    {
        PyErr_SetString(PyExc_ValueError, "tde is not NPY_CARRAY");
        goto except;
    }
    if (!PyArray_ISCARRAY(w))
    {
        PyErr_SetString(PyExc_ValueError, "w is not NPY_CARRAY");
        goto except;
    }
    if (!PyArray_ISCARRAY(z))
    {
        PyErr_SetString(PyExc_ValueError, "z is not NPY_CARRAY");
        goto except;
    }
    if (PyArray_NDIM(phi) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "phi has the wrong ndim");
        goto except;
    }
    if (PyArray_NDIM(y) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "y has the wrong ndim");
        goto except;
    }
    if (PyArray_NDIM(tde) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "tde has the wrong ndim");
        goto except;
    }
    if (PyArray_NDIM(w) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "w has the wrong ndim");
        goto except;
    }
    if (PyArray_NDIM(z) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "z has the wrong ndim");
        goto except;
    }

    /* Get the data */
    cphi = (npy_uintp *)PyArray_DATA(phi);
    cy = (npy_double *)PyArray_DATA(y);
    ctde = (npy_double *)PyArray_DATA(tde);
    cw = (npy_double *)PyArray_DATA(w);
    cz = (npy_double *)PyArray_DATA(z);
    nobs = PyArray_DIMS(phi)[0];
    ntilings = PyArray_DIMS(phi)[1];
    nweights = PyArray_DIMS(w)[0];

    /* Do the learning */
//    printf("\nThe first 3 elements of tde were %f, %f, %f\n", ctde[0], ctde[1], ctde[2]);
    learn(cphi, cy, ctde, cw, cz, nobs, ntilings, nweights, gamma, lambda, alpha);
//    printf("The first 3 elements of tde are now %f, %f, %f\n", ctde[0], ctde[1], ctde[2]);

    /* Return object creation, ret must be a new reference. */
    ret = PyBool_FromLong(0L);
    if (!ret)
    {
        PyErr_SetString(PyExc_ValueError, "Ooops again.");
        goto except;
    }
    assert(!PyErr_Occurred());
    assert(ret);
    goto finally;
except:
    Py_XDECREF(ret);
    assert(PyErr_Occurred());
    ret = NULL;
finally:
    /* Only do this if obj_a is a borrowed reference. */
    PyArray_XDECREF(phi);
    PyArray_XDECREF(y);
    PyArray_XDECREF(tde);
    PyArray_XDECREF(w);
    PyArray_XDECREF(z);
    Py_DECREF(args);
    Py_XDECREF(kwargs);
    return ret;
};

static struct PyMethodDef methods[] =
    {
        {"clearn", clearn, METH_VARARGS | METH_KEYWORDS,
         clearn_doc},
        {NULL, NULL, 0, NULL}};

static struct PyModuleDef clearnMod =
    {
        PyModuleDef_HEAD_INIT,
        "clearn",                     /* name of module */
        "New cross-platform TD(lambda)", /* module documentation, may be NULL */
        -1,                              /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        methods};

PyMODINIT_FUNC
PyInit_clearn(void)
{
    PyObject *initret = PyModule_Create(&clearnMod);
    import_array();
    return initret;
}