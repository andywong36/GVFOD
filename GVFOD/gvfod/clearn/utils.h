#include <Python.h>
#include <numpy/arrayobject.h>

// Learn the weight vectors and save them. This is the TD(lambda) algorithm.
int learn(
    const npy_uintp x[], const npy_double y[],
    npy_double tde[], npy_double w[], npy_double z[],
    const npy_uintp nobs,     // the length of x, y, and tderr
    const npy_uintp ntilings, // the number of columns of x
    const npy_uintp nweights, // the length of w, z
    const double gamma, const double lambda, const double alpha);

// Calculate the TD errors using a fixed weight vector
int eval(
    const npy_uintp x[], const npy_double y[], // non-mutable
    npy_double tde[], const npy_double w[],  // mutable
    const npy_uintp nobs,                    // the length of x, y, and tderr
    const npy_uintp ntilings,                // the number of columns of x
    const double gamma);






