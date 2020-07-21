#include <Python.h>
#include <numpy/arrayobject.h>

int learn(
    const npy_uintp x[], const npy_double y[],
    npy_double tde[], npy_double w[], npy_double z[],
    const npy_uintp nobs,     // the length of x, y, and tderr
    const npy_uintp ntilings, // the number of columns of x
    const npy_uintp nweights, // the length of w, z
    const double gamma, const double lambda, const double alpha);
int eval(
    const npy_uintp x[], const npy_double y[], // non-mutable
    npy_double tde[], const npy_double w[],  // mutable
    const npy_uintp nobs,                    // the length of x, y, and tderr
    const npy_uintp ntilings,                // the number of columns of x
    const double gamma);

npy_double sumidx(const npy_double vec[], const npy_uintp idx[], const npy_uintp ntilings)
{
    npy_double sum = 0;
    for (npy_uintp i = 0; i < ntilings; i++)
    {
        sum += vec[idx[i]];
    }
    return sum;
}

npy_double tderr(
    const npy_double yp, const npy_double w[], const npy_uintp x[], const npy_uintp xp[],
    npy_double gamma, npy_uintp ntilings)
{
    npy_double val = 0;
    val += yp;
    if (gamma != 0.0)
        val += (gamma * sumidx(w, xp, ntilings));
    val -= sumidx(w, x, ntilings);

    return val;
}

void scalarVectorMultiply(npy_double v[], npy_uintp nv, npy_double s)
{
    npy_uintp i;
    for (i = 0; i < nv; i++)
        v[i] *= s;
    return;
}

int learn(
    const npy_uintp x[], const npy_double y[],        // non-mutable
    npy_double tde[], npy_double w[], npy_double z[], // mutable
    const npy_uintp nobs,                             // the length of x, y, and tderr
    const npy_uintp ntilings,                         // the number of columns of x
    const npy_uintp nweights,                         // the length of w, z
    const double gamma, const double lambda, const double alpha)
{
    if (alpha == 0.0)
    {
        npy_uintp k;
        for (k = 0; k < nweights; k++)
            z[k] = 0;
        return eval(x, y, tde, w, nobs, ntilings, gamma);
    }
    else
    {
        const double zscale = gamma * lambda;

        npy_uintp j, t;

        for (t = 0; t < nobs - 1; t++)
        {
            npy_double delta;
            const npy_uintp *row = x + ntilings * t;
            npy_uintp k;

            delta = tderr(y[t + 1], w, row, row + ntilings, gamma, ntilings);
            tde[t] = delta;

            scalarVectorMultiply(z, nweights, zscale);

            for (j = 0; j < ntilings; j++)
            {
                z[row[j]] += 1;
            }

            for (k = 0; k < nweights; k++)
                w[k] += z[k] * alpha * delta;
        }
        return 0;
    }
}

int eval(
    const npy_uintp *x, const npy_double *y, // non-mutable
    npy_double tde[],                        // mutable
    const npy_double w[],                    // non-mutable
    const npy_uintp nobs,                    // the length of x, y, and tderr
    const npy_uintp ntilings,                // the number of columns of x
    const double gamma)
{
    npy_uintp t;

    npy_double *state_values = (npy_double *)malloc(NPY_SIZEOF_DOUBLE * nobs);

    if (state_values)
    {

        // #pragma loop(hint_parallel(0))
        // #pragma loop(ivdep)
        for (t = 0; t < nobs; t++)
        {
            state_values[t] = sumidx(w, x + t * ntilings, ntilings);
        }

        // #pragma loop(hint_parallel(0))
        // #pragma loop(ivdep)
        for (t = 0; t < nobs - 1; t++)
        {
            tde[t] = y[t + 1] + gamma * state_values[t + 1] - state_values[t];
        }
        free(state_values);
        return 0;
    }
    else
    {
        printf("Error allocating memory");
        return 1;
    }
}