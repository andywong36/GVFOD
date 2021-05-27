#include <stdlib.h>
#include <Python.h>
#include <math.h>
#include <float.h>
#include <numpy/arrayobject.h>
#include "utils.h"
#define CLEANUP(ptr) \
    if (ptr)         \
    {                \
        free(ptr);   \
        ptr = NULL;  \
    }
#define SQUARED(val) ((val) * (val))

static npy_double sumidx(const npy_double vec[], const npy_uintp idx[], const npy_uintp ntilings);
static void scalarVectorMultiply(npy_double v[], npy_uintp nv, npy_double s);
static void scalarVectorMultiplyOut(npy_double v[], npy_double vout[], npy_uintp nv, npy_double s);
static npy_double tderr(
    const npy_double yp, const npy_double w[], const npy_uintp x[], const npy_uintp xp[],
    npy_double gamma, npy_uintp ntilings);
static void init_mat_zeros(npy_double *A[], npy_uintp r, npy_uintp c);
static void set_mat_zeros(npy_double *A[], npy_uintp r, npy_uintp c);
static void set_vec_zeros(npy_double a[], npy_uintp l);
static void update_state_buffer(npy_double state_diff_buffer_row[],
                                const npy_double w[], npy_uintp nweights,
                                const npy_uintp x[], const npy_uintp xp[], npy_double gamma, npy_uintp ntilings);

/*
Matrix algebra functions. 
*/
// A has shape (r, c), x has shape (c,), and b has shape (r, )
static npy_double sum_Ax_plus_b(const npy_double *A[], const npy_double x[], const npy_double b[], npy_uintp r, npy_uintp c);
static npy_double sum_vec(const npy_double v[], npy_uintp n);
static npy_double vec_dot_vec(const npy_double v1[], const npy_double v2[], npy_uintp n);
static npy_double vec_dot_mat_dot_vec(npy_double x[], npy_double *A[], npy_double y[], npy_uintp nx, npy_uintp ny);

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
        const double ZSCALE = gamma * lambda;

        npy_uintp j, t;

        for (t = 0; t < nobs - 1; t++)
        {
            npy_double delta;
            const npy_uintp *row = x + ntilings * t;
            npy_uintp k;

            delta = tderr(y[t + 1], w, row, row + ntilings, gamma, ntilings);
            tde[t] = delta;

            scalarVectorMultiply(z, nweights, ZSCALE);

            for (j = 0; j < ntilings; j++)
            {
                z[row[j]] += 1;
            }

            for (k = 0; k < nweights; k++)
            {
                w[k] += z[k] * alpha * delta;
            }
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
        return -1;
    }
}

int learn_eval(
    const npy_uintp x[], const npy_double y[],
    npy_double tde[], npy_double w[], npy_double z[],
    const npy_uintp nobs,
    const npy_uintp ntilings,
    const npy_uintp nweights,
    const double gamma, const double lambda, const double alpha,
    npy_double ude[],    // Online surprise
    const npy_uintp beta // Bandwidth for surprise moving average calculation
)
{
    int OutOfMemoryError = 0;
    npy_double *p = NULL, *h = NULL, **H = NULL;
    npy_double *H_update_vec = NULL;
    npy_double **state_diff_buffer = NULL, *c_buffer;
    npy_double *deltaw;

    double ns = 0, ndbn = 0;
    const double ZSCALE = gamma * lambda;

    p = calloc(nweights, sizeof(npy_double));
    h = calloc(nweights, sizeof(npy_double));
    H = malloc(nweights * sizeof(npy_double *) + nweights * nweights * sizeof(npy_double));

    H_update_vec = calloc(nweights, sizeof(npy_double));
    state_diff_buffer = malloc(beta * sizeof(npy_double *) + beta * nweights * sizeof(npy_double));
    c_buffer = calloc(beta, sizeof(npy_double));

    deltaw = calloc(nweights, sizeof(npy_double));

    if (!p || !h || !H || !H_update_vec || !state_diff_buffer || !c_buffer || !deltaw)
    {
        OutOfMemoryError = 1;
        goto MemoryCleanup;
    }

    init_mat_zeros(H, nweights, nweights);
    init_mat_zeros(state_diff_buffer, beta, nweights);

    for (npy_uintp n = 0; n < nobs - 1; n++)
    {
        // TD(\lambda) Declarations
        npy_double delta;
        npy_uintp const *row, *rowp;

        // Declarations for OGVFOD
        npy_double avg_tde;
        npy_double deltap;
        npy_double dbn, dbn_last;
        npy_double A;

        // TD(\lambda)
        row = x + n * ntilings;
        rowp = x + (n + 1) * ntilings;
        delta = tderr(y[n + 1], w, row, rowp, gamma, ntilings);
        tde[n] = delta;

        scalarVectorMultiply(z, nweights, ZSCALE);

        for (int j = 0; j < ntilings; j++)
        {
            z[row[j]] += 1;
        }

        scalarVectorMultiplyOut(z, deltaw, nweights, alpha * delta);

        for (int k = 0; k < nweights; k++)
        {
            w[k] += deltaw[k];
        }

        /* Calculate the average of the last td errors:
        Update the previous TD errors with the new weights
        */
        c_buffer[n % beta] = y[n + 1];
        update_state_buffer(state_diff_buffer[n % beta], w, nweights, row, rowp, gamma, ntilings);

        avg_tde = sum_Ax_plus_b(state_diff_buffer, w, c_buffer, beta, nweights) / beta;

        deltap = tderr(y[n + 1], w, row, rowp, gamma, ntilings);

        // Update the mean estimates
        ndbn += deltap + vec_dot_vec(deltaw, p, nweights);

        dbn_last = dbn;
        dbn = ndbn / (n + 1);

        // Update the complicated trace item, A
        A = vec_dot_vec(deltaw, h, nweights) + 2 * vec_dot_mat_dot_vec(w, H, deltaw, nweights, nweights) - vec_dot_mat_dot_vec(deltaw, H, deltaw, nweights, nweights);
        ns += A + SQUARED(deltap) - SQUARED(dbn) - n * (SQUARED(dbn) - SQUARED(dbn_last));

        // Update the UDE
        if (n < 2)
        {
            ude[n] = 0;
        }
        else
        {
            npy_double std = sqrt(ns > 0 ? ns / n : 0.) + DBL_EPSILON;
            ude[n] = avg_tde >= 0. ? avg_tde / std : -avg_tde / std;
        }

        // Update p(n)
        for (npy_uintp i = 0; i < ntilings; i++)
        {
            p[rowp[i]] += gamma;
            p[row[i]] -= 1;
        }

        // Update h(n)
        for (npy_uintp i = 0; i < ntilings; i++)
        {
            h[rowp[i]] += 2 * y[n + 1] * gamma;
            h[row[i]] -= 2 * y[n + 1];
        }

        // Update H(n)
        set_vec_zeros(H_update_vec, nweights);
        for (npy_uintp i = 0; i < ntilings; i++)
        {
            H_update_vec[rowp[i]] = gamma;
            H_update_vec[row[i]] -= 1;
        }
        for (npy_uintp i = 0; i < nweights; i++)
        {
            for (npy_uintp j = 0; j < nweights; j++)
            {
                H[i][j] += H_update_vec[i] * H_update_vec[j];
            }
        }
    }

MemoryCleanup:
    // Cleanup memory
    CLEANUP(p);
    CLEANUP(h);
    CLEANUP(H);
    CLEANUP(H_update_vec);
    CLEANUP(state_diff_buffer);
    CLEANUP(c_buffer);
    goto Finally;

Finally:
    if (OutOfMemoryError)
        return -1;
    return 0;
}

static npy_double sumidx(const npy_double vec[], const npy_uintp idx[], const npy_uintp ntilings)
{
    npy_double sum = 0;
    for (npy_uintp i = 0; i < ntilings; i++)
    {
        sum += vec[idx[i]];
    }
    return sum;
}

static void scalarVectorMultiply(npy_double v[], npy_uintp nv, npy_double s)
{
    npy_uintp i;
    for (i = 0; i < nv; i++)
        v[i] *= s;
    return;
}

static void scalarVectorMultiplyOut(npy_double v[], npy_double vout[], npy_uintp nv, npy_double s)
{
    npy_uintp i;
    for (i = 0; i < nv; i++)
        vout[i] = v[i] * s;
    return;
}

static npy_double tderr(
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

static void init_mat_zeros(npy_double *A[], npy_uintp r, npy_uintp c)
{
    npy_double *ptr = (npy_double *)(A + r);
    for (npy_uintp i = 0; i < r; i++)
    {
        A[i] = ptr + c * i;
    }
    set_mat_zeros(A, r, c);
}

static void set_mat_zeros(npy_double *A[], npy_uintp r, npy_uintp c)
{
    for (npy_uintp i = 0; i < r; i++)
    {
        for (npy_uintp j = 0; j < c; j++)
        {
            A[i][j] = 0.;
        }
    }
}

static void set_vec_zeros(npy_double a[], npy_uintp l)
{
    for (npy_uintp i = 0; i < l; i++)
    {
        a[i] = 0.;
    }
}

static void update_state_buffer(npy_double state_diff_buffer_row[],
                                const npy_double w[], npy_uintp nweights,
                                const npy_uintp x[], const npy_uintp xp[], npy_double gamma, npy_uintp ntilings)
{
    set_vec_zeros(state_diff_buffer_row, nweights);
    for (npy_uintp tile = 0; tile < ntilings; tile++)
    {
        state_diff_buffer_row[xp[tile]] += gamma;
        state_diff_buffer_row[x[tile]] -= 1.;
    }
}

static npy_double sum_Ax_plus_b(const npy_double *A[], const npy_double x[], const npy_double b[], npy_uintp r, npy_uintp c)
{
    npy_double s = 0;
    for (npy_uintp i = 0; i < r; i++)
    {
        s += vec_dot_vec(A[i], x, c);
    }
    s += sum_vec(b, r);
    return s;
}

static npy_double sum_vec(const npy_double v[], npy_uintp n)
{
    npy_double s = 0;
    for (npy_uintp i = 0; i < n; i++)
    {
        s += v[i];
    }
    return s;
}

static npy_double vec_dot_vec(const npy_double v1[], const npy_double v2[], npy_uintp n)
{
    npy_double s = 0;
    for (npy_uintp i = 0; i < n; i++)
    {
        s += v1[i] * v2[i];
    }
    return s;
}

static npy_double vec_dot_mat_dot_vec(npy_double x[], npy_double *A[], npy_double y[], npy_uintp nx, npy_uintp ny)
{
    npy_double s = 0;
    for (npy_uintp i = 0; i < nx; i++)
    {
        for (npy_uintp j = 0; j < ny; j++)
        {
            s += x[i] * A[i][j] * y[j];
        }
    }
    return s;
}
