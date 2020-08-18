import unittest

import numpy as np
from numpy import testing

from gvfod.clearn import clearn, clearn_ude
from gvfod.learn import learn, learn_ude, learn_ude_naive


class MyTestCase(unittest.TestCase):
    def test_clearn(self):
        tde_c = np.arange(5, dtype=np.double)
        tde_py = np.arange(5, dtype=np.double)
        w_c = np.arange(20, dtype=np.double)
        w_py = np.arange(20, dtype=np.double)
        resc = clearn(
            phi=np.arange(10, dtype=np.uintp).reshape(5, 2),
            y=np.arange(5, dtype=np.double),
            tde=tde_c,
            w=w_c,
            z=np.arange(20, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1
        )
        respy = learn(
            phi=np.arange(10, dtype=np.uintp).reshape(5, 2),
            y=np.arange(5, dtype=np.double),
            tde=tde_py,
            w=w_py,
            z=np.arange(20, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1
        )

        testing.assert_equal(tde_c, tde_py)
        testing.assert_equal(w_c, w_py)

    def test_clearn_eval(self):
        tde_c = np.arange(10, dtype=np.double)
        tde_py = np.arange(10, dtype=np.double)
        tde_py_n = np.arange(10, dtype=np.double)
        tde_no_ude = np.arange(10, dtype=np.double)


        w_c = np.arange(10, dtype=np.double)
        w_py = np.arange(10, dtype=np.double)
        w_py_n = np.arange(10, dtype=np.double)
        w_no_ude = np.arange(10, dtype=np.double)

        ude_c = np.arange(10, dtype=np.double)
        ude_py = np.arange(10, dtype=np.double)
        ude_py_n = np.arange(10, dtype=np.double)

        res_no_ude = learn(
            phi=np.tile(np.arange(10, dtype=np.uintp), 2).reshape(10, 2),
            y=np.arange(10, dtype=np.double),
            tde=tde_no_ude,
            w=w_no_ude,
            z=np.arange(10, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1,
        )

        resc = clearn_ude(
            phi=np.tile(np.arange(10, dtype=np.uintp), 2).reshape(10, 2),
            y=np.arange(10, dtype=np.double),
            tde=tde_c,
            w=w_c,
            z=np.arange(10, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1,
            ude=ude_c, beta=3
        )
        respy = learn_ude(
            phi=np.tile(np.arange(10, dtype=np.uintp), 2).reshape(10, 2),
            y=np.arange(10, dtype=np.double),
            tde=tde_py,
            w=w_py,
            z=np.arange(10, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1,
            ude=ude_py, beta=3
        )
        respyn = learn_ude_naive(
            phi=np.tile(np.arange(10, dtype=np.uintp), 2).reshape(10, 2),
            y=np.arange(10, dtype=np.double),
            tde=tde_py_n,
            w=w_py_n,
            z=np.arange(10, dtype=np.double),
            gamma=0.1, lambda_=0.1, alpha=0.1,
            ude=ude_py_n, beta=3
        )
        testing.assert_equal(tde_py, tde_py_n)
        testing.assert_equal(tde_c, tde_py)
        testing.assert_equal(tde_c, tde_py_n)

        testing.assert_equal(w_no_ude, w_py_n)
        testing.assert_equal(w_no_ude, w_c)
        testing.assert_equal(w_py, w_py_n)
        testing.assert_equal(w_c, w_py)
        testing.assert_equal(w_c, w_py_n)

        testing.assert_allclose(ude_py, ude_py_n, rtol=2E-15)
        testing.assert_allclose(ude_c, ude_py, rtol=2E-15)
        testing.assert_allclose(ude_c, ude_py_n, rtol=2E-15)



if __name__ == '__main__':
    unittest.main()
