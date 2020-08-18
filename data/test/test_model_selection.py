import unittest

from numpy import testing

from ..model_selection import *

class TestTSF(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.tsf_ex1 = TimeSeriesFolds(
            n_splits=2, min_train_size=2, max_train_size=2,
            min_test_size=2, max_test_size=2, delay=1
        )
        self.tsf_ex2 = TimeSeriesFolds(
            n_splits=2, delay=2
        )
        self.tsf_ex3 = TimeSeriesFolds(
            n_splits=3, min_train_size=2, max_train_size=6,
            min_test_size=2, max_test_size=2, delay=2
        )
        super().__init__(*args, **kwargs)

    def test_TSF1(self):

        X = np.arange(6).reshape(-1, 1)

        idxs = []
        for trn_idx, tst_idx in self.tsf_ex1.split(X):
            idxs.append((trn_idx, tst_idx))

        self.assertEqual(len(idxs), 2)
        testing.assert_equal(idxs[0][0], [0, 1])
        testing.assert_equal(idxs[0][1], [3, 4])
        testing.assert_equal(idxs[1][0], [1, 2])
        testing.assert_equal(idxs[1][1], [4, 5])

    def test_TSF2(self):

        X = np.arange(5).reshape(-1, 1)
        with self.assertRaises(ValueError):
            for trn_idx, tst_idx in self.tsf_ex1.split(X):
                print(trn_idx, tst_idx)

    def test_TSF3(self):
        expected = [
            ([0], [3,4,5]),
            ([0,1,2], [5])
        ]
        X = np.arange(6).reshape(-1, 1)
        for a, b in zip(self.tsf_ex2.split(X), expected):
            testing.assert_equal(a[0], b[0])
            testing.assert_equal(a[1], b[1])

    def test_TSF4(self):
        with self.assertRaises(ValueError):
            tsf_err = TimeSeriesFolds(5, min_train_size=10, max_train_size=5)
        with self.assertRaises(ValueError):
            tsf_err = TimeSeriesFolds(5, min_test_size=10, max_test_size=5)

    def test_TSF5(self):
        X = np.arange(10).reshape(-1, 1)
        expected = [
            ([0, 1], [4, 5]),
            ([0, 1, 2, 3], [6, 7]),
            ([0, 1, 2, 3, 4, 5], [8, 9])
        ]
        for a, b in zip(self.tsf_ex3.split(X), expected):
            testing.assert_equal(a[0], b[0])
            testing.assert_equal(a[1], b[1])