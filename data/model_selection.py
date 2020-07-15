""" Additional utilities to parallel the functionality of sklearn.model_selection"""
import numpy as np

from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples


class TimeSeriesFolds(_BaseKFold):
    """
    An implementation of Time-Series cross-validating
    A class to split time series databases such that training requirements can be evaluated.

    Initialization parameters:
    Args:
        n_splits (int): number of folds to create (same as sklearn.model_selection.KFold)
        min_train_size (int, None): Ensure sufficient training data
        max_train_size (int, None): whether or not to throw out earlier data. Useful for statistical validation
        min_test_size (int, None): the minimum testing data to retain. Useful for high n_splits, such that the last
                                   folds don't have too small of a testing set
        max_test_size (int, None): the maximum testing data to use. If none, uses all possible testing data.
        delay (int): The number of samples to delay between
    """

    def __init__(self, n_splits,
                 min_train_size=1, max_train_size=None,
                 min_test_size=1, max_test_size=None,
                 delay=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        for arg in (max_train_size, max_test_size):
            assert isinstance(arg, int) or (arg is None)
        for arg in (min_train_size, min_test_size, delay):
            assert isinstance(arg, int)
        if (max_train_size is not None) and (min_train_size > max_train_size) :
            raise ValueError("Min/Max value mismatch")
        if (max_test_size is not None) and (min_test_size > max_test_size):
            raise ValueError("Min/Max value mismatch")

        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.min_test_size = min_test_size
        self.max_test_size = max_test_size
        self.delay = delay

    def split(self, X, y=None, groups=None):
        """ Generates indices to split training and testing data

        Args:
            X:
            y: Not used, exists for compatibility
            groups: Not used, exists for compatibility

        Returns:
            train (np.ndarray): indices for training set
            test (np.ndarray): indices for testing set

        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)  # len(X)
        n_splits = self.n_splits

        trainm = self.min_train_size
        trainM = self.max_test_size if self.max_test_size is not None else np.inf
        testm = self.min_test_size
        testM = self.max_test_size if self.max_test_size is not None else np.inf
        delay = self.delay

        if (n_samples - (trainm + delay + testm) < n_splits - 1):
            raise ValueError

        # The datum for each fold will be the index of the first test sample
        self.test_starts = np.linspace(trainm + delay, n_samples - testm, n_splits, dtype=int)
        indices = np.arange(n_samples)
        for test_start in self.test_starts:
            test_end = min(test_start + testM, n_samples)

            train_end = test_start - delay
            train_start = max(test_start - delay - trainM, 0)

            yield indices[train_start:train_end], indices[test_start:test_end]
