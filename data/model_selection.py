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
        min_test_size (int, None): the minimum testing data to retain. Useful for high n_splits, such that the last
                                   folds don't have too small of a testing set
        max_train_size (int, None): whether or not to throw out earlier data. Useful for statistical validation
        use_all_testing_data (bool): default True. if False, then the testing set is only ever min_test_size in size.
    """

    def __init__(self, n_splits, min_test_size=None, max_train_size=None, use_all_testing_data=True):
        super().__init__(n_splits, shuffle=False, random_state=None)
        if min_test_size is not None:
            assert isinstance(min_test_size, int)
        if max_train_size is not None:
            assert isinstance(max_train_size, int)

        self.min_test_size = min_test_size
        self.max_train_size = max_train_size
        self.use_all_testing_data = use_all_testing_data

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
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(f"Cannot have n_folds = {n_folds} > n_samples = {n_samples}")
        indices = np.arange(n_samples)
        min_test_size = self.min_test_size if self.min_test_size else (n_samples // n_folds)
        _max_train_size = n_samples - min_test_size
        test_starts = range(_max_train_size // n_splits + _max_train_size % n_splits,
                            _max_train_size + 1,
                            _max_train_size // n_splits)
        for test_start in test_starts:
            if self.use_all_testing_data:
                test_indices = indices[test_start:n_samples]
            else:
                test_indices = indices[test_start:test_start + min_test_size]

            if self.max_train_size and self.max_train_size < test_start:
                train_indices = indices[test_start - self.max_train_size:test_start]
            else:
                train_indices = indices[:test_start]

            yield (train_indices, test_indices)


def unit_tests():
    X = np.zeros(22).reshape(-1, 1)
    splitter = TimeSeriesFolds(8, min_test_size=6)
    # for train, test in splitter.split(X):
    #     print(f"Train: {train}, Test: {test}")
    assert len(list(splitter.split(X))) == 8

    splitter = TimeSeriesFolds(8, min_test_size=6, max_train_size=5)
    # for train, test in splitter.split(X):
    #     print(f"Train: {train}, Test: {test}")

    splitter = TimeSeriesFolds(8, min_test_size=6, max_train_size=5, use_all_testing_data=False)
    for train, test in splitter.split(X):
        print(f"Train: {train}, Test: {test}")

    X = np.zeros(23).reshape(-1, 1)
    splitter = TimeSeriesFolds(8, min_test_size=6)
    # for train, test in splitter.split(X):
    #     print(f"Train: {train}, Test: {test}")
    assert len(list(splitter.split(X))) == 8

    splitter = TimeSeriesFolds(8, min_test_size=6, max_train_size=5)
    for train, test in splitter.split(X):
        # print(f"Train: {train}, Test: {test}")
        assert len(train) == 5 or len(train) == 3

    splitter = TimeSeriesFolds(8, min_test_size=6, max_train_size=5, use_all_testing_data=False)
    for train, test in splitter.split(X):
        # print(f"Train: {train}, Test: {test}")
        assert len(train) == 5 or len(train) == 3
        assert len(test) == 6
    assert len(list(splitter.split(X))) == 8


if __name__ == "__main__":
    unit_tests()
