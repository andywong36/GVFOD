import numpy as np
import pandas as pd
from tqdm import trange

import rlod.tiles3 as tile


class Tiler:
    """ Creates tile coded space """

    def __init__(self, scaling: dict, state_size=4096, tilings=32):
        """
        Args:
            x:
            options:
        """
        # Variable initialization
        assert np.iinfo(int).max > state_size + 1
        self.iht = tile.IHT(state_size)
        self.state_size = state_size
        self.tilings = tilings
        self.scaling = scaling

        self.x = None
        self.x_scaled = None
        self.phi = None
        self.columns = None

    def encode(self, x: pd.DataFrame):
        """
        Args:
            x: DataFrame to encode

        Returns:
            self.phi (np.ndarray): Tiled indices of binary vector
            self.x_scaled (pd.DataFrame): scaled version of input array

        """
        self.x = x

        # Scale the variables, and define the ints
        x_scaled = self.x.copy()
        for column in x_scaled.columns:
            spec = self.scaling[column]
            if spec["divs"] == "ignore":
                x_scaled.drop(columns=column, inplace=True)
                continue
            elif spec["divs"] == "int":
                x_scaled[column] = x_scaled[column].astype(np.int)
                continue
            else:
                x_scaled[column] = x_scaled[column] / (spec['max'] - spec['min']) * spec['divs']

        self.phi = np.empty((x.shape[0], self.tilings), dtype=int)
        self.x_scaled = x_scaled
        if self.columns is None:
            self.columns = self.x_scaled.columns
        assert np.all(self.columns == self.x_scaled.columns), "There are mismatched column labels in this dataset"

        # Identify variables to feed into tileswrap
        float_cols = self._float_columns(self.x_scaled.columns)
        int_cols = self._int_columns(self.x_scaled.columns)
        wrap = self._wrap_columns(float_cols)
        float_arr = self.x_scaled.loc[:, float_cols].values
        int_arr = self.x_scaled.loc[:, int_cols].values if len(int_cols) > 0 else [list()] * len(self.x_scaled)

        print("Tiling:")
        for i in trange(len(x)):
            self.phi[i] = tile.tileswrap(self.iht, self.tilings,
                                         float_arr[i],
                                         wrap,
                                         int_arr[i])

        return self.phi, self.x_scaled

    def _float_columns(self, existing):
        """ Returns the subset of float columns in self.scaling that actually exist in existing """
        cols = []
        for col in existing:
            if isinstance(self.scaling[col]["divs"], int):
                cols.append(col)
        return cols

    def _int_columns(self, existing):
        """ Returns the subset of int columns in self.scaling that actually exist in existing """
        cols = []
        for col in existing:
            if self.scaling[col]["divs"] == "int":
                cols.append(col)
        return cols

    def _wrap_columns(self, existing):
        """ Returns the number to wrap around, if the column is to be wrapped. else None
        Args:
            existing: the names of the columns for which FLOAT data exists
        Returns:

        """
        wraps = []
        for col in existing:
            if self.scaling[col]["wrap"]:
                wraps.append(self.scaling[col]["divs"])
            else:
                wraps.append(False)
        return wraps


def test():
    from rlod.scaling import scaling
    tiler = Tiler(scaling["machine"])
    sample_data = pd.read_pickle("data/pickles/machine_normal.pkl").iloc[:, :]
    hashed = tiler.encode(sample_data)
    return hashed


if __name__ == "__main__":
    print(test())
