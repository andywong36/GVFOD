import unittest


class TestState(unittest.TestCase):
    def test_main(self):
        import pandas as pd

        from ..scaling import scaling
        from ..state import Tiler

        tiler = Tiler(scaling["machine"])
        sample_data = pd.read_pickle("data/pickles/machine_normal.pkl").iloc[:1000000, :]
        phi, x_scaled = tiler.encode(sample_data)

        self.assertLess(phi.max(), tiler.state_size)
        self.assertEqual(len(sample_data), len(phi))
        self.assertEqual(phi.shape[1], tiler.tilings)


if __name__ == '__main__':
    unittest.main()
