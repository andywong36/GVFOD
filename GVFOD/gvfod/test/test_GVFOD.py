import os
import unittest

import numpy as np

class test_GVFOD(unittest.TestCase):
    def test_main(self):
        import time

        from sklearn.metrics import accuracy_score

        from data.dataloader import get_robot_arm_data
        from gvfod import GVFOD

        pct_train = 0.6  # Percentage of training data

        X, y = get_robot_arm_data()
        X_nor = X[y == 0]
        X_nor = X_nor[:int(len(X_nor) * 0.05)]
        X_abn = X[y != 0]

        model = GVFOD(space=[[10, 180],  # Angle limits
                             [-1, 1],  # Torque limits
                             [0, 300]  # Tension limits
                             ],
                      divs_per_dim=[4, 4, 4],
                      wrap_idxs=None,
                      int_idxs=None,
                      numtilings=32,
                      discount_rate=0.986,
                      learn_rate=0.005,
                      lamda=0.25,
                      beta=1000,
                      contamination=0.05)

        cutoff = int(len(X_nor) * pct_train)
        start = time.time()
        print("Starting fitting")
        model.fit(X_nor[:cutoff])
        print(f"It takes {time.time() - start}s to train {cutoff} samples.")

        normal_pred = model.predict(X_nor)
        abnorm_pred = model.predict(X_abn)

        acc = accuracy_score(
            [0]*len(normal_pred) + [1] * len(abnorm_pred),
            np.hstack([normal_pred, abnorm_pred]))
        print("Accuracy score is:", acc)
        self.assertGreaterEqual(acc, 0.6)


if __name__ == '__main__':
    unittest.main()
