import os
import unittest

import numpy as np

results = {
    0.05: 0.06523427030779635,
    0.01: 0.12134414357168773,
    0.001: 2444842671982.708,
}


class test_GVFOD(unittest.TestCase):
    def test_main(self):
        import sys
        import time

        from sklearn.metrics import accuracy_score

        from data.dataloader import get_robot_arm_data
        from gvfod import GVFOD

        pct_normal = 0.01  # Percentage of normal data to use in both train/test
        pct_train = 0.6  # Percentage of training data

        X, y = get_robot_arm_data()
        X_nor = X[y == 0]
        X_nor = X_nor[:int(len(X_nor) * pct_normal)]
        X_abn = X[y != 0]

        model = GVFOD(
            space=[[10, 180],  # Angle limits
                   [-1, 1],  # Torque limits
                   [0, 300]],  # Tension limits

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
        print(model.models[0].tderrors[:10])
        print(model.models[0].tderrors[-10:])
        print(model.models[0].surprise[:10])
        print(model.models[0].surprise[-10:])
        print(f"The outlier threshold is {model.threshold_}")
        if pct_normal in results:
            self.assertEqual(model.threshold_, results[pct_normal])

        normal_pred = model.predict(X_nor)
        abnorm_pred = model.predict(X_abn)

        acc = accuracy_score(
            [0] * len(normal_pred) + [1] * len(abnorm_pred),
            np.hstack([normal_pred, abnorm_pred]))
        print("Accuracy score is:", acc)
        # self.assertGreaterEqual(acc, 0.6)



if __name__ == '__main__':
    unittest.main()
