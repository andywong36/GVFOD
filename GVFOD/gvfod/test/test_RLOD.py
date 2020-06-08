import os
import unittest


class TestRLOD(unittest.TestCase):
    def test_main(self):
        import time

        import numpy as np
        from sklearn.metrics import accuracy_score

        from data.dataloader import get_robot_arm_data
        from ..RLOD import RLOD

        pt_train = 0.6  # Percentage of training data

        X, y = get_robot_arm_data()
        X_nor = X[y == 0]
        X_abn = X[y != 0]

        model = RLOD(n_sensors=X.shape[1] // 2000,
                     divisions=[4, 4, 4],
                     wrap_idxs=None,
                     int_idxs=None,
                     numtilings=32,
                     state_size=2048,
                     discount_rate=0.986,
                     learn_rate=0.005,
                     lamda=0.25,
                     beta=1000,
                     contamination=0.05)

        cutoff = int(len(X_nor) * pt_train)
        start = time.time()
        model.fit(X_nor[:cutoff])
        print(f"It takes {time.time() - start}s to train {cutoff} samples.")

        normal_pred = model.predict(X_nor)
        abnorm_pred = model.predict(X_abn)

        print("Accuracy score is (normal): " + accuracy_score(0, normal_pred))
        self.assertGreaterEqual(accuracy_score(0, normal_pred), 0.6)
        print("Accuracy score is (abnorm): " + accuracy_score(1, abnorm_pred))
        self.assertGreaterEqual(accuracy_score(0, abnorm_pred), 0.6)



if __name__ == '__main__':
    unittest.main()
