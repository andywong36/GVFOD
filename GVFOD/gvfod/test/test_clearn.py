import unittest

import numpy as np
from ..surprise import TDLambdaGVF
from ..clearn import learn


class TestClearn(unittest.TestCase):
    def test_main(self):
        """ Tests clearn to see if it works the same as the learn method in TDLambdaGVF"""
        n_params = 10
        n_steps = 100
        np.random.seed(12345)
        w = np.random.randint(20, size=n_params)
        x = np.vstack(list(
            (np.random.choice(n_params, size=2, replace=False) for _ in range(n_steps))))
        n = np.random.normal(size=n_steps)
        y = np.zeros(n_steps)
        for i in range(n_steps - 1):
            y[i + 1] = np.sum(w[x[i, :]]) + n[i]

        agent1 = TDLambdaGVF(state_size=n_params, discount_rate=0.0, learn_rate=0.1, lamda=0.0, beta=3)
        agent2 = TDLambdaGVF(state_size=n_params, discount_rate=0.0, learn_rate=0.1, lamda=0.0, beta=3)

        agent1.learn(x, y)

        agent2.tderrors = np.zeros_like(agent1.tderrors)
        learn(x, y,
              agent2.tderrors, agent2.w, agent2.z, agent2.gamma, agent2.lamda,
              agent2.alpha)

        np.testing.assert_allclose(agent1.tderrors, agent2.tderrors)


if __name__ == "__main__":
    unittest.main()
