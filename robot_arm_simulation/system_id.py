"""
A script for identifying the ODE parameters of the robot-arm testbed, based upon 2 sweeps (40s of data) of the arm.

Uses the torque as input, with the arm position as output. The loss is calculated as the SSE of the calculated vs true
arm positions.

Optimization is done with the hyperopt package. Distributed optimization is done using the MongoDB client, implemented
within hyperopt.
"""
from multiprocessing import Pool, TimeoutError
import math

import numpy as np
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

from dynamics import RobotArmDynamics


def objective(x):
    """ Same function signature as ode_const(). Used to define the objective for hyperopt """
    import time

    from scipy import integrate
    from hyperopt import STATUS_OK, STATUS_FAIL

    kwargs = x
    model = RobotArmDynamics(**kwargs)

    y0 = np.array([5.33249302e-01, 0, 5.33249314e-01, 0, 0.344213835078322, 0])

    start = time.time()

    p = Pool(processes=1)
    res = p.apply_async(func=integrate.solve_ivp,
            kwds={
                "fun": model.ode,
                "t_span": (0, 20),
                "y0": y0,
                "t_eval": model.data["Time"],
                "method": "LSODA",
                "jac": model.J,
            })

    try:
        results = res.get(timeout=300)
        return {
            # Main results
            'loss': np.sum((results.y[4, :] - model.data["Angle"].values) ** 2),
            'status': STATUS_OK,

            # Other results
            'start_time': start,
            'elapsed': time.time() - start,
            # 'responses': results.y.tostring()
        }
    except TimeoutError:
        return {
            "status": STATUS_FAIL
        }
    finally:
        p.close()
        p.terminate()

if __name__ == "__main__":
    trials = MongoTrials(r'mongo://melco.cs.ualberta.ca:27017/hyperopt_db/jobs',
                         exp_key='exp1-fixedT')
    space = {
        'r_m': hp.uniform('r_m', 0.01, 0.015),
        'r_i': hp.uniform('r_i', 0.01, 0.015),
        'r_a': hp.uniform('r_a', 0.017, 0.021),
        'J_m': hp.loguniform('J_m', -11, -8),
        'I_i': hp.loguniform('I_i', -15, -11),
        'I_a': hp.uniform('I_a', 0.3, 0.8),
        'l_1': hp.uniform('l_1', 0.126, 0.128),
        'l_2': hp.uniform('l_2', 0.1515, 0.1535),
        'l_3': hp.uniform('l_3', 0.177, 0.179),
        'slope1': hp.normal('slope1', -0.28, 0.02),
        'slope2': hp.normal('slope2', 0.21, 0.02),
        'EA': hp.loguniform('EA', 8, 12.5),
        'C_L': hp.uniform('C_L', 5, 10),
        'base_T': 155,
        'f1m': hp.lognormal('f1m', math.log(0.00015), 1),
        'f2m': hp.lognormal('f2m', math.log(0.00015), 1),
        'f1i': hp.lognormal('f1i', math.log(0.00001), 1),
        'f2i': hp.lognormal('f2i', math.log(0.00001), 1),
        'f1a': hp.lognormal('f1a', math.log(0.0004), 1),
        'f2a': hp.lognormal('f2a', math.log(0.0006), 1),
    }

    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=40000,
        trials=trials,
        max_queue_len=32,
    )

    # space = hp.uniform('x', -2, 2)
    # best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
