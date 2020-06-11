from contextlib import redirect_stdout
from functools import partial
import pickle
from multiprocessing import Pool
import os
import sys

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm

from main import SystemProperties
from system_id import objective


def sign(x):
    return 1 if x >= 0 else -1


def main():
    savefile = "sensitivity_results_o1.pkl"

    delta = 0.1
    if os.name != 'nt':
        os.nice(10)
        n_runs = 8000
    else:
        n_runs = 1
    # Define the search space
    kwargs = SystemProperties.optimized_params
    _d = delta
    _bounds = [[val[0] * (1 - _d * sign(val[0])), val[0] * (1 + _d * sign(val[0]))] for val in kwargs.values()]
    problem = {
        'num_vars': len(kwargs),
        'names': list(kwargs.keys()),
        'bounds': _bounds
    }
    param_values = saltelli.sample(problem, n_runs, calc_second_order=False)
    print(f"Number of tests: {len(param_values)}")
    # Set up parallel processing
    p = Pool(processes=15)
    _obj = partial(objective, tight_tol=False, argnames=list(kwargs.keys()))
    imap_iter = p.imap(_obj, param_values)
    # Calculate the sensitivity
    Y = np.zeros([len(param_values)])
    for i, res in enumerate(tqdm(imap_iter, total=len(param_values))):
        Y[i] = res['loss']
    # Finish up the pool
    p.close()
    p.join()

    # Perform analysis
    _stdout = sys.stdout
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=True)

    # Save results
    with open(savefile, 'wb') as f:
        pickle.dump(Si, f)

    return Si


def visualize(pkl_file):
    import matplotlib.pyplot as plt
    import pandas as pd

    with open(pkl_file, 'rb') as f:
        Si = pickle.load(f)
    _o1_cols = ['S1', 'S1_conf', 'ST', 'ST_conf']
    order1 = pd.DataFrame(index=SystemProperties.optimized_params.keys(),
                          columns=_o1_cols,
                          data=np.array([Si[col] for col in _o1_cols]).T)
    # order1["S1a"] = np.abs(order1["S1"])

    # order1.sort_values(by="S1a", ascending=False, inplace=True)
    order1.sort_values(by="S1", ascending=False, inplace=True)
    f, ax = plt.subplots(figsize=(6, 15))

    # sns.set_color_codes("muted")
    # sns.barplot(x="ST", y=order1.index, data=order1,
    #             label="Total Effects", color="b")

    fig, ax = plt.subplots()

    ax.bar(np.arange(len(order1)), order1["S1"],
           # label="First Order Effects", color="b",
           yerr=order1["S1_conf"])
    plt.xticks(np.arange(len(order1)), order1.index)

    # ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(
        # xlim=(0.0001, 0.1), ylabel="",
        xlabel="Sensitivity",
        title="Sensitivity to 10% deviation in parameter value",
        # xscale="log",
    )


if __name__ == "__main__":

    # Run the analysis, save to file
    if os.name == "posix":
        # Run this on the server
        Si = main()

    else:
        visualize("sensitivity_results_o2.pkl")
