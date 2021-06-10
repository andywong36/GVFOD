import os
from multiprocessing import pool, RawArray
import time

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..data import dataloader as dtl
from ..data import model_selection as ms
from .train_size import global_data, init_data, arm_period, process
from .train_size_settings import *


@click.group()
def cli():
    pass


@cli.command("test")
@click.option("-l", "--lag", "delay", show_default=True, help="The amount of samples between training and testing",
              default=0)
@click.option("--default-param/--no-default-param", "default", required=True,
              help="use default or tuned (no-default) parameters")
@click.option("--tuning-data/--no-tuning-data", "tuning_data", required=True,
              help="evaluate on the tuning data, or testing (no-tuning-data) data")
@click.argument("dest", type=click.Path(exists=True))
@click.argument("algs", nargs=-1)
def test(dest, delay, default, tuning_data, algs):
    """

    """
    if delay != 0:
        raise NotImplementedError("Profiling only for --lag 0")
    if default:
        raise NotImplementedError("Profiling only for --no-default-param")
    if tuning_data:
        raise NotImplementedError("Profiling only for --no-tuning-data")

    file = os.path.join(dest, f"runtimes_delay_{delay}_default_{default}_trainloss_{tuning_data}.json")

    n_experiments = 20  # Training data starts are offset between experiments
    minutes_between_exps = 10  # The time between each experiment start
    # delay = 0  # 720 is 2 hours
    total_normal = 1400 + 595 + delay  # Amount of normal data per experiment, includes both training and testing sizes
    # Note that hours_of_data = n_sweeps * arm_period / 3600
    n_splits = 2

    min_train_size = 700
    min_test_size = 595
    max_test_size = 595
    n_param_test = 1000  # the [minimum] test size used in parameter sweeping.

    start = time.time()

    ## Read all the data and store it in global memory
    X, y, class_labels = dtl.get_robot_arm_data(return_class_labels=True)
    highTfilter = (y == 0) | (y == class_labels.index("highT"))
    X = X[highTfilter, :]
    y = y[highTfilter]
    y = (y != 0).astype(int)
    class_labels = [class_labels[0], "highT"]
    print(class_labels)
    ## Use only the data that was not used for training in parameter search.
    # Since parameter search used only the first 50% of the data, we can use
    # the last 50%, as well as the last 1000 samples (testing) in the first 50%.
    _X_list = []
    _y_list = []
    for i in np.unique(y):
        X_train, X_holdout, y_train, y_holdout = train_test_split(X[y == i], y[y == i], train_size=0.5, shuffle=False)
        if not tuning_data:
            if i == 0:
                # This data was never trained on in hyperparameter sweeping.
                _X_list.append(X_train[-n_param_test:])
                _y_list.append(y_train[-n_param_test:])
            _X_list.append(X_holdout)
            _y_list.append(y_holdout)
        else:
            _X_list.append(X_train)
            _y_list.append(y_train)
            if i == 0:
                # This data was never trained on in hyperparameter sweeping.
                _X_list.append(X_holdout[:n_param_test])
                _y_list.append(y_holdout[:n_param_test])
    X = np.concatenate(_X_list)
    y = np.concatenate(_y_list)

    X_nor, y_nor = X[y == 0], y[y == 0]
    X_abn, y_abn = X[y != 0], y[y != 0]

    # Scale the data
    ss = StandardScaler()
    X_nor_ss = ss.fit_transform(X_nor)
    X_abn_ss = ss.transform(X_abn)

    # Save the data into a RawArray
    def raw_array_from_ndarray(arr):
        if arr.dtype == np.float64:
            raw = RawArray('d', arr.size)
            _tempnparray = np.frombuffer(raw, np.float64).reshape(arr.shape)
        elif arr.dtype == np.int64 or arr.dtype == np.int32:
            raw = RawArray('i', arr.size)
            _tempnparray = np.frombuffer(raw, np.int32).reshape(arr.shape)
        else:
            raise TypeError(f"Unknown numpy dtype: {arr.dtype}")
        np.copyto(_tempnparray, arr)
        return raw

    X_nor_r = raw_array_from_ndarray(X_nor)
    X_nor_ss_r = raw_array_from_ndarray(X_nor_ss)
    y_nor_r = raw_array_from_ndarray(y_nor)
    X_abn_r = raw_array_from_ndarray(X_abn)
    X_abn_ss_r = raw_array_from_ndarray(X_abn_ss)
    y_abn_r = raw_array_from_ndarray(y_abn)

    processes = 1

    p = pool.Pool(processes=processes, initializer=init_data, initargs=[X_nor_r, X_nor_ss_r, y_nor_r, X_nor.shape,
                                                                        X_abn_r, X_abn_ss_r, y_abn_r, X_abn.shape])
    jobs = []  # Contains the returns from pool.apply_async. Iterate through these to get the Series.

    # Plan the experiments
    # The manipulated variables are:
    #   Algorithm
    #   Time of start
    #   Training Size
    # The return should be:
    #   (c_{class}, ic_{class} for class in [normal, abn1, abn2, ...])
    #   where c_{class} is the number of correct classifications of a sample whose true class is class
    #     and ic{class} is the number of incorrect classifications

    # The results dataframe is defined as:
    columns = [("Algorithm", str),
               ("Time of Start", int),
               ("Training Size", int),
               ("Testing Size", int),
               ("Delay", int),
               ("Train Time", float),
               ("Test Time", float),
               ]
    for data_class in class_labels:
        columns.append((f"c_{data_class}", int))
        columns.append((f"ic_{data_class}", int))
    results_df = pd.DataFrame({name: pd.Series([], dtype=dtype) for name, dtype in columns})

    # Calculate start times
    # Starts every 30 minutes
    step = minutes_between_exps * 60 / arm_period
    experiment_starts = np.arange(0, n_experiments * step, step, dtype=int)

    splitter = ms.TimeSeriesFolds(n_splits=n_splits,
                                  min_train_size=min_train_size,
                                  min_test_size=min_test_size,
                                  max_test_size=max_test_size,
                                  delay=delay)

    for exp_start in reversed(experiment_starts):
        assert exp_start + total_normal < len(X_nor)  # check that there is enough data
        for i, (train_idx, test_idx) in enumerate(splitter.split(X_nor[exp_start:exp_start + total_normal])):
            for exp in [
                if_exp, ocsvm_exp, lof_exp, abod_exp, kNN_exp, hbos_exp, mcd_exp, pca_exp,
                markov_exp,
                hmm_exp,
                gvfod_exp
            ]:
                if (len(algs) > 0) and (exp.clfname not in algs):
                    continue
                if default:
                    raise NotImplementedError
                    # if exp.clfname not in ["GVFOD", "MarkovChain", "HMM"]:
                    #     contamination = exp.kwargs["contamination"]
                    #     exp.kwargs.clear()
                    #     exp.kwargs["contamination"] = contamination
                    # elif exp.clfname == "GVFOD":
                    #     exp.kwargs["divs_per_dim"] = [10, 10, 10]
                    #     exp.kwargs["numtilings"] = 10
                    #     exp.kwargs["discount_rate"] = 0.9
                    #     exp.kwargs["learn_rate"] = 0.01
                    #     exp.kwargs["lamda"] = 0.1
                    #     exp.kwargs["beta"] = 250
                    # else:
                    #     pass  # Keep MarkovChain and HMM experiments the same.

                jobs.append(
                    p.apply_async(
                        process,
                        args=(exp, train_idx + exp_start, test_idx + exp_start, class_labels)
                    )
                )

    p.close()

    for i, job in enumerate(tqdm(jobs)):
        results_df.loc[i] = job.get()

    p.join()
    results_df.to_json(file)

    print(f"Time elapsed: {time.time() - start}")


@cli.command()
@click.argument("dir", type=click.Path(exists=True))
def summary(dir):
    json_files = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            json_files.append(os.path.join(dir, file))

    data = None
    if len(json_files) != 1:
        raise ValueError("More than one json file in provided directory")
    for file in json_files:
        data = pd.read_json(file)

    print(data)
    algorithms = data["Algorithm"].unique()
    trainsizes = data["Training Size"].unique()
    runtimes = pd.DataFrame(index=algorithms, columns=[*[f"train{n}" for n in trainsizes], "test"])
    melted = data.melt(id_vars=["Algorithm", "Time of Start", "Training Size"], value_vars=["Train Time", "Test Time"],
                       value_name="Time")
    # For each algorithm
    #   For each training size in [700, 1400]
    #       Get 5th pctile runtime (linear regression, 95CI on
    #   Get 5th pctile runtime testing size

    fig, axs = plt.subplots(3, 4)
    for i, alg in enumerate(algorithms):
        melted2 = pd.DataFrame(columns=["var", "val"])
        for trainsize in trainsizes:
            # select that algorithm, and training size
            selection = data.loc[
                (data["Training Size"] == trainsize) & (data["Algorithm"] == alg),
                ["Training Size", "Train Time"]
            ]
            melted2 = melted2.append(pd.DataFrame({
                "var": [f"train{trainsize}"] * len(selection),
                "val": selection["Train Time"].values.flatten()
            }))
            assert len(selection) > 0
            runtimes.loc[alg, f"train{trainsize}"] = selection.quantile(0.5)["Train Time"]
        # Testing time
        selection = data.loc[
            ((data["Algorithm"] == alg) & (data["Training Size"] == trainsizes[0])),
            ["Test Time"]
        ]
        melted2 = melted2.append(pd.DataFrame({
            "var": ["test"] * len(selection),
            "val": selection["Test Time"].values.flatten()
        }))
        runtimes.loc[alg, "test"] = selection.quantile(0.05)["Test Time"]

        # Plotting of each algorithm

        ax = axs[i // 4][i % 4]
        sns.boxplot(x="val", y="var", data=melted2, ax=ax)
        ax.set(title=alg)

    pd.options.display.float_format = "{:,.2f}".format
    print(runtimes)
    plt.show()


if __name__ == "__main__":
    cli()
