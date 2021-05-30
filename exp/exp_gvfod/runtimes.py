import os

import click
import pandas as pd

@click.command()
@click.argument("dir", type=click.Path(exists=True))
def main(dir):
    json_files = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            json_files.append(os.path.join(dir, file))
    print(json_files)

    # Combine all the files, add columns for default vs non-default parameters, add columns for tuning-data vs test-data
    data = None
    for file in json_files:
        df = pd.read_json(file)
        df["default_params"] = 1 if "default_True" in os.path.split(file)[1] else 0
        df["tuning_data"] = 1 if "trainloss_True" in os.path.split(file)[1] else 0
        if data is None:
            data = df
        else:
            data = data.append(df, ignore_index=True)

    # Hacky fix for how GVFOD and MC are evaluated (5 times - once for each outlier type)
    data.loc[
        data["Algorithm"].isin(["MarkovChain", "GVFOD"]),
        "Test Time"
    ] /= 5

    algorithms = data["Algorithm"].unique()
    runtimes = pd.DataFrame(index=algorithms, columns=["train665", "train1384", "test"])
    # For each algorithm
    #   For each training size in [700, 1400]
    #       Get 5th pctile runtime (linear regression, 95CI on
    #   Get 5th pctile runtime testing size

    for alg in algorithms:
        for trainsize in [665, 1384]:
            # select that algorithm, and training size
            selection = data.loc[
                (data["Training Size"] == trainsize) & (data["Algorithm"] == alg),
                ["Training Size", "Train Time"]
            ]
            assert len(selection) > 0
            runtimes.loc[alg, f"train{trainsize}"] = selection.quantile(0.05)["Train Time"]
        # Testing time
        selection = data.loc[
            ((data["Algorithm"] == alg) & (data["Training Size"] == 665)),
            ["Test Time"]
        ]
        runtimes.loc[alg, "test"] = selection.quantile(0.05)["Test Time"]

    pd.options.display.float_format = "{:,.2f}".format
    print(runtimes)

if __name__ == "__main__":
    main()