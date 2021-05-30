import os

import click
import pandas as pd
from scipy.stats import ttest_rel

from .vis_train_size import score, f1_score


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--train", "trainsize", default=2000)
def main(file, trainsize):
    data = pd.read_json(file)

    assert (data.loc[:, "Training Size"] == trainsize).any()
    # Drop all other training sizes
    data = data.loc[data["Training Size"] == trainsize, :]
    assert len(data) > 0

    data["f1_loose_l1"] = score(f1_score,
                                data["c_Normal"],
                                data["ic_Normal"],
                                data["c_loose_l1"],
                                data["ic_loose_l1"])

    algorithms = data["Algorithm"].unique()
    p_values = pd.DataFrame(index=algorithms, columns=algorithms)

    for alg1 in algorithms:
        for alg2 in algorithms:
            if alg1 == alg2:
                continue
            selection = data.loc[(data["Algorithm"].isin([alg1, alg2])), :]
            pivoted = pd.pivot(selection, values="f1_loose_l1", columns="Algorithm",
                               index=["Time of Start"])

            p_values.at[alg1, alg2] = ttest_rel(pivoted[alg1], pivoted[alg2])[1]

    print(p_values)

if __name__ == "__main__":
    main()
