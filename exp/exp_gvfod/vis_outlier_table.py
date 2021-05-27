import numpy as np
import pandas as pd

from exp_gvfod.vis_train_size import *

outliers = ["loose_l1", "loose_l2", "tight", "sandy", "highT"]

def create_outlier_table(file, metric):

    res = pd.read_json(file)

    # Select the runs with maximum training size
    res = res.loc[res["Training Size"] == res["Training Size"].max()]

    res["Algorithm Class"] = res["Algorithm"].map(
        lambda alg: "RL-based" if alg in ["GVFOD", "MarkovChain"] else "Outlier-Detection")

    outlier_names = parse_outlier_names_from_columns(res.columns)
    metric_name = metric.__name__

    for abn_str in outlier_names[1:]:
        nor_str = outlier_names[0]
        res[f"{metric_name}_{abn_str}"] = score(metric,
                                                res["c_" + nor_str],
                                                res["ic_" + nor_str],
                                                res["c_" + abn_str],
                                                res["ic_" + abn_str])

    # Put these results into a table
    table = pd.DataFrame(columns=outliers, index=res["Algorithm"].unique(), dtype=float)
    for alg, row in table.iterrows():
        for abn_str in outlier_names[1:]:
            avg = res.loc[
                (res["Algorithm"] == alg),
                f"{metric_name}_{abn_str}"
            ].mean()
            table.loc[alg, abn_str] = avg

    ax = sns.heatmap(table, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set(title="Recall of Outlier Detection Algorithms\n"
                 "Delay: 720\n"
                 "Using Optimized Parameters\n"
                 "Training Size: 2000 (Evaluation Data)")
    plt.tight_layout()
    plt.show()

    pass


if __name__ == "__main__":
    create_outlier_table(r"exp_gvfod\results_for_2020_08_report\exp_train_size_delay_720_default_False.json", recall_score)