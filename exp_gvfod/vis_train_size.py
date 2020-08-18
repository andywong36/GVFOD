import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from exp_gvfod.exp_train_size import arm_period


def plot_results(json_filepath, metric, linelabel=True):
    from itertools import cycle

    title = "Outlier Detection Training Data Requirements: Outlier OUTLIER \n Delay: DELAY"
    xlabel = r"Training Data (minutes of operation)"

    res = pd.read_json(json_filepath)
    res["Algorithm Class"] = res["Algorithm"].map(lambda alg: "RL-based" if alg in ["GVFOD", "MarkovChain"] else "Outlier-Detection")

    outlier_names = parse_outlier_names_from_columns(res.columns)
    metric_name = metric.__name__

    for abn_str in outlier_names[1:]:
        fig, ax = plt.subplots()
        nor_str = outlier_names[0]
        res[f"{metric_name}_{abn_str}"] = score(metric,
                                                res["c_" + nor_str],
                                                res["ic_" + nor_str],
                                                res["c_" + abn_str],
                                                res["ic_" + abn_str])

        sns.lineplot(
            x='Training Size',
            y=f"{metric_name}_{abn_str}",
            hue='Algorithm',
            style='Algorithm Class',
            ci=95,
            data=res,
            ax=ax
        )

        ax.legend()
        for line in ax.get_legend().get_texts():
            if line.get_text() in ["Algorithm", "Algorithm Class"]:
                line.set_text(r"$\mathbf{" + line.get_text().split(" ")[-1] + "}$")
        ax.set_ylabel(metric.__name__)
        ax.set_xlabel(xlabel)
        ax.set_title(title.replace("OUTLIER", abn_str).replace("DELAY", str(res.Delay.unique()[0])))

    return res


def parse_outlier_names_from_columns(columns):
    outlier_names = []
    for cname in columns:
        if cname.startswith("c_"):
            outlier_names.append("_".join(cname.split("_")[1:]))
    return outlier_names


def score(metric, true_negative, false_positive, true_positive, false_negative):
    scores = []
    for tn, fp, tp, fn in zip(true_negative, false_positive, true_positive, false_negative):
        true_vec, pred_vec = _create_true_pred_vec_from_count(tn, fp, tp, fn)
        scores.append(metric(true_vec, pred_vec))
    return scores


def _create_true_pred_vec_from_count(true_negative, false_positive, true_positive, false_negative):
    n_neg_true = true_negative + false_positive
    n_pos_true = true_positive + false_negative

    true_vec = np.concatenate([np.zeros(n_neg_true), np.ones(n_pos_true)]).astype(int)
    pred_vec = np.concatenate([np.zeros(true_negative),
                               np.ones(false_positive),
                               np.zeros(false_negative),
                               np.ones(true_positive)]).astype(int)
    return true_vec, pred_vec


if __name__ == "__main__":
    def flip_relevant_elements(metric):
        def new_metric(pred, true, **kwargs):
            return metric(1-pred, 1-true, **kwargs)
        return new_metric
    plot_results(r"exp_gvfod\results_for_2020_08_report\exp_train_size_results_v6.json",
                 flip_relevant_elements(recall_score))