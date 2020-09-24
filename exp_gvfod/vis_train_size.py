import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

ylabel_map = {f1_score: "F1",
              precision_score: "Precision",
              recall_score: "Recall"}


def plot_results(json_filepath, metrics, linelabel=True):
    from itertools import cycle

    title = "Outlier Detection Training Data Requirements \n " \
            "Outlier: OUTLIER \n " \
            "Delay: DELAY"
    xlabel = r"Training Data"

    res = pd.read_json(json_filepath)
    res["Algorithm Class"] = res["Algorithm"].map(
        lambda alg: "RL-based" if alg in ["GVFOD", "MarkovChain"] else "Outlier-Detection")

    outlier_names = parse_outlier_names_from_columns(res.columns)


    for abn_str in outlier_names[1:]:
        fig, axs = plt.subplots(1, len(metrics), figsize=(10, 6))
        for i, (ax, metric) in enumerate(zip(axs, metrics)):
            nor_str = outlier_names[0]
            metric_name = metric.__name__
            res[f"{metric_name}_{abn_str}"] = score(metric,
                                                    res["c_" + nor_str],
                                                    res["ic_" + nor_str],
                                                    res["c_" + abn_str],
                                                    res["ic_" + abn_str])

            p = sns.lineplot(
                x='Training Size',
                y=f"{metric_name}_{abn_str}",
                hue='Algorithm',
                style='Algorithm Class',
                ci=95,
                data=res,
                ax=ax,
            )

            if ax is not axs[-1]:
                p.legend_.remove()
                pass
            else:
                ax.legend(loc="lower right")
                for line in ax.get_legend().get_texts():
                    if line.get_text() in ["Algorithm", "Algorithm Class"]:
                        line.set_text(r"$\mathbf{" + line.get_text().split(" ")[-1] + "}$")
            ax.set_ylabel(ylabel_map[metric])
            ax.set_ylim(0, 1)
            ax.set_xlabel(xlabel)
        plt.suptitle(title.replace("OUTLIER", abn_str).replace("DELAY", str(res.Delay.unique()[0])))
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(json_filepath.replace(".json", f"_{abn_str}.png"))


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
            return metric(1 - pred, 1 - true, **kwargs)

        return new_metric


    # plot_results(
    #     r"exp_gvfod\results_for_2020_08_report\exp_train_size_delay_720_default_False.json",
    #     [precision_score, recall_score, f1_score]
    # )

    dir = "exp_gvfod/results_for_2020_08_report/"
    for file in os.listdir(dir):
        if file.endswith(".json"):
            plot_results(
                os.path.join(dir, file),
                [precision_score, recall_score, f1_score]
            )
