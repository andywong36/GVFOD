import os
import sys
from multiprocessing import Pool

import click

from matplotlib import pyplot as plt
import matplotlib.collections
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression as LR


@click.command()
@click.argument("src", nargs=1)
@click.argument("dst", nargs=1)
@click.argument("failures", nargs=-1)
def main(src, dst, failures):
    p = Pool()

    for file in os.listdir(src):
        if file.endswith(".json"):
            p.apply_async(plot_results, [os.path.join(src, file), dst,
                                         [precision_score, recall_score, f1_score], failures])

    p.close()
    p.join()

    print("Plotting Complete")


def plot_results(json_filepath, dst, metrics, failures):
    plt.rcParams["font.family"] = "Times New Roman"
    xlabel = r"Training Data (samples)"

    res = pd.read_json(json_filepath)
    res["Class"] = res["Algorithm"].map(
        lambda alg: "Temporal" if alg in ["GVFOD", "MarkovChain", "HMM"] else "Multivariate")

    outlier_names = parse_outlier_names_from_columns(res.columns)

    for abn_str in outlier_names[1:]:
        if (abn_str not in failures) and (len(failures) > 0):
            continue
        print(f"Working on abnormal class: {abn_str}")
        fig, axs = plt.subplots(1, len(metrics), figsize=(7 * 1.5, 7 * 0.8))
        for i, (ax, metric) in enumerate(zip(axs, metrics)):
            nor_str = outlier_names[0]
            metric_name = metric.__name__
            res[f"{metric_name}_{abn_str}"] = score(metric,
                                                    res["c_" + nor_str],
                                                    res["ic_" + nor_str],
                                                    res["c_" + abn_str],
                                                    res["ic_" + abn_str])

            if metric_name == "f1_score" and abn_str == "loose_l1":
                selection = res.loc[
                          (res["Algorithm"] == "GVFOD") & (res["Training Size"].isin([973, 1076])),
                          ["Training Size", f"{metric_name}_{abn_str}"]]
                model = LR()
                model.fit(selection["Training Size"].values.reshape(-1, 1), selection[f"{metric_name}_{abn_str}"].values.reshape(-1, 1))
                print(f"F1 score @ 1000, {os.path.split(json_filepath)[1]}: {model.predict(np.array([[1000]]))}")

            p = sns.lineplot(
                x='Training Size',
                y=f"{metric_name}_{abn_str}",
                hue='Algorithm',
                style='Class',
                ci=95,
                data=res,
                ax=ax,
                legend="brief",
            )

            # Assume the GVFOD is last to be plotted, set its error band to gray
            last_poly = None
            for item in ax.get_children():
                # print(item)
                if isinstance(item, matplotlib.collections.PolyCollection):
                    last_poly = item
            last_poly.set_color([(0.1, 0.1, 0.1, 0.2)])

            # Set GVFOD to heavier, and change it to black
            for line in ax.get_lines():
                # This is pretty hacky - but it works
                if ("GVFOD" in str(line)) or ("_line10" in str(line)):
                    line.set_lw(1.5)
                    line.set_c("k")

            if i is not 1:  # put legend in Recall graph
                p.legend_.remove()
            else:
                ax.legend(loc="lower right", frameon=True)
                for line in ax.get_legend().get_texts():
                    if line.get_text() in ["Algorithm", "Class"]:
                        if line.get_text() == "Class":
                            line.set_text("Algorithm Type")
                        line.set_ha("left")
                        line.set_position((-35, 0))

            ylabel_map = {f1_score: "F1",
                          precision_score: "Precision",
                          recall_score: "Recall"}

            subplot_map = {0: "(a)",
                           1: "(b)",
                           2: "(c)"}

            ax.set_title(subplot_map[i] + "  " + ylabel_map[metric], weight="bold", y=-0.15, fontsize=10)
            ax.set_ylim(0, 1.01)
            ax.set_xlim(0, 2000)
            ax.set_xlabel(xlabel)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            p.yaxis.label.set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 1])
        filename = (os.path.split(json_filepath)[1]) \
            .replace(".json", f"_{abn_str}.png") \
            .replace("train_size_delay_0_", "3-")
        print(filename)
        plt.savefig(os.path.join(dst, filename), dpi=300)

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
    main()
