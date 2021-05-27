import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.collections
from sklearn.metrics import precision_score, recall_score, f1_score

from multiprocessing import Pool

ylabel_map = {f1_score: "F1",
              precision_score: "Precision",
              recall_score: "Recall"}

subplot_map = {0: "(a)",
               1: "(b)",
               2: "(c)"}

plt.rcParams["font.family"] = "Linux Libertine"

TESTRUN = False
LOOSE_L1_ONLY = True


def plot_results(json_filepath, metrics, linelabel=True):
    from itertools import cycle

    # title = "Outlier Detection Training Data Requirements \n " \
    #         "Outlier: OUTLIER \n " \
    #         "Delay: DELAY"
    xlabel = r"Training Data (samples)"

    res = pd.read_json(json_filepath)
    res["Class"] = res["Algorithm"].map(
        lambda alg: "Temporal" if alg in ["GVFOD", "MarkovChain"] else "Multivariate")

    outlier_names = parse_outlier_names_from_columns(res.columns)

    for abn_str in outlier_names[1:]:
        if LOOSE_L1_ONLY and (abn_str != "loose_l1"):
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
                # print(line)
                if ("GVFOD" in str(line)) or ("line9" in str(line)):
                    line.set_lw(1.5)
                    line.set_c("k")

            if i is not 1:  # put legend in Recall graph
                p.legend_.remove()
            else:
                ax.legend(loc="lower right", frameon=True)
                for line in ax.get_legend().get_texts():
                    if line.get_text() in ["Algorithm", "Class"]:
                        # line.set_text(r"$\mathbf{" + line.get_text().split(" ")[-1] + "}$")
                        # line.set_text(line.get_text().split(" ")[-1])
                        if line.get_text() == "Class":
                            line.set_text("Algorithm Type")
                        line.set_ha("left")
                        # print(line.get_position())
                        line.set_position((-35, 0))
            # if i is not 0:
            #     ax.get_yaxis().set_ticks([])
            #     ax.set_yticklabels([])
            ax.set_title(subplot_map[i] + "  " + ylabel_map[metric], weight="bold", y=-0.15)
            ax.set_ylim(0, 1.01)
            ax.set_xlim(0, 2000)
            ax.set_xlabel(xlabel)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            p.yaxis.label.set_visible(False)
        # plt.suptitle(title.replace("OUTLIER", abn_str).replace("DELAY", str(res.Delay.unique()[0])))
        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(json_filepath.replace(".json", f"_{abn_str}.png").replace("exp_train_size_", "3-"), dpi=300)
        if TESTRUN:
            break

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


    p = Pool()

    # plot_results(
    #     r"exp_gvfod\results_for_2020_08_report\exp_train_size_delay_720_default_False.json",
    #     [precision_score, recall_score, f1_score]
    # )

    dir = "exp_gvfod/results_for_2020_08_report/"
    for file in os.listdir(dir):
        if file.endswith(".json"):
            if TESTRUN:
                plot_results(
                    os.path.join(dir, file),
                    [precision_score, recall_score, f1_score]
                )
                break
            else:
                p.apply_async(plot_results, [os.path.join(dir, file), [precision_score, recall_score, f1_score]])

    p.close()
    p.join()

    print("Plotting Complete")
