import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data.dataloader import get_robot_arm_data

plt.rcParams["font.family"] = "Linux Libertine"



def main(compare_to_failure=None):
    """ Plots a single period of normal robot arm data """
    # select some data
    X, y, labels = get_robot_arm_data(return_class_labels=True)
    X = X[y == 0, :]  # Get normal
    X = X[100, :]  # Get the 100th normal period of data
    # print(X)

    if compare_to_failure is not None:
        Xf, _, _ = get_robot_arm_data(return_class_labels=True)
        Xf = Xf[y == labels.index(compare_to_failure), :]
        Xf = Xf[100, :]

    time = np.tile(np.arange(2000) / 200, 3)
    sensor = (["Position (rad)"] * 2000) + (["Torque (Nm)"] * 2000) + (["Tension (N)"] * 2000)

    ts = pd.DataFrame({"value": X, "time": time, "sensor": sensor})
    if compare_to_failure is not None:
        ts["value_failure"] = Xf
    print(ts)

    fig, axs = plt.subplots(3, 1, figsize=(3.33 * 1.5, 3 * 1.5))
    # plot the data

    for ax, s in zip(axs, ["Position (rad)", "Torque (Nm)", "Tension (N)"]):

        ax.text(0.9, 0.7,
                s,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        g = sns.lineplot(
            data=ts.loc[ts["sensor"] == s],
            x="time", y="value",
            # palette="crest",
            linewidth=2,
            color="black",
            # zorder=5,
            # col_wrap=3,
            legend=False,
            ax=ax
        )
        if compare_to_failure is not None:
            g = sns.lineplot(
                data=ts.loc[ts["sensor"] == s],
                x="time", y="value_failure",
                # palette="crest",
                linewidth=2,
                color="red",
                # zorder=5,
                # col_wrap=3,
                legend=False,
                ax=ax
            )

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(None)
        if s == "Tension (N)":
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel(None)

    if compare_to_failure:
        plt.suptitle(f"Comparing normal (black) to {compare_to_failure} failure (red)")
    plt.tight_layout()
    plt.show()
    # plt.savefig("data/vis_robot_arm_data_out.png", dpi=300)


    return

if __name__ == "__main__":
    main("loose_l2")