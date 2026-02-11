#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import matplotlib

from common import *

def load_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(result_dir,"boottime", "mem.csv"))
    return df


def main():
    data = load_data()
    #print(data)

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
    data_melt = data.melt(var_name="System", value_name="Memory usage")
    data_melt["memMB"] = data_melt["Memory usage"]/1024
    plot = sns.barplot(ax=ax, data=data_melt, y ="System", x ="memMB", hue="System", errorbar="ci"
                        , edgecolor="black", linewidth=0.5
                        , palette=palette
                       #marker="H"
                       )
    #plot.set_xticks([0, 100, 200, 300])
    # ax.set_xticklabels(data['setup'].unique(), fontsize=FONTSIZE)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=FONTSIZE)
    plot.set_xticks([0, 100, 200, 300])
    #plot.set_yticklabels([0, 50, 100], fontsize=FONTSIZE)
    ax.set_xlabel("Memory usage (MiB)")
    # ax.set_xlabel("")
    # ax.legend(loc="upper left", title=None,fontsize=FONTSIZE
    # #    bbox_to_anchor=(0.5, -0.15),
    # #    ncol=2,
    # )
    ax.set_title(left_better_str, fontsize=FONTSIZE, color="navy")

    patch_linux = ax.patches[1]

    ax.text(
        patch_linux.get_width()-22,
        patch_linux.get_y()+patch_linux.get_height() / 2.0,
        f"{patch_linux.get_width():.0f} MiB",
        ha="center",
        va="center",
        fontsize=5,
    )

    patch_lros = ax.patches[0]

    ax.text(
        patch_lros.get_width()-28,
        patch_lros.get_y()+patch_lros.get_height() / 2.0,
        f"{patch_lros.get_width():.0f} MiB",
        ha="center",
        va="center",
        fontsize=5,
        )

    ax.set_ylabel(None)

    #plt.grid()
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "mem_usage.pdf"), format="pdf")

if __name__ == "__main__":
    main()
