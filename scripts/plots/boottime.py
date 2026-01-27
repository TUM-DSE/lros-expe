#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import matplotlib

from common import *

def load_data() -> pd.DataFrame:
    dfLinux = pd.read_csv(os.path.join(result_dir,"boottime", "linux.csv"))
    dfLros = pd.read_csv(os.path.join(result_dir,"boottime", "lros.csv"))
    concat = pd.concat([dfLros, dfLinux], keys=["LROS", "Linux"], axis=0)
    concat=concat.sub(concat.iloc[:, 0], axis=0).div(1000*1000*1000)
    return concat.reset_index()


def main():
    data = load_data()
    #print(data)

    # matplotlib.use('TkAgg')

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))

    plot = sns.barplot(ax=ax, data=data, x="level_0", y="QEMU: kvm_cpu_exec",
                       color=palette[0]  # , style = "level_0"
                       , edgecolor="black", linewidth=0.5
                       , errorbar=None
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    # for x in plot.patches:
    #     if x.get_hatch() is None:
    #         x.set_hatch(hatch_def[0])
    #         x.set_label("_")

    plot = sns.barplot(ax=ax, data=data, x="level_0", y="231 Linux: init_start",
                       color=palette[1]  # , style = "level_0"
                       , edgecolor="black", linewidth=0.5
                       , zorder=-1
                       , errorbar=None
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    # for x in plot.patches:
    #     if x.get_hatch() is None:
    #         x.set_hatch(hatch_def[1])
    #         x.set_label("_")

    plot = sns.barplot(ax=ax, data=data, x="level_0", y="240 Linux: systemd init end",
                       color=palette[2]  # , style = "level_0""
                       , edgecolor="black", linewidth=0.5
                       , zorder=-2
                       , errorbar=None
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )


    # for x in plot.patches:
    #     if x.get_hatch() is None:
    #         x.set_hatch(hatch_def[2])
    #         x.set_label("_")

    plot = sns.barplot(ax=ax, data=data, x="level_0", y="250 Unikraft: init end",
                       color=palette[1]  # , style = "level_0"
                       , edgecolor="black", linewidth=0.5
                       , zorder=-1
                       , errorbar=None
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    # for x in plot.patches:
    #     if x.get_hatch() is None:
    #         x.set_hatch(hatch_def[1])
    #         x.set_label("_")
    # l1 = ax.legend(fontsize=FONTSIZE, loc='upper left')

    patch_linux = ax.patches[2]

    ax.text(
        patch_linux.get_x() + patch_linux.get_width() / 2.0,
        patch_linux.get_height()+0.1,
        f"{patch_linux.get_height():.2f} s",
        ha="center",
        va="bottom",
        fontsize=5,
    )

    patch_linux = ax.patches[3]

    ax.text(
        patch_linux.get_x() + patch_linux.get_width() / 2.0,
        patch_linux.get_height()+0.1,
        f"{patch_linux.get_height():.2f} s",
        ha="center",
        va="bottom",
        fontsize=5,
    )

    patch_lros = ax.patches[4]

    ax.text(
        patch_lros.get_x() + patch_lros.get_width() / 2.0,
        patch_lros.get_height()+0.1,
        f"{patch_lros.get_height():.2f} s",
        ha="center",
        va="bottom",
        fontsize=5,
    )



# for container in ax.containers:
    #     ax.bar_label(container, fmt="%d", padding=0)

    # Color legend
    hidden = [ax.bar(0, 0, color=palette[0], label="QEMU"),
              ax.bar(0, 0, color=palette[1], label="Kernel"),
              ax.bar(0, 0, color=palette[2], label="Init"),
              #ax.bar(0, 0, color="gray", hatch=hatch_def[2], label="TextGen")
              ]

    plot.legend(handles=hidden, loc='upper left', fontsize=FONTSIZE)
    # ax.add_artist(l1)

    ax.set_xlabel(None)
    ax.set_ylabel("Time (s)")
    ax.set_ylim(0,11.5)
    ax.set_title(f"(a) {lower_better_str}", fontsize=FONTSIZE, color="navy")
    plt.savefig(os.path.join(plots_dir, "boottime.pdf"), format="pdf", pad_inches=0, bbox_inches="tight")

    # plt.show()

if __name__ == "__main__":
    main()
