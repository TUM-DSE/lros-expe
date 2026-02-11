#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import matplotlib

from common import *

def load_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(result_dir,"boottime", "img.csv"))
    return df


def main():
    data = load_data()
    #print(data)

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
    data_melt = data.melt(var_name="System", value_name="Img Size").head(2)
    data_melt["imgMB"] = data_melt["Img Size"]/1024
    plot = sns.barplot(ax=ax, data=data_melt, y ="System", x ="imgMB", hue="System"
                        , edgecolor="black", linewidth=0.5
                        , palette=palette
                       #marker="H"
                       )
    #plot.set_xticks([0, 100, 200, 300])
    # ax.set_xticklabels(data['setup'].unique(), fontsize=FONTSIZE)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=FONTSIZE)
    # plot.set_yticks([0, 100, 200, 300])
    #plot.set_yticklabels([0, 50, 100], fontsize=FONTSIZE)
    ax.set_xlabel("Image size (MiB)")
    # ax.set_xlabel("")
    # ax.legend(loc="upper left", title=None,fontsize=FONTSIZE
    # #    bbox_to_anchor=(0.5, -0.15),
    # #    ncol=2,
    # )
    ax.set_title(left_better_str, fontsize=FONTSIZE, color="navy")

    patch_linux = ax.patches[1]

    """ax.text(
        patch_linux.get_x() + patch_linux.get_width() / 2.0,
        patch_linux.get_height()+10,
        f"{patch_linux.get_height():.0f} MiB",
        ha="center",
        va="bottom",
        fontsize=5,
    )"""
    ax.text(
        patch_linux.get_width()+75,
        patch_linux.get_y()+patch_linux.get_height() / 2.0,
        f"{patch_linux.get_width():.0f} MiB",
        ha="center",
        va="center",
        fontsize=5,
        )


    patch_lros = ax.patches[0]

    ax.text(
            patch_lros.get_width()-110,
            patch_lros.get_y()+patch_lros.get_height() / 2.0,
            f"{patch_lros.get_width():.0f} MiB",
            ha="center",
            va="center",
            fontsize=5,
        )

    ax.set_ylabel(None)
    #ax.set_xlim(0,1600)

    plot.get_legend().remove()
    #plt.grid()
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(plots_dir, "img_size.pdf"), format="pdf")

if __name__ == "__main__":
    main()
