#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common import *

def load_data() -> pd.DataFrame:
    df = pd.read_csv(result_dir+'/micro_pow.csv')
    return df


def main():
    data = load_data()
    #print(data)

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
    plot = sns.lineplot(ax=ax, data=data, x = "time", y = "power", 
                        #marker="H"
        )
    ax.annotate("First request", xy=(0,3), xytext=(0, 6), arrowprops=dict(facecolor='grey', shrink=0.05, width=2, headwidth=7))
    ax.annotate("Second request", xy=(5,5), xytext=(2, 7), arrowprops=dict(facecolor='grey', shrink=0.05, width=2, headwidth=7))
    ax.annotate("Thermal throttling", xy=(7,8), xytext=(8, 6), arrowprops=dict(facecolor='grey', shrink=0.05, width=2, headwidth=7))
    #plot.set_xticks([0, 100, 200, 300])
    ax.set_xticklabels(ax.get_xticks(), fontsize=FONTSIZE)
    ax.set_yticklabels(ax.get_yticks(), fontsize=FONTSIZE)
    #plot.set_yticks([0, 50_000, 100_000])
    #plot.set_yticklabels([0, 50, 100], fontsize=FONTSIZE)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time (s)")
    #ax.legend(loc="lower right", title=None,fontsize=FONTSIZE
    #    bbox_to_anchor=(0.5, -0.15),
    #    ncol=2,
    #)
    #ax.set_title(higher_better_str, fontsize=FONTSIZE, color="navy")
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(result_dir, "micro_pow.pdf"), format="pdf", pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    main()
