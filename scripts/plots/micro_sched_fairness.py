#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common import *

def load_data() -> pd.DataFrame:
    df = pd.read_csv(result_dir+'/micro_sched_fairness.csv')
    return df


def main():
    data = load_data()
    #print(data)

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
    plot = sns.boxplot(ax=ax, data=data, x = "setup", y = "latency", hue="task", 
                        #marker="H"
        )
    #plot.set_xticks([0, 100, 200, 300])
    ax.set_xticklabels(data['setup'].unique(), fontsize=FONTSIZE)
    ax.set_yticklabels(ax.get_yticks(), fontsize=FONTSIZE)
    #plot.set_yticks([0, 50_000, 100_000])
    #plot.set_yticklabels([0, 50, 100], fontsize=FONTSIZE)
    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", title=None,fontsize=FONTSIZE
    #    bbox_to_anchor=(0.5, -0.15),
    #    ncol=2,
    )
    ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(result_dir, "micro_sched_fairness.pdf"), format="pdf", pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    main()
