#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common import *

def load_data() -> pd.DataFrame:
    df = pd.read_csv(result_dir+'/micro_pow_delay.csv')
    return df


def main():
    data = load_data()
    #print(data)

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
    plot = sns.lineplot(ax=ax, data=data, x = "time", y = "power", hue="system", style="system" 
                        #marker="H"
        )
    ax.annotate(text="Prefill", xy=(1, 4), size=FONTSIZE)
    ax.annotate(text="Decode", xy=(4, 4), size=FONTSIZE)
    ax.set_xticklabels(ax.get_xticks(), fontsize=FONTSIZE)
    ax.set_yticklabels(ax.get_yticks(), fontsize=FONTSIZE)
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", title=None,fontsize=FONTSIZE
    #    bbox_to_anchor=(0.5, -0.15),
    #    ncol=2,
    )
    #ax.set_title(higher_better_str, fontsize=FONTSIZE, color="navy")
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(result_dir, "micro_pow_delay.pdf"), format="pdf", pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    main()
