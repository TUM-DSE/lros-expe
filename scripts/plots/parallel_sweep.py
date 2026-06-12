#!/usr/bin/env python3
"""
Box plot of per-request throughput vs. #apps (n_parallel),
with the mean throughput overlaid as a line.

Usage:
    python3 parallel_sweep.py <sweep_csv> [out.pdf]

CSV format (produced by bench.just bench_parallel_sweep):
    n_parallel,seq_id,speed_t_per_s
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import *


def load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def plot(df: pd.DataFrame, out_path: str | None = None) -> None:
    n_pars = sorted(df["n_parallel"].unique())

    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))

    sns.boxplot(
        data=df,
        x="n_parallel",
        y="speed_t_per_s",
        hue="n_parallel",
        order=n_pars,
        palette="pastel",
        width=0.5,
        flierprops=dict(marker=".", markersize=3),
        legend=False,
        ax=ax,
    )

    means = df.groupby("n_parallel")["speed_t_per_s"].mean().reindex(n_pars)
    positions = list(range(len(n_pars)))
    ax.plot(
        positions,
        means.values,
        "o-",
        color=darken(palette[1]),
        linewidth=1.5,
        markersize=4,
        label="mean",
        zorder=5,
    )

    ax.set_xlabel("# apps (n_parallel)")
    ax.set_ylabel("Throughput (t/s per request)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(n_pars)))
    ax.set_xticklabels(n_pars)
    ax.legend(frameon=False)
    ax.annotate(higher_better_str, xy=(1, 1), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=FONTSIZE - 1, style="italic")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    df = load(sys.argv[1])
    plot(df, sys.argv[2] if len(sys.argv) > 2 else None)
