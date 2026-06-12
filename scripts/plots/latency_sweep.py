#!/usr/bin/env python3
"""
Box plot of per-request end-to-end latency vs. #apps (n_parallel),
with the mean latency overlaid as a line.

Latency is the wall-clock time from request arrival to completion
("total X.XX s" field in the DONE log line), including any waiting
time while higher-priority requests occupied the slot.

Usage:
    python3 latency_sweep.py <sweep_csv> [out.pdf]

CSV format (produced by bench.just bench_parallel_sweep):
    n_parallel,n_ctx,run_id,seq_id,latency_s,speed_t_per_s,...
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
        y="latency_s",
        hue="n_parallel",
        order=n_pars,
        palette="pastel",
        width=0.5,
        flierprops=dict(marker=".", markersize=3),
        legend=False,
        ax=ax,
    )

    means = df.groupby("n_parallel")["latency_s"].mean().reindex(n_pars)
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

    ax.set_xticks(range(len(n_pars)))
    ax.set_xticklabels(n_pars)
    ax.set_xlabel("# apps (n_parallel)")
    ax.set_ylabel("Latency (s)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.annotate(lower_better_str, xy=(1, 1), xycoords="axes fraction",
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
