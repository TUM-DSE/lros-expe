#!/usr/bin/env python3
"""
Bar plot of KV-cache context-switch latency by switch type.

Three switch types measured in parallel.cpp:
  complete  – slot reset when a request finishes (seq_rm + seq_cp from system)
  preempt   – state saved to parking slot (seq_cp to park + seq_rm + seq_cp from system)
  resume    – state restored from parking slot (seq_rm + seq_cp from park + seq_rm park)

Preempt and resume are expected to be ~2× slower than complete because they
copy the full KV history to/from the parking sequence before clearing the slot.

Usage:
    python3 ctx_switch.py <ctx_switch_csv> [out.pdf]

CSV format (produced by bench.just bench_ctx_switch):
    type,n_tokens,elapsed_us
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from common import *

TYPE_ORDER  = ["complete", "preempt", "resume"]
TYPE_LABELS = ["Complete", "Preempt", "Resume"]
TYPE_COLORS = [palette[0], palette[3], palette[2]]


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["type"] = df["type"].str.strip()
    return df


def plot(df: pd.DataFrame, out_path: str | None = None) -> None:
    if df.empty or df["elapsed_us"].eq(0).all():
        print("No CTX_SWITCH data found — rebuild the binary and re-run bench_ctx_switch.", file=sys.stderr)
        sys.exit(1)

    # Keep only types that actually appear in the data
    present = [t for t in TYPE_ORDER if t in df["type"].values]
    labels  = [TYPE_LABELS[TYPE_ORDER.index(t)] for t in present]
    colors  = [TYPE_COLORS[TYPE_ORDER.index(t)] for t in present]

    stats = (
        df[df["type"].isin(present)]
        .groupby("type")["elapsed_us"]
        .agg(mean="mean", std="std", count="count")
        .reindex(present)
    )

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))

    x = np.arange(len(present))
    bars = ax.bar(
        x,
        stats["mean"],
        yerr=stats["std"],
        capsize=3,
        color=colors,
        edgecolor=[darken(c) for c in colors],
        linewidth=0.8,
        error_kw=dict(elinewidth=0.8, ecolor="dimgray"),
        width=0.55,
    )

    # Overlay individual measurements as a strip
    for i, t in enumerate(present):
        vals = df.loc[df["type"] == t, "elapsed_us"].values
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=8, color=darken(colors[i]),
                   alpha=0.6, zorder=5, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (µs)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.annotate(lower_better_str, xy=(1, 1), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=FONTSIZE - 1, style="italic")

    # Annotate bar tops with mean ± std
    #for bar, (_, row) in zip(bars, stats.iterrows()):
    #    ax.text(
    #        bar.get_x() + bar.get_width() / 2,
    #        bar.get_height() + row["std"] + ax.get_ylim()[1] * 0.02,
    #        f"{row['mean']:.0f}±{row['std']:.0f}",
    #        ha="center", va="bottom", fontsize=FONTSIZE - 1,
    #    )

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
