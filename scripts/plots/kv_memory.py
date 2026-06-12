#!/usr/bin/env python3
"""
Stacked bar chart of measured memory components vs. #apps (n_parallel).

Measured components come from llama.cpp's own allocator output:
  model_mib   — model weights, constant across runs
  compute_mib — compute buffer, grows with n_parallel from np≈4 onward
  output_mib  — output logit buffer, grows linearly but small
  kv_mib      — KV cache, scales exactly with n_ctx = (2·np+1)·tokens_per_seq

Usage:
    python3 kv_memory.py <sweep_csv> [out.pdf]

CSV format (produced by bench.just bench_parallel_sweep):
    n_parallel,n_ctx,run_id,seq_id,latency_s,speed_t_per_s,model_mib,kv_mib,compute_mib,output_mib
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from common import *

MEM_COLS = ["model_mib", "kv_mib", "compute_mib", "output_mib"]


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Average memory values across all runs/requests for each n_parallel.
    mem = df.groupby("n_parallel")[["n_ctx"] + MEM_COLS].mean().reset_index()
    return mem.sort_values("n_parallel")


def plot(df: pd.DataFrame, out_path: str | None = None) -> None:
    np_vals = df["n_parallel"].values
    x = np.arange(len(np_vals))  # bar positions

    c_model   = palette[0]
    c_compute = palette[1]
    c_output  = palette[2]
    c_kv      = palette[3]
    bar_w = 0.6

    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))

    bottom = np.zeros(len(df))
    for col, color, label in [
        ("model_mib",   c_model,   "Model weights"),
        ("compute_mib", c_compute, "Compute buffer"),
        ("output_mib",  c_output,  "Output buffer"),
        ("kv_mib",      c_kv,      "KV cache"),
    ]:
        vals = df[col].values
        ax.bar(x, vals, bottom=bottom, width=bar_w,
               color=color, edgecolor=darken(color), linewidth=0.5, label=label)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(np_vals)
    ax.set_xlabel("# apps (n_parallel)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(frameon=False, fontsize=FONTSIZE - 1, ncol=2)

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
