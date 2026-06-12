#!/usr/bin/env python3
"""
Maximum #apps as a function of memory budget for Llama-3.2-1B-Instruct-Q8_0.

KV cache parameters (from GGUF metadata):
  n_layer=16, n_head_kv=8, head_dim=64, dtype=f16 (2 bytes)
  KV per token = 16 × 8 × (64 + 64) × 2 = 32,768 bytes = 32 KB/token

Fixed memory overhead (measured):
  model weights : 1,252 MiB  (Q8_0, 1.22 GiB)
  compute buffer:   280 MiB  (batch_size=2048, single thread)
  output buffer :     2 MiB
  total fixed   : 1,534 MiB

With preemption the scheduler keeps 2N+1 KV sequences (1 system + N active + N parked):
  n_ctx   = (2N+1) × tokens_per_seq
  KV_mem  = n_ctx × 32 KB
  N_max   = ⌊( (budget_MiB - 1534) × 1024² / (tokens_per_seq × 32768) − 1 ) / 2⌋

Usage:
    python3 kv_capacity.py [out.pdf]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common import *

FIXED_MIB = 1534
KV_BYTES_PER_TOKEN = 16 * 8 * (64 + 64) * 2  # 32,768


def max_apps(budget_mib: float, tokens_per_seq: int) -> int:
    available = (budget_mib - FIXED_MIB) * 1024 * 1024  # bytes
    if available <= 0:
        return 0
    return max(0, int((available / (tokens_per_seq * KV_BYTES_PER_TOKEN) - 1) / 2))


def plot(out_path: str | None = None) -> None:
    budgets_gb = np.array([2, 4, 6, 8, 12, 16])
    budgets_mib = budgets_gb * 1024

    token_configs = [
        (256,  "256 tok/seq"),
        (512,  "512 tok/seq"),
        (1024, "1k tok/seq"),
        (2048, "2k tok/seq"),
    ]

    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))

    for (tokens, label), color, marker in zip(token_configs, palette, marker_def):
        ns = [max_apps(b, tokens) for b in budgets_mib]
        ax.plot(budgets_gb, ns, marker=marker, color=color,
                linewidth=1.5, markersize=5, label=label)

    ax.set_xlabel("Memory budget (GB)")
    ax.set_ylabel("Max # apps")
    ax.set_xticks(budgets_gb)
    ax.legend(title="context length", frameon=False)
    ax.annotate(higher_better_str, xy=(1, 1), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=FONTSIZE - 1, style="italic")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else None)
