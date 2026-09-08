#!/usr/bin/env python3
"""Mock figures for the "Memory management" evaluation subsection.

  mem-weights-tp.pdf  decode throughput as the VM's memory budget shrinks
                      below the model size, for the three loading strategies.
  mem-kvcache.pdf     cost of getting a suspended request's KV cache back:
                      recompute vs. swap to disk vs. the LROS policy.
"""
from common import *

OUT_DIR = parse_out_dir(description=__doc__)

RNG = np.random.default_rng(31415)

# ------------------------- weight paging under pressure --------------------
# x axis: VM memory budget as a multiple of the model's weight footprint.
BUDGET = np.array([0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5])
FITS = BUDGET >= 1.0

# mmap falls off a cliff once the working set no longer fits: every decode
# iteration sweeps all weights, so it faults them back in one page at a time.
mmap_tp = np.array([0.08, 0.10, 0.14, 0.51, 5.90, 5.92, 5.95])
# read() cannot page out at all, so it only exists for budgets that fit
read_tp = np.where(FITS, 6.10, np.nan)
# the prefetcher overlaps the next layer's I/O with the current computation
prefetch_tp = np.array([3.55, 3.90, 4.59, 5.31, 6.00, 6.02, 6.04])

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
ax.axvspan(BUDGET[0] - 0.05, 1.0, color="0.90", zorder=0)
ax.text(0.22, 0.94, "model does not fit", transform=ax.transAxes, ha="center",
        va="top", fontsize=FONTSIZE_ANNOTATION, color="0.35")

for key, values in (("mmap", mmap_tp), ("read", read_tp), ("prefetch", prefetch_tp)):
    st = style_for(key)
    ax.plot(BUDGET, values, color=st["color"], marker=st["marker"], markersize=3,
            markeredgecolor="black", markeredgewidth=0.3, linewidth=1.1,
            label=st["label"], zorder=3)

# the headline number from the text lives at the deepest memory pressure point
k = 2
ax.annotate("", xy=(BUDGET[k], prefetch_tp[k]), xytext=(BUDGET[k], mmap_tp[k]),
            arrowprops=dict(arrowstyle="-|>", color="#8B0000", lw=0.8, shrinkB=0))
ax.text(BUDGET[k] + 0.03, np.sqrt(prefetch_tp[k] * mmap_tp[k]),
        speedup_label(prefetch_tp[k] / mmap_tp[k]), fontsize=FONTSIZE_ANNOTATION,
        ha="left", va="center")

ax.set_yscale("log")
ax.set_ylim(0.05, 14)
ax.set_xlim(BUDGET[0] - 0.05, BUDGET[-1] + 0.05)
ax.set_xlabel("Memory budget / model size", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylabel("Throughput [tok/s]", fontsize=FONTSIZE_AXIS_LABEL)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.35, zorder=1)
ax.set_axisbelow(False)
thin_spines(ax)
better_hint(ax, higher=True, xy=(0.50, 0.06))
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.6,
          columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(pad=0.1)
save(fig, "mem-weights-tp.pdf", OUT_DIR)


# ------------------------------- KV cache ----------------------------------
# x axis: how many conversations share the memory budget. The more sessions,
# the more often an arriving request finds its KV cache evicted.
SESSIONS = [2, 4, 8, 16]
# recompute pays the full prefill again, and grows with the context length
recompute = np.array([0.42, 1.35, 3.10, 6.45])
# swapping is bounded by virtio-blk bandwidth, so it grows with cache size only
swap = np.array([0.31, 0.78, 1.72, 3.55])
# the policy recomputes short prefixes and reloads long ones, whichever is cheaper
policy = np.array([0.24, 0.51, 0.96, 1.78])

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
values = {"recompute": recompute, "swap": swap, "policy": policy}
errors = {k: v * 0.07 for k, v in values.items()}
grouped_bars(ax, [str(s) for s in SESSIONS], ["recompute", "swap", "policy"],
             values, errors=errors, baseline="recompute", annotate_series=["policy"])
ax.set_xlabel("Concurrent sessions", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylabel("Cache restore cost [s]", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylim(0, recompute.max() * 1.32)
better_hint(ax)
ax.legend(*legend_handles(["recompute", "swap", "policy"]), loc="lower center",
          bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=FONTSIZE_LEGEND,
          frameon=False, handlelength=1.4, handleheight=0.8, columnspacing=1.2,
          handletextpad=0.4)
fig.tight_layout(pad=0.1)
save(fig, "mem-kvcache.pdf", OUT_DIR)
