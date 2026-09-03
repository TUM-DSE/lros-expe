#!/usr/bin/env python3
"""Mock figures for the "Scheduler" evaluation subsection.

  sched-strategies.pdf  TTFT and aggregate throughput of the execution
                        strategies as simultaneously arriving requests pile up.
  sched-itl-cdf.pdf     inter-token latency seen inside a single request.
"""
from common import *

OUT_DIR = parse_out_dir(description=__doc__)

RNG = np.random.default_rng(2718)

STRATEGIES = ["batched", "concurrent", "hybrid", "adaptive"]
CONCURRENCY = [1, 2, 4, 8]

# TTFT [s]: batching makes every request wait for the slowest one in the batch;
# running requests concurrently on separate units keeps latency low but wastes
# capacity; LROS batches only while the queue is short.
TTFT = {
    "batched":    np.array([3.1, 4.4, 7.2, 13.6]),
    "concurrent": np.array([3.0, 3.4, 4.6, 8.9]),
    "hybrid":     np.array([3.0, 3.2, 4.1, 7.1]),
    "adaptive":   np.array([2.9, 3.1, 3.5, 4.4]),
}
# Aggregate decode throughput [tok/s]: the mirror image -- batching wins at
# high load, which is exactly the tradeoff the subsection is about.
THROUGHPUT = {
    "batched":    np.array([18.5, 31.0, 48.5, 66.0]),
    "concurrent": np.array([18.3, 25.4, 34.0, 41.5]),
    "hybrid":     np.array([19.8, 29.1, 41.7, 52.3]),
    "adaptive":   np.array([19.6, 32.4, 50.1, 64.2]),
}

fig, axes = plt.subplots(1, 2, figsize=(figwidth_full, fig_height))

grouped_bars(axes[0], [str(c) for c in CONCURRENCY], STRATEGIES, TTFT,
             errors={k: v * 0.06 for k, v in TTFT.items()}, baseline="batched",
             annotate_series=["adaptive"])
axes[0].set_xlabel("Simultaneous requests", fontsize=FONTSIZE_AXIS_LABEL)
axes[0].set_ylabel("TTFT [s]", fontsize=FONTSIZE_AXIS_LABEL)
axes[0].set_ylim(0, max(v.max() for v in TTFT.values()) * 1.28)
better_hint(axes[0])
panel_caption(axes[0], "(a) Request latency", y=-0.32)

grouped_bars(axes[1], [str(c) for c in CONCURRENCY], STRATEGIES, THROUGHPUT,
             errors={k: v * 0.05 for k, v in THROUGHPUT.items()}, baseline="batched",
             annotate_series=["adaptive"])
axes[1].set_xlabel("Simultaneous requests", fontsize=FONTSIZE_AXIS_LABEL)
axes[1].set_ylabel("Throughput [tok/s]", fontsize=FONTSIZE_AXIS_LABEL)
axes[1].set_ylim(0, max(v.max() for v in THROUGHPUT.values()) * 1.28)
better_hint(axes[1], higher=True)
panel_caption(axes[1], "(b) Aggregate throughput", y=-0.32)

fig.legend(*legend_handles(STRATEGIES), loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=4, fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.4,
           handleheight=0.8, columnspacing=1.6, handletextpad=0.4)
fig.tight_layout(pad=0.2, w_pad=1.5)
save(fig, "sched-strategies.pdf", OUT_DIR)


# ---------------------------- inter-token latency --------------------------
# At 8 simultaneous requests: sharded batching stalls a request whenever the
# batch is re-formed, which shows up as a long tail rather than a shifted mean.
SHAPE = {
    "batched":    (0.150, 0.55, 0.30),   # (median, lognormal sigma, tail weight)
    "concurrent": (0.205, 0.22, 0.05),
    "hybrid":     (0.178, 0.30, 0.10),
    "adaptive":   (0.158, 0.20, 0.04),
}

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
for key in STRATEGIES:
    median, sigma, tail = SHAPE[key]
    body = median * RNG.lognormal(0.0, sigma, size=4000)
    stalls = RNG.random(4000) < tail
    body[stalls] *= RNG.uniform(2.0, 9.0, size=stalls.sum())
    st = style_for(key)
    xs, ys = cdf(body)
    ax.plot(xs, ys, color=st["color"], linewidth=1.1, label=st["label"], zorder=3)

ax.set_xscale("log")
ax.set_xlim(0.06, 3.0)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Inter-token latency [s]", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylabel("CDF", fontsize=FONTSIZE_AXIS_LABEL)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
thin_spines(ax)
ax.annotate(left_better_str, color="blue", xy=(0.02, 0.90),
            xycoords="axes fraction", fontsize=FONTSIZE_ANNOTATION)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=FONTSIZE_LEGEND,
          frameon=False, handlelength=1.6, handletextpad=0.4, labelspacing=0.25)
fig.tight_layout(pad=0.1)
save(fig, "sched-itl-cdf.pdf", OUT_DIR)
