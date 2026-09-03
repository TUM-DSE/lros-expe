#!/usr/bin/env python3
"""Mock figures for the "Accelerators" evaluation subsection.

  accel-perf.pdf  TTFT and throughput per platform for CPU-only execution,
                  the VIAI (vAccel) path inside the VM, and native host
                  execution with the vendor backend.
  accel-loc.pdf   implementation effort of the VIAI plugins, in LoC.
"""
from common import *

OUT_DIR = parse_out_dir(description=__doc__)

MODELS = ["Llama3.1-1B", "Gemma3-1B", "Qwen2.5-1.5B"]
BACKENDS = ["cpu", "viai", "native"]

# (TTFT [s], decode throughput [tok/s]) for the CPU-only baseline, per platform.
CPU_BASE = {
    "OrangePi 5 Ultra": {"Llama3.1-1B": (14.8, 4.9), "Gemma3-1B": (11.4, 6.1),
                         "Qwen2.5-1.5B": (22.1, 3.5)},
    "Jetson AGX Orin":  {"Llama3.1-1B": (7.9, 9.4), "Gemma3-1B": (6.1, 11.7),
                         "Qwen2.5-1.5B": (11.8, 6.7)},
}
# speedup of each backend over CPU-only; VIAI keeps most of the native gain,
# the remainder is the tunnelling and data-transfer overhead.
GAIN = {
    "OrangePi 5 Ultra": {"viai": (2.6, 4.0), "native": (3.1, 4.7)},
    "Jetson AGX Orin":  {"viai": (3.4, 5.2), "native": (4.3, 6.9)},
}


# how well each model's operator mix maps onto the accelerator
MODEL_FIT = {"Llama3.1-1B": 1.00, "Gemma3-1B": 1.09, "Qwen2.5-1.5B": 0.93}


def panel_values(platform, metric):
    idx = 0 if metric == "ttft" else 1
    out = {}
    for backend in BACKENDS:
        vals = []
        for model in MODELS:
            base = CPU_BASE[platform][model][idx]
            if backend == "cpu":
                vals.append(base)
            else:
                g = GAIN[platform][backend][idx] * MODEL_FIT[model]
                vals.append(base / g if metric == "ttft" else base * g)
        out[backend] = vals
    return out


# ------------------------------ accel-perf ---------------------------------
fig, axes = plt.subplots(1, 4, figsize=(figwidth_full, fig_height * 0.95))
panels = [
    (axes[0], "OrangePi 5 Ultra", "ttft", "TTFT [s]", "(a) OrangePi: TTFT"),
    (axes[1], "OrangePi 5 Ultra", "tp", "Throughput [tok/s]", "(b) OrangePi: decode"),
    (axes[2], "Jetson AGX Orin", "ttft", "TTFT [s]", "(c) Jetson: TTFT"),
    (axes[3], "Jetson AGX Orin", "tp", "Throughput [tok/s]", "(d) Jetson: decode"),
]

SHORT = {"Llama3.1-1B": "Llama", "Gemma3-1B": "Gemma", "Qwen2.5-1.5B": "Qwen"}

for ax, platform, metric, ylabel, caption in panels:
    values = panel_values(platform, metric)
    # the message is the VIAI bar, so only that one carries a speedup label
    grouped_bars(ax, [SHORT[m] for m in MODELS], BACKENDS, values,
                 baseline="cpu", annotate_fmt="speedup", annotate_series=["viai"],
                 annotate_invert=(metric == "ttft"))
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylim(0, max(max(v) for v in values.values()) * 1.30)
    better_hint(ax, higher=(metric == "tp"))
    panel_caption(ax, caption, y=-0.18)

fig.legend(*legend_handles(BACKENDS), loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=3, fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.4,
           handleheight=0.8, columnspacing=1.6, handletextpad=0.4)
fig.tight_layout(pad=0.2, w_pad=1.0)
save(fig, "accel-perf.pdf", OUT_DIR)


# ------------------------------- accel-loc ---------------------------------
# Implementation effort: a VIAI plugin only has to bind the operators the
# inference engine actually issues, which is why it stays far below a full
# in-tree backend.
PLUGINS = [
    ("RKNN (OrangePi NPU)", 190, 610),
    ("CUDA (Jetson GPU)", 165, 740),
    ("CPU fallback (ggml)", 120, 130),
]
REFERENCE = ("llama.cpp in-tree CUDA backend", 46200)
SEGMENTS = [("Plugin glue", LROS_COLOR, ''), ("Operator kernels", lighten(LROS_COLOR, 0.45), 'xx')]

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height * 0.8))
y = np.arange(len(PLUGINS))[::-1].astype(float)
glue = np.array([p[1] for p in PLUGINS], dtype=float)
kernels = np.array([p[2] for p in PLUGINS], dtype=float)

ax.barh(y, glue, height=0.55, color=SEGMENTS[0][1], edgecolor='black',
        linewidth=0.4, hatch=SEGMENTS[0][2], label=SEGMENTS[0][0], zorder=3)
ax.barh(y, kernels, left=glue, height=0.55, color=SEGMENTS[1][1], edgecolor='black',
        linewidth=0.4, hatch=SEGMENTS[1][2], label=SEGMENTS[1][0], zorder=3)

for yi, (_, g, k) in zip(y, PLUGINS):
    ax.text(g + k + 20, yi, f"{int(g + k):,}", va="center", ha="left",
            fontsize=FONTSIZE_ANNOTATION)

ax.set_xlim(0, 1150)
ax.set_xticks([0, 250, 500, 750, 1000])
ax.set_ylim(-0.6, len(PLUGINS) - 0.1)
ax.set_yticks(y)
ax.set_yticklabels([p[0] for p in PLUGINS], fontsize=FONTSIZE_TICK_LABEL)
ax.set_xlabel("Lines of code", fontsize=FONTSIZE_AXIS_LABEL)
ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
thin_spines(ax)
ax.annotate(left_better_str, color="blue", xy=(0.98, 0.96),
            xycoords="axes fraction", ha="right", va="top",
            fontsize=FONTSIZE_ANNOTATION)

# A full in-tree backend is two orders of magnitude larger, so it is stated
# rather than plotted -- a bar for it would flatten the three plugins.
ax.annotate(f"{REFERENCE[0]}: {format_big_numbers(REFERENCE[1])} LoC",
            xy=(0.98, 0.06), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=FONTSIZE_ANNOTATION, color="0.30",
            bbox=dict(boxstyle="round,pad=0.25", fc="0.94", ec="0.75", lw=0.4))

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.4,
          handleheight=0.8, columnspacing=1.2, handletextpad=0.4)
fig.tight_layout(pad=0.1)
save(fig, "accel-loc.pdf", OUT_DIR)
