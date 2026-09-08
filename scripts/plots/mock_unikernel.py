#!/usr/bin/env python3
"""Mock figure for the "Unikernel" evaluation subsection.

  unikernel-micro.pdf  the three microbenchmarks the methodology describes:
                       boot time split by phase, image size and the memory
                       footprint touched during boot plus engine startup.

LROS has no separate user space, so the corresponding segments are zero by
construction rather than missing data.
"""
from common import *

OUT_DIR = parse_out_dir(description=__doc__)

SYSTEMS = ["Linux VM", "LROS"]

# ---- boot time [s], split into the phases the methodology traces -----------
BOOT = {
    "Hypervisor (QEMU)": np.array([0.121, 0.088]),
    "Kernel init":       np.array([0.802, 0.082]),
    "Userspace (init)":  np.array([9.611, 0.000]),
}
# ---- image size [MB]: Linux needs a root filesystem next to the kernel -----
IMAGE = {
    "Kernel / unikernel": np.array([11.4, 6.8]),
    "Root filesystem":    np.array([248.0, 0.0]),
}
# ---- memory touched until the engine is up [MB] ----------------------------
MEMORY = {
    "Memory footprint": np.array([174.0, 21.0]),
}

SEG_COLORS = [LROS_COLOR, lighten(LROS_COLOR, 0.40), lighten(LROS_COLOR, 0.72)]
SEG_HATCHES = ['', 'xx', '//']


def stacked(ax, segments, ylabel, unit, log=False):
    x = np.arange(len(SYSTEMS), dtype=float)
    bottom = np.zeros(len(SYSTEMS))
    for i, (name, vals) in enumerate(segments.items()):
        ax.bar(x, vals, bottom=bottom, width=0.55, label=name,
               color=SEG_COLORS[i], edgecolor='black', linewidth=0.4,
               hatch=SEG_HATCHES[i], zorder=3)
        bottom = bottom + vals
    totals = bottom

    for xi, total in zip(x, totals):
        ax.text(xi, total * 1.06, f"{total:.3g}\\,{unit}" if USETEX else f"{total:.3g} {unit}",
                ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION)

    # the whole point of the section is the ratio between the two bars; the
    # arrow sits beside the LROS bar so it does not cover its total label
    ratio = totals[0] / totals[1]
    arrow_x = x[1] + 0.36
    ax.annotate("", xy=(arrow_x, totals[1]), xytext=(arrow_x, totals[0]),
                arrowprops=dict(arrowstyle="-|>", color="#8B0000", lw=0.8, shrinkB=0))
    ax.text(arrow_x + 0.06, np.sqrt(totals[0] * totals[1]) if log else (totals[0] + totals[1]) / 2,
            speedup_label(ratio) + (r"$\,\downarrow$" if USETEX else " ↓"),
            fontsize=FONTSIZE_ANNOTATION, ha='left', va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(SYSTEMS, fontsize=FONTSIZE_TICK_LABEL)
    ax.set_xlim(-0.6, len(SYSTEMS) + 0.15)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(totals.min() * 0.35, totals.max() * 6)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_ylim(0, totals.max() * 1.35)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    better_hint(ax)
    return totals


fig, axes = plt.subplots(1, 3, figsize=(figwidth_full, fig_height))

stacked(axes[0], BOOT, "Boot time [s]", "s", log=True)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
               fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.3,
               handleheight=0.8, handletextpad=0.4, labelspacing=0.2)
panel_caption(axes[0], "(a) Boot time", y=-0.20)

stacked(axes[1], IMAGE, "Image size [MB]", "MB")
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
               fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.3,
               handleheight=0.8, handletextpad=0.4, labelspacing=0.2)
panel_caption(axes[1], "(b) Image size", y=-0.20)

stacked(axes[2], MEMORY, "Memory touched [MB]", "MB")
axes[2].get_legend_handles_labels()
panel_caption(axes[2], "(c) Memory footprint", y=-0.20)

fig.tight_layout(pad=0.2, w_pad=1.5)
save(fig, "unikernel-micro.pdf", OUT_DIR)
