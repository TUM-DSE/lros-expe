#!/usr/bin/env python3
"""Figure for the "Unikernel" evaluation subsection.

  unikernel-micro.pdf  the three microbenchmarks the methodology describes:
                       boot time split by phase, kernel image size and the
                       memory footprint touched during boot plus engine
                       startup.

Boot time and memory come from `just boottime` (bench/out/boottime/
boottime.csv, mean over runs), image sizes from `just imgsize`.
LROS has no separate user space, so its userspace segment is zero by
construction rather than missing data.
"""
from common import *

# bench/out and bench/plots, as swap.py names them (common.py does not).
result_dir = os.path.join(dir_path, "../../bench/out")
plots_dir = os.path.join(dir_path, "../../bench/plots")
OUT_DIR = parse_out_dir(default=plots_dir, description=__doc__)

SYSTEMS = ["Linux VM", "LROS"]
CONFIGS = ["linux", "lros"]  # CSV config names, in SYSTEMS order

runs = pd.read_csv(os.path.join(result_dir, "boottime", "boottime.csv"))
# an incomplete trace leaves every phase column empty; per-config phases the
# recipe never measures (linux firmware, lros init) become 0 after the mean
runs = runs.dropna(subset=["total_ms"])
mean = runs.groupby("config").mean(numeric_only=True).fillna(0)


def phase_s(col):
    return np.array([mean.at[c, col] for c in CONFIGS]) / 1e3


# ---- boot time [s]: everything up to the end of kernel init, then userspace -
# Hypervisor startup, firmware and kernel init are one segment: the split
# says little next to the userspace Linux needs and LROS does not.
BOOT = {
    "VM + kernel boot": phase_s("qemu_ms") + phase_s("firmware_ms") + phase_s("kernel_ms"),
    "Userspace":        phase_s("init_ms"),
}
# ---- kernel image size [MB] ------------------------------------------------
# `just imgsize`; only the kernels are compared. Linux's rootfs, the lros
# `image` row (loader.img, the fixed 64 MiB ESP padding mkuefi.sh wraps
# around the loader) and `bootloader` (the 7.7 KiB UEFI stub) stay in the
# CSV only.
img = pd.read_csv(os.path.join(result_dir, "boottime", "imgsize.csv"))
img = img.set_index(["config", "component"])["size_bytes"]

IMAGE = {
    "Kernel": np.array([img[(c, "kernel")] for c in CONFIGS]) / 2**20,
}
# ---- memory touched until the engine is up [MB] ----------------------------
# One figure per system, not split, so no legend.
MEMORY = {
    "Memory footprint": np.array([mean.at[c, "rss_kib"] for c in CONFIGS]) / 1024,
}

# Shade (how far lightened) and hatch per segment, over each system's colour:
# Linux in oranges, LROS in blues. The legend shows the shades in grey, as
# they stand for the segment in either system. Single-segment panels are
# solid.
SYSTEM_COLOR = [LINUX_COLOR, LROS_COLOR]  # in SYSTEMS order
LEGEND_COLOR = '#A6A6A6'
SEG_STYLE = {"VM + kernel boot": (0.0, ''), "Userspace": (0.6, '//'),
             "Memory footprint": (0.0, ''), "Kernel": (0.0, '')}


def amount(total, unit):
    """A total in its unit, switched rather than let .3g fall into
    scientific notation."""
    t, u = total, unit
    if unit == "MB" and total >= 1000:
        t, u = total / 1000, "GB"
    elif unit == "GB" and total < 0.1:
        t, u = total * 1000, "MB"
    return f"{t:.3g}\\,{u}" if USETEX else f"{t:.3g} {u}"


def row(ax, segments, title, unit, fewer="less", legend=True, hint=False):
    """One measure as horizontal stacked bars, Linux VM above LROS. Each
    total is written after its bar, LROS's with the factor it saves. Above
    the axes: (legend) the segments' legend in one line; hint puts the
    better-direction note inside the axes, bottom right. The title goes
    under the x tick labels, centred, as the panel's caption."""
    y = np.arange(len(SYSTEMS), dtype=float)[::-1]
    left = np.zeros(len(SYSTEMS))
    handles = []
    for name, vals in segments.items():
        shade, hatch = SEG_STYLE[name]
        ax.barh(y, vals, left=left, height=0.62, color=[lighten(c, shade) for c in SYSTEM_COLOR],
                hatch=hatch, edgecolor='black', linewidth=0.4, zorder=3)
        handles.append(mpatches.Patch(facecolor=lighten(LEGEND_COLOR, shade), hatch=hatch,
                                      edgecolor='black', linewidth=0.4, label=name))
        left = left + vals
    totals = left

    pad = totals.max() * 0.015
    for yi, total in zip(y, totals):
        text = amount(total, unit)
        if total < totals.max():
            text += " (" + speedup_label(totals.max() / total) + f" {fewer})"
        ax.text(total + pad, yi, text, ha='left', va='center', fontsize=FONTSIZE_ANNOTATION)

    ax.set_yticks(y)
    ax.set_yticklabels(SYSTEMS, fontsize=FONTSIZE_TICK_LABEL)
    ax.set_ylim(-0.55, len(SYSTEMS) - 0.45)
    # Room after the longer bar for its total.
    ax.set_xlim(0, totals.max() * 1.18)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    ax.tick_params(axis='y', length=0)

    # The legend sits right on the axes.
    if legend:
        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(segments),
                  fontsize=FONTSIZE_LEGEND, frameon=False, borderaxespad=0.1, borderpad=0.1,
                  handlelength=1.3, handleheight=0.8, handletextpad=0.4, columnspacing=1.0)
    if hint:
        # Inside, bottom right: LROS's bar and its total end well short of it.
        ax.annotate(left_better_str, xy=(1, 0), xycoords='axes fraction', xytext=(-2, 2),
                    textcoords='offset points', ha='right', va='bottom', color='blue',
                    fontsize=FONTSIZE_ANNOTATION)
    panel_caption(ax, title, y=-TICKS_H / AXES_H)


PANELS = [  # segments, title, unit, fewer, legend
    (BOOT, "(a) Boot time (s)", "s", "less", True),
    (MEMORY, "(b) Memory touched (MB)", "MB", "less", False),
    (IMAGE, "(c) Kernel image size (MB)", "MB", "smaller", False),
]
# Laid out by hand, in inches, so that the gap between one panel's caption
# and the next panel is the same whether or not a legend sits in between
# (tight_layout spaces the axes evenly instead).
AXES_H, TICKS_H, CAPTION_H, GAP_H, TOP_H, LEGEND_H = 0.42, 0.16, 0.14, 0.08, 0.04, 0.17
LEFT_W, RIGHT_W = 0.62, 0.05
height = sum(TOP_H + LEGEND_H * legend + AXES_H + TICKS_H + CAPTION_H for *_, legend in PANELS) \
    + GAP_H * (len(PANELS) - 1)
fig = plt.figure(figsize=(figwidth_half, height))
top = height
for segments, title, unit, fewer, legend in PANELS:
    top -= TOP_H + LEGEND_H * legend
    ax = fig.add_axes([LEFT_W / figwidth_half, (top - AXES_H) / height,
                       (figwidth_half - LEFT_W - RIGHT_W) / figwidth_half, AXES_H / height])
    row(ax, segments, title, unit, fewer=fewer, legend=legend, hint=True)
    top -= AXES_H + TICKS_H + CAPTION_H + GAP_H
save(fig, "unikernel-micro.pdf", OUT_DIR, png=True)
