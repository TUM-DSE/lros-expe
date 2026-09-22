# Shared plotting style for the LROS paper: Libertine text, a colour and hatch
# per system, and a blue "lower/higher is better" hint. LaTeX renders the text
# when a latex binary exists (force with PLOT_USETEX=1/0), mathtext otherwise.
import argparse
import os
import shutil
import sys

# workaround to select Agg as backend consistenly
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb, to_hex
import matplotlib.patches as mpatches
import seaborn as sns  # type: ignore
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
paper_plots_dir = os.path.realpath(os.path.join(dir_path, "../../../paper/plots"))
mock_dir = os.path.join(paper_plots_dir, "mock")

# 3.3 inch for single column, 7 inch for double column
figwidth_third = 2
figwidth_half = 3.3
figwidth_full = 7
fig_height = 2
FONTSIZE = 7

# Derived font sizes for consistency
FONTSIZE_AXIS_LABEL = FONTSIZE
FONTSIZE_TICK_LABEL = FONTSIZE
FONTSIZE_LEGEND = FONTSIZE - 1
FONTSIZE_TITLE = FONTSIZE - 1
FONTSIZE_ANNOTATION = FONTSIZE - 2

palette = sns.color_palette("pastel")
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": FONTSIZE, "ytick.major.size": FONTSIZE})
sns.set_context("paper", rc={"font.size": FONTSIZE, "axes.titlesize": FONTSIZE, "axes.labelsize": FONTSIZE,
                             "xtick.labelsize": FONTSIZE, "ytick.labelsize": FONTSIZE,
                             "legend.fontsize": FONTSIZE, "legend.title_fontsize": FONTSIZE})

# must be done after sns styles, otherwise they force sans-serif
mpl.use("Agg")


def _want_usetex() -> bool:
    override = os.environ.get("PLOT_USETEX")
    if override is not None:
        return override not in ("0", "", "no", "false")
    return shutil.which("latex") is not None


USETEX = _want_usetex()

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
if USETEX:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Libertine"],
        "text.latex.preamble": r"""
\usepackage[tt=false, type1=true]{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{amsmath}
""",
    })
else:
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertine", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
    })


# --- text, valid in both pipelines ------------------------------------------

def bf(text: str) -> str:
    """Bold `text`; under usetex the markup has to be explicit."""
    return r"\textbf{%s}" % text if USETEX else text


def pct_label(value: float, signed: bool = True) -> str:
    """A percentage, set in math mode so it matches the paper's body font."""
    sign = ("+" if value >= 0 else "-") if signed else ""
    return r"$%s%.0f\%%$" % (sign, abs(value) if signed else value)


def speedup_label(factor: float, decimals: int = 1) -> str:
    return r"$%.*f\times$" % (decimals, factor)


def _arrow(name: str) -> str:
    if USETEX:
        return r"$\%s$" % name
    return {"downarrow": "↓", "uparrow": "↑",
            "leftarrow": "←", "rightarrow": "→"}[name]


lower_better_str = "Lower is better " + _arrow("downarrow")
higher_better_str = "Higher is better " + _arrow("uparrow")
left_better_str = "Lower is better " + _arrow("leftarrow")


# --- colours ------------------------------------------------------------------

def darken(color):
    hue, saturation, value = rgb_to_hsv(to_rgb(color))
    return hsv_to_rgb((hue, saturation, value * 0.9))


def lighten(color, factor=0.5):
    rgb = to_rgb(color)
    return to_hex([c + (1.0 - c) * factor for c in rgb])


hatch_def = [
    "//",
    '',
    'xx',
    '*',
    "--",
    "++",
    "||",
    "..",
    "oo",
    "\\\\",
]

# Qualitative base colours (ColorBrewer Set1/Paired), one hue per role.
LROS_COLOR = '#1F78B4'          # blue  -- our system, everywhere
LINUX_COLOR = '#9E9E9E'         # grey  -- unmodified llama.cpp on Linux
CPU_COLOR = '#984EA3'           # purple -- CPU-only execution
NATIVE_COLOR = '#4DAF4A'        # green -- native host / upper bound
ALT_COLOR = '#FF7F00'           # orange -- the other competitor in a panel
ALT2_COLOR = '#E7298A'          # magenta -- fourth competitor, rarely needed

LROS_HATCH = ''
LINUX_HATCH = '//'
CPU_HATCH = 'xx'
NATIVE_HATCH = '..'
ALT_HATCH = '\\\\'
ALT2_HATCH = '++'

# One entry per series the plots use: its colour, hatch, label and marker.
style_map = {
    # end-to-end comparison
    'linux':      {'color': LINUX_COLOR, 'hatch': LINUX_HATCH, 'label': 'Linux', 'marker': 'o'},
    'lros':       {'color': LROS_COLOR,  'hatch': LROS_HATCH,  'label': 'LROS',  'marker': 'D'},
    # accelerator integration
    'cpu':        {'color': CPU_COLOR,    'hatch': CPU_HATCH,    'label': 'CPU-only', 'marker': 'x'},
    'lros-cpu':   {'color': CPU_COLOR,    'hatch': CPU_HATCH,    'label': 'LROS (CPU)', 'marker': 'x'},
    'viai':       {'color': LROS_COLOR,   'hatch': LROS_HATCH,   'label': 'LROS (VIAI)', 'marker': 'D'},
    'native':     {'color': NATIVE_COLOR, 'hatch': NATIVE_HATCH, 'label': 'Native host', 'marker': '^'},
    # model loading / weight paging
    'mmap':       {'color': LINUX_COLOR, 'hatch': LINUX_HATCH, 'label': r'llama.cpp (mmap)', 'marker': 'o'},
    'read':       {'color': ALT_COLOR,   'hatch': ALT_HATCH,   'label': r'llama.cpp (read)', 'marker': 's'},
    'prefetch':   {'color': LROS_COLOR,  'hatch': LROS_HATCH,  'label': 'LROS (prefetcher)', 'marker': 'D'},
    # KV cache management
    'recompute':  {'color': CPU_COLOR, 'hatch': CPU_HATCH, 'label': 'Recompute', 'marker': 'x'},
    'swap':       {'color': ALT_COLOR, 'hatch': ALT_HATCH, 'label': 'Swap to disk', 'marker': 's'},
    'policy':     {'color': LROS_COLOR, 'hatch': LROS_HATCH, 'label': 'LROS (policy)', 'marker': 'D'},
    # scheduling strategies
    'batched':    {'color': LINUX_COLOR, 'hatch': LINUX_HATCH, 'label': 'Sharded batch', 'marker': 'o'},
    'concurrent': {'color': ALT_COLOR,   'hatch': ALT_HATCH,   'label': 'Concurrent', 'marker': 's'},
    'hybrid':     {'color': NATIVE_COLOR, 'hatch': NATIVE_HATCH, 'label': 'CPU+Acc. hybrid', 'marker': '^'},
    'adaptive':   {'color': LROS_COLOR,  'hatch': LROS_HATCH,  'label': 'LROS', 'marker': 'D'},
    # motivation: the two phases, wherever they are separated
    'prefill':    {'color': ALT_COLOR,   'hatch': ALT_HATCH,   'label': 'Prefill', 'marker': 's'},
    'decode':     {'color': LROS_COLOR,  'hatch': LROS_HATCH,  'label': 'Decode', 'marker': 'D'},
    'faults':     {'color': ALT2_COLOR,  'hatch': ALT2_HATCH,  'label': 'Major faults', 'marker': 'v'},
    # motivation: switch-or-wait
    'switch':     {'color': ALT_COLOR,   'hatch': ALT_HATCH,   'label': 'Switch now', 'marker': 's'},
    'wait':       {'color': LINUX_COLOR, 'hatch': LINUX_HATCH, 'label': 'Wait', 'marker': 'o'},
    # motivation: preemption granularity
    'engine':     {'color': LINUX_COLOR, 'hatch': LINUX_HATCH, 'label': 'llama-server', 'marker': 'o'},
    'fair':       {'color': ALT_COLOR,   'hatch': ALT_HATCH,   'label': 'llama-server + fair share', 'marker': 'D'},
    'separate':   {'color': CPU_COLOR,   'hatch': CPU_HATCH,   'label': 'Two processes (kernel)', 'marker': '^'},
    'stopped':    {'color': NATIVE_COLOR, 'hatch': NATIVE_HATCH, 'label': 'Kernel + priority', 'marker': 'v'},
    'prio':       {'color': ALT2_COLOR,  'hatch': ALT2_HATCH,  'label': 'Engine + priority', 'marker': 'P'},
    # motivation: paging and the one budget
    'madv-seq':   {'color': ALT_COLOR,   'hatch': ALT_HATCH,   'label': 'mmap + MADV_SEQUENTIAL', 'marker': 's'},
    'madv-rand':  {'color': ALT2_COLOR,  'hatch': ALT2_HATCH,  'label': 'mmap + MADV_RANDOM', 'marker': 'P'},
    'read-buf':   {'color': CPU_COLOR,   'hatch': CPU_HATCH,   'label': 'read into a buffer', 'marker': 'x'},
    'npu':        {'color': NATIVE_COLOR, 'hatch': NATIVE_HATCH, 'label': 'NPU-resident copy', 'marker': '^'},
    # motivation: an arrival into a running batch
    'alone':      {'color': NATIVE_COLOR, 'hatch': NATIVE_HATCH, 'label': 'Alone', 'marker': '^'},
    'gate':       {'color': LROS_COLOR,   'hatch': LROS_HATCH,   'label': 'Engine + held-out batch', 'marker': 'D'},
}


def style_for(key: str) -> Dict[str, Any]:
    return style_map.get(key, {'color': 'gray', 'hatch': '', 'label': key, 'marker': 'o'})


def legend_handles(keys: List[str]):
    """Patch handles in a fixed order, independent of what each axes drew."""
    handles = []
    for k in keys:
        st = style_for(k)
        handles.append(mpatches.Patch(facecolor=st['color'], hatch=st['hatch'],
                                      edgecolor='black', linewidth=0.4, label=st['label']))
    return handles, [style_for(k)['label'] for k in keys]


# --- axes -------------------------------------------------------------------

def format_big_numbers(x, pos=None):
    if x >= 1e9:
        return f'{x / 1e9:.0f}B'
    elif x >= 1e6:
        return f'{x / 1e6:.0f}M'
    elif x >= 1e3:
        return f'{x / 1e3:.0f}K'
    else:
        return f'{x:.0f}'


def better_hint(ax, higher: bool = False, xy=(0.0, 1.02)):
    """The blue hint above the axes telling which direction is good."""
    ax.annotate(higher_better_str if higher else lower_better_str,
                color='blue', xy=xy, xycoords='axes fraction',
                va='bottom', annotation_clip=False,
                fontsize=FONTSIZE_ANNOTATION)


def panel_caption(ax, text: str, y: float = -0.30):
    """Bold "(a) Something." caption underneath a subplot."""
    ax.text(0.5, y, bf(text), transform=ax.transAxes, ha='center', va='top',
            fontsize=FONTSIZE_TITLE, fontweight='bold')


def thin_spines(ax, hide=('top', 'right')):
    for s in hide:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK_LABEL, length=2, pad=1)


def grouped_bars(ax, categories, series, values, errors=None, baseline=None,
                 width_total=0.8, annotate=True, annotate_fmt='pct',
                 annotate_series=None, annotate_invert=False):
    """Grouped bars of `values[key]` over `categories`, one series per style
    key; against `baseline`, the others labelled with their relative difference
    (only `annotate_series` if given). Returns the group positions."""
    x = np.arange(len(categories), dtype=float)
    width = width_total / len(series)
    offsets = {}
    for i, key in enumerate(series):
        st = style_for(key)
        off = (i - len(series) / 2 + 0.5) * width
        offsets[key] = off
        ax.bar(x + off, values[key], width=width, label=st['label'],
               color=st['color'], edgecolor='black', linewidth=0.4,
               hatch=st['hatch'],
               yerr=None if errors is None else errors.get(key),
               capsize=1.5, error_kw=dict(lw=0.5, capthick=0.5), zorder=3)

    if annotate and baseline is not None:
        base = np.asarray(values[baseline], dtype=float)
        targets = annotate_series if annotate_series is not None else series
        for key in series:
            if key == baseline or key not in targets:
                continue
            vals = np.asarray(values[key], dtype=float)
            for k in range(len(categories)):
                if not (base[k] > 0 and vals[k] > 0):
                    continue
                top = max(vals[k], base[k])
                if errors is not None:
                    top = max(top, vals[k] + (errors.get(key, np.zeros_like(vals))[k]),
                              base[k] + (errors.get(baseline, np.zeros_like(base))[k]))
                if annotate_fmt == 'pct':
                    text = pct_label((vals[k] - base[k]) / base[k] * 100)
                else:
                    ratio = base[k] / vals[k] if annotate_invert else vals[k] / base[k]
                    text = speedup_label(ratio)
                ax.text(x[k] + (offsets[key] + offsets[baseline]) / 2, top * 1.04,
                        text, ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=FONTSIZE_TICK_LABEL)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    return x


def cdf(values):
    """x, y of the empirical CDF of `values`."""
    xs = np.sort(np.asarray(values, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


# --- output -------------------------------------------------------------------

PLATFORM_LABEL = {'orin': 'Jetson Orin', 'rk3588': 'RK3588'}


def save_path(fig, path):
    """Write to an explicit file path."""
    path = os.path.realpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.012)
    plt.close(fig)
    print("Saved " + min(path, os.path.relpath(path, os.getcwd()), key=len))
    return path


def parse_out_dir(default=None, description=None):
    """The first positional argument, else $PLOT_OUT_DIR, else the paper's mock
    folder; unknown arguments are ignored so a driver can pass its own."""
    fallback = default or os.environ.get("PLOT_OUT_DIR") or mock_dir
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("out_dir", nargs="?", default=fallback,
                        help=f"directory to write the figures into "
                             f"(default: {os.path.relpath(fallback, os.getcwd())})")
    args, _ = parser.parse_known_args()
    return os.path.realpath(args.out_dir)


def save(fig, name, directory=None):
    """Write `name` (a bare file name) into `directory`, creating it if needed."""
    out_dir = directory or mock_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.012)
    plt.close(fig)
    print("Saved " + min(path, os.path.relpath(path, os.getcwd()), key=len))
    return path
