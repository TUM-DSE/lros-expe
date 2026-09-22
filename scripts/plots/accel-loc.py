#!/usr/bin/env python3
"""Lines of code to reach a device through VIAI, per platform, stacked by part,
with the native backend a guest would otherwise need given as a number.

    python3 accel-loc.py <accel-loc.csv> [out.pdf]
"""
import sys

from common import *

PLATFORMS = [('orin', 'Jetson Orin (CUDA)'), ('rk3588', 'RK3588 (NPU)')]
PARTS = [('guest backend', 'Guest backend', LROS_COLOR, ''),
         ('host server', 'Host server', lighten(LROS_COLOR, 0.45), '//'),
         ('device adapter', 'Device adapter', ALT_COLOR, 'xx')]


def plot(csv, out):
    df = pd.read_csv(csv).set_index(['platform', 'part'])['loc']
    platforms = [(p, n) for p, n in PLATFORMS if (p, 'guest backend') in df.index]

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height * 0.75))
    y = np.arange(len(platforms))[::-1].astype(float)
    left = np.zeros(len(platforms))
    for part, label, color, hatch in PARTS:
        vals = np.array([df.get((p, part), 0) for p, _ in platforms], dtype=float)
        ax.barh(y, vals, left=left, height=0.55, color=color, edgecolor='black',
                linewidth=0.4, hatch=hatch, label=label, zorder=3)
        left += vals

    right = left.max()
    for yi, (p, _), total in zip(y, platforms, left):
        native = df.get((p, 'native backend'), 0)
        ax.text(total + right * 0.02, yi, f"{int(total):,}", va='center', ha='left',
                fontsize=FONTSIZE_ANNOTATION)
        ax.text(right * 1.62, yi, f"native: {int(native):,}", va='center', ha='right',
                fontsize=FONTSIZE_ANNOTATION, color='0.30')
        print(f"{p:7s} viai={int(total):5d}  native backend={int(native):6d}")

    ax.set_xlim(0, right * 1.65)
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in platforms], fontsize=FONTSIZE_TICK_LABEL)
    ax.set_xlabel('Lines of code', fontsize=FONTSIZE_AXIS_LABEL)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.4,
              handleheight=0.8, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.1)
    save_path(fig, out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('.csv', '.pdf')
    plot(sys.argv[1], out)
