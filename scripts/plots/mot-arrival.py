#!/usr/bin/env python3
"""TTFT of an interactive request arriving into a background batch, per arm,
the median over trials, against the uncontended floor.

    python3 mot-arrival.py <mot-arrival.csv> [out.pdf]
"""
import sys

from common import *

# (csv arm, style key, tick label)
ARMS = [('alone', 'alone', 'alone'),
        ('base', 'engine', 'in batch'),
        ('deadline', 'gate', 'with prio')]


def plot(csv, out_path):
    g = pd.read_csv(csv).groupby('arm').agg(ttft=('ttft_ms', 'median'), late=('late_frac', 'median'))

    fig, ax = plt.subplots(figsize=(2.2, 1.6))
    floor = g.loc['alone', 'ttft'] / 1000
    top = g.ttft.max() / 1000
    for i, (arm, key, _) in enumerate(ARMS):
        st = style_for(key)
        v, late = g.loc[arm, 'ttft'] / 1000, g.loc[arm, 'late'] * 100
        ax.bar(i, v, width=0.7, color=st['color'], edgecolor='black', linewidth=0.4,
               hatch=st['hatch'], zorder=3)
        ax.text(i, v + top * 0.03, f"{v:.1f} s", ha='center', va='bottom',
                fontsize=FONTSIZE_ANNOTATION)
    ax.axhline(floor, color='black', linestyle=':', linewidth=0.6, zorder=2)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([t for _, _, t in ARMS], rotation=15, ha='right', fontsize=FONTSIZE_TICK_LABEL)
    ax.set_ylabel('TTFT (s)')
    ax.set_ylim(0, top * 1.2)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    panel_caption(ax, '(c) Arrival into a batch', y=-0.36)
    for arm, _, _ in ARMS:
        print(f"{arm:9s} ttft {g.loc[arm, 'ttft'] / 1000:5.1f} s  late {g.loc[arm, 'late'] * 100:4.0f}%")
    fig.tight_layout(pad=0.2)
    save_path(fig, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('.csv', '.pdf'))
