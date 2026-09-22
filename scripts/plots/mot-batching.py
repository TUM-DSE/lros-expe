#!/usr/bin/env python3
"""Aggregate and per-request decode rate as the batch grows.

    python3 mot-batching.py <mot-batching.csv> [out.pdf]
"""
import sys

from common import *

READING_SPEED = 4.8   # tok/s, one token every 208 ms


def plot(df, out_path):
    g = df.groupby('slots').agg(total=('s_tg', 'mean'), total_sd=('s_tg', 'std')).reset_index()
    g['per'] = g.total / g.slots
    g['per_sd'] = g.total_sd / g.slots

    fig, ax = plt.subplots(figsize=(2.2, 1.6))
    st_a, st_p = style_for('batched'), style_for('prefill')
    ax.errorbar(g.slots, g.total, yerr=g.total_sd, color=darken(st_a['color']), marker=st_a['marker'],
                markersize=2.5, linewidth=0.9, capsize=1.5, elinewidth=0.5, label='Aggregate', zorder=3)
    ax.errorbar(g.slots, g.per, yerr=g.per_sd, color=st_p['color'], marker=st_p['marker'],
                markersize=2.5, linewidth=0.9, capsize=1.5, elinewidth=0.5, label='Per request', zorder=3)
    ax.axhline(READING_SPEED, color='black', linestyle=':', linewidth=0.6, zorder=2)
    ax.text(g.slots.max() + 0.3, READING_SPEED, 'reading\nspeed', ha='right', va='bottom',
            fontsize=FONTSIZE_ANNOTATION)

    ax.set_xlabel('Requests in the batch')
    ax.set_ylabel('Decode (tok/s)')
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xlim(0.5, g.slots.max() + 0.5)
    ax.set_ylim(0, g.total.max() * 1.15)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    ax.legend(loc='center right', frameon=False, fontsize=FONTSIZE_LEGEND, handlelength=1.4)
    panel_caption(ax, '(b) Batching', y=-0.36)
    for _, r in g.iterrows():
        print(f"slots {int(r.slots):2d}  aggregate {r.total:5.1f}  per request {r.per:4.2f}")
    fig.tight_layout(pad=0.2)
    save_path(fig, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    plot(pd.read_csv(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('.csv', '.pdf'))
