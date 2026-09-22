#!/usr/bin/env python3
"""Decode under a memory quota as Linux pages the weights or swaps the KV cache,
from the log kvswap.sh writes.

    python3 mot-kvswap.py <kvswap.log> [out.pdf]
"""
import re
import sys

from common import *

# (log label, style key, tick label)
ARMS = [('A no-quota', 'alone', 'no quota'),
        ('B swappiness0', 'decode', 'weights paged'),
        ('C swappiness100', 'prefill', 'KV swapped')]

ROW = re.compile(r'^(?P<lbl>[A-D] \S+)\s+s_pp (?P<pp>\S+)\s+s_tg (?P<tg>\S+)\s+peakRSS (?P<rss>\d+)'
                 r'\s+swapped (?P<swp>\d+)\s+majflt (?P<mf>\d+)')


def load(path):
    rows = []
    for line in open(path):
        m = ROW.match(line.strip())
        if m and m['tg'] != 'FAIL':
            rows.append(dict(lbl=m['lbl'], pp=float(m['pp']), tg=float(m['tg']),
                             majflt=int(m['mf']), swapped=int(m['swp'])))
    return pd.DataFrame(rows)


def plot(df, out_path):
    g = df.groupby('lbl').agg(tg=('tg', 'mean'), sd=('tg', 'std'), mf=('majflt', 'mean'))
    fig, ax = plt.subplots(figsize=(2.2, 1.6))
    top = g.tg.max()
    for i, (lbl, key, _) in enumerate(ARMS):
        if lbl not in g.index:
            continue
        st = style_for(key)
        v, e = g.loc[lbl, 'tg'], 0 if np.isnan(g.loc[lbl, 'sd']) else g.loc[lbl, 'sd']
        ax.bar(i, v, width=0.7, yerr=e, color=st['color'], edgecolor='black', linewidth=0.4,
               hatch=st['hatch'], capsize=1.5, error_kw=dict(lw=0.5, capthick=0.5), zorder=3)
        ax.text(i, v + e + top * 0.03, f"{v:.2f}", ha='center', va='bottom',
                fontsize=FONTSIZE_ANNOTATION)
        print(f"{lbl:16s} decode {v:5.2f} t/s  major faults {g.loc[lbl, 'mf']:.0f}")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([t for _, _, t in ARMS], rotation=15, ha='right', fontsize=FONTSIZE_TICK_LABEL)
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel('Decode (tok/s)')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    panel_caption(ax, '(a) Weights or KV cache', y=-0.36)
    fig.tight_layout(pad=0.2)
    save_path(fig, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('.log', '.pdf')
    plot(load(sys.argv[1]), out)
