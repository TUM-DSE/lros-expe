#!/usr/bin/env python3
"""Decode throughput against the share of the weights in memory: llama.cpp on
Linux as shipped and with part of the model locked, LROS, and the bound of an
engine that reads the missing bytes at the disk's bandwidth while it computes,
over the limited shares.

    python3 mot-paging.py <out.pdf> [--default csv] [--mlock csv] [--lros csv] [--gbs 3.1]
"""
import argparse

from common import *
import matplotlib.lines as mlines

DISK_GBS = 3.1   # joy's NVMe, 4 MiB direct reads


def medians(csv):
    d = pd.read_csv(csv, dtype={'quota_pct': str})
    g = d.groupby('quota_pct').s_tg.median()
    return {100 if q == 'none' else int(q): v for q, v in g.items()}, d


def plot(out_path, arms, gbs):
    data = {k: medians(c) for k, c in arms.items() if c}
    series = [k for k in ('linux', 'linux-mlock', 'lros') if k in data]
    linux_key = next(k for k in series if k != 'lros')
    shares = sorted(data[linux_key][0], reverse=True)   # from the whole model down to a tenth
    model_bytes = data[linux_key][1].model_mb.iloc[0] * 2 ** 20
    t_compute = 1 / data[linux_key][0][100]
    bound = lambda s: 1 / max(t_compute, model_bytes * (1 - s / 100) / (gbs * 1e9))

    values = {k: [data[k][0].get(s, np.nan) for s in shares] for k in series}
    fig, ax = plt.subplots(figsize=(2.2, 1.6))
    x = grouped_bars(ax, [str(s) for s in shares], series, values, annotate=False,
                     width_total=0.8 if len(series) > 2 else 0.7)
    for i, s in enumerate(shares):
        if s != 100:
            ax.hlines(bound(s), x[i] - 0.45, x[i] + 0.45, color='black', linestyle='--',
                      linewidth=0.8, zorder=4)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    low = np.nanmin([v for k in series for v in values[k]])
    ax.set_ylim(0.3 if low > 0.4 else 0.1, np.nanmax([v for k in series for v in values[k]]) * 4)
    ax.set_xlabel('Weights in memory (%)')
    ax.set_ylabel('Decode (tokens/s)')
    better_hint(ax, higher=True)
    # llama.cpp as shipped and LROS are in the legend above the figure.
    handles = legend_handles(['linux-mlock'])[0] if 'linux-mlock' in series else []
    handles.append(mlines.Line2D([], [], color='black', linestyle='--', linewidth=0.8,
                                 label='I/O bound (computed)'))
    ax.legend(handles=handles, fontsize=FONTSIZE_LEGEND, frameon=False, loc='upper right',
              handlelength=1.6, labelspacing=0.2)
    panel_caption(ax, '(a) Weight paging', y=-0.36)
    for s in shares:
        row = '  '.join(f"{k} {data[k][0].get(s, float('nan')):6.2f}" for k in series)
        print(f"{s:3d}%  {row}  bound {bound(s):6.2f}")
    fig.tight_layout(pad=0.2)
    save_path(fig, out_path)


if __name__ == '__main__':
    a = argparse.ArgumentParser(usage=__doc__.strip())
    a.add_argument('out')
    a.add_argument('--default')
    a.add_argument('--mlock')
    a.add_argument('--lros')
    a.add_argument('--gbs', type=float, default=DISK_GBS)
    o = a.parse_args()
    plot(o.out, {'linux': o.default, 'linux-mlock': o.mlock, 'lros': o.lros}, o.gbs)
