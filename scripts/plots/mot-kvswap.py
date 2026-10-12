#!/usr/bin/env python3
"""Time to the next token after a request's KV cache left memory, against its
context length: the state file of llama.cpp on Linux, which moves the whole
cache, against LROS with all or 10% of the cache evicted, and the decode step
with the cache resident.

    python3 mot-kvswap.py <mot-kvswap.csv> [out.pdf]
"""
import sys

from common import *
import matplotlib.lines as mlines


def plot(csv, out_path):
    d = pd.read_csv(csv)
    d['keep'] = d['keep'].fillna(-1).astype(int)
    d['resume'] = (d.t_in + d.t_dec2) / 1000
    med = lambda q, col: q.groupby('n')[col].median()

    arms = [('kv-state', med(d[d.arm == 'linux'], 'resume')),
            ('kv-all', med(d[(d.arm == 'lros') & (d.keep == 0)], 'resume')),
            ('kv-part', med(d[(d.arm == 'lros') & (d.keep == 90)], 'resume'))]
    resident = med(d[d.arm != 'recompute-gpu'], 't_dec') / 1000
    ns = sorted(arms[0][1].index)

    fig, ax = plt.subplots(figsize=(2.2, 1.6))
    values = {k: [s.get(n, np.nan) for n in ns] for k, s in arms}
    x = grouped_bars(ax, [format_big_numbers(n) for n in ns], [k for k, _ in arms], values,
                     annotate=False, width_total=0.8)
    for i, n in enumerate(ns):
        ax.hlines(resident[n], x[i] - 0.45, x[i] + 0.45, color='black', linestyle='--',
                  linewidth=0.8, zorder=4)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.03, 0.1, 0.3, 1, 3]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('Context (tokens)')
    ax.set_ylabel('Time to next token (s)')
    better_hint(ax)
    # llama.cpp and LROS (all evicted) are in the legend above the figure.
    handles, _ = legend_handles(['kv-part'])
    handles.append(mlines.Line2D([], [], color='black', linestyle='--', linewidth=0.8, label='no eviction'))
    ax.legend(handles=handles, fontsize=FONTSIZE_LEGEND, frameon=False, loc='upper left',
              ncol=2, handlelength=1.6, columnspacing=0.8)
    ax.set_ylim(0.03, np.nanmax(values['kv-state']) * 5)
    panel_caption(ax, '(b) KV cache eviction', y=-0.36)
    for k, s in arms:
        print(f"{style_for(k)['label']:24s} " + "  ".join(f"{n}:{v:.3f}" for n, v in s.items()))
    print(f"{'no eviction':24s} " + "  ".join(f"{n}:{v:.3f}" for n, v in resident.items()))
    fig.tight_layout(pad=0.2)
    save_path(fig, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('.csv', '.pdf'))
