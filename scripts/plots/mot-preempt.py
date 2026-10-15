#!/usr/bin/env python3
"""Time to first token of an interactive request that arrives during a
background prefill, per device: llama.cpp (the slots engine, fifo) against
LROS, the median with the quartiles, and the same request alone on the device.

    python3 mot-preempt.py <out.pdf> <device label> <sched csv> [...]
"""
import sys

from common import *
import matplotlib.lines as mlines

ARMS = [('fifo', 'llamacpp'), ('lros', 'lros')]


def plot(out_path, groups):
    labels = [g[0] for g in groups]
    values = {k: [] for _, k in ARMS}
    errors = {k: [[], []] for _, k in ARMS}
    alone = []
    for _, csv in groups:
        d = pd.read_csv(csv)
        d = d[d.prio == 0]
        for arm, key in ARMS:
            t = d[d.arm == arm].ttft_s
            m = t.median()
            values[key].append(m)
            errors[key][0].append(m - t.quantile(0.25))
            errors[key][1].append(t.quantile(0.75) - m)
        a = d[d.arm == 'alone'].ttft_s
        alone.append(a.median() if len(a) else np.nan)

    fig, ax = plt.subplots(figsize=(2.2, 1.4))
    x = grouped_bars(ax, labels, [k for _, k in ARMS], values, annotate=False, width_total=0.7,
                     errors={k: np.array(v) for k, v in errors.items()})
    for i in range(len(labels)):
        if not np.isnan(alone[i]):
            ax.hlines(alone[i], x[i] - 0.4, x[i] + 0.4, color='black', linestyle='--',
                      linewidth=0.8, zorder=4)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1, 3, 10]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylabel('TTFT (s)')
    ax.set_ylim(top=np.nanmax(values['llamacpp']) * 6)
    better_hint(ax)
    # llama.cpp and LROS are in the legend above the figure.
    handles = [mlines.Line2D([], [], color='black', linestyle='--', linewidth=0.8, label='alone')]
    ax.legend(handles=handles, fontsize=FONTSIZE_LEGEND, frameon=False, loc='upper left',
              handlelength=1.6)
    panel_caption(ax, '(c) Preemption of a prefill', y=-0.25)
    for i, lab in enumerate(labels):
        a, b = values['llamacpp'][i], values['lros'][i]
        print(f"{lab:4s} llama.cpp {a:6.2f} s  lros {b:6.2f} s  x{a / b:.1f}  alone {alone[i]:6.2f} s")
    fig.tight_layout(pad=0.2)
    save_path(fig, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2:
        sys.exit(__doc__.strip())
    a = sys.argv[2:]
    plot(sys.argv[1], [tuple(a[i:i + 2]) for i in range(0, len(a), 2)])
