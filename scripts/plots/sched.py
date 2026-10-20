#!/usr/bin/env python3
"""Interactive requests arriving into background work: interactive decode against
reading speed, with LROS's cost to the background, and TTFT, per device.

    python3 sched.py <out.pdf> --cpu <sched-cpu.csv> [...] --accel <sched-accel.csv> [...]
                     [--devices CPU,GPU,NPU]
"""
import argparse

from common import *

READING_TPS = 4.8   # 240 words per minute, Brysbaert 2019

PRIO_COLOR = '#B6D7A8'   # pastel green -- llama.cpp with priorities
ARMS = [
    ('fifo',             'llama.cpp',              LINUX_COLOR, LINUX_HATCH),
    ('priority-chunked', 'llama.cpp + priorities', PRIO_COLOR,  ALT_HATCH),
    ('lros',             'LROS',                   LROS_COLOR,  LROS_HATCH),
]
DEVICE = {'orin': 'GPU', 'rk3588': 'NPU'}

COLUMNS = [
    ('dec',  'Decode (tok/s)', True),
    ('ttft', 'TTFT',           False),
]


def summarise(df, by):
    """Per group: each metric's mean over the repetitions and its spread."""
    rows = []
    for key, d in df.groupby(by):
        per_rep = []
        for rep, r in d.groupby('rep'):
            i, bg = r[r.prio == 0], r[r.prio != 0]
            first_to_last = i.waited_s + i.secs - i.ttft_s
            per_rep.append(dict(
                dec=((i.gen - 1) / first_to_last).mean(),
                ttft=i.ttft_s.mean() * 1000,
                bg=(bg.prompt + bg.gen).sum() / bg.t_done_s.max(),
                bgct=(bg.waited_s + bg.secs).mean(),
            ))
        p = pd.DataFrame(per_rep)
        row = dict(zip(by, key), prompt=d[d.prio != 0].prompt.mean())
        for k in p:
            row[k], row[k + '_sd'] = p[k].mean(), p[k].std()
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_ms(v, _=None):
    return f"{v:.0f} ms" if v < 1000 else f"{v / 1000:.3g} s"


# TTFT spans orders of magnitude: logarithmic, fitted to the values, labelled
# in time units, with no more ticks than a narrow panel holds.
def ttft_axis(ax, s):
    lo = (s['ttft'] - s['ttft_sd'].fillna(0)).clip(lower=s['ttft'].min() / 2).min()
    hi = (s['ttft'] + s['ttft_sd'].fillna(0)).max()
    ax.set_yscale('log')
    ax.set_ylim(lo / 1.3, hi * 1.3)
    subs = (1, 2, 5) if hi / lo < 10 else (1, 3) if hi / lo < 300 else (1,)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=subs))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_ms))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


# Grouped bars: a group per load, a bar per arm.
def panel(ax, s, y, ylabel, higher):
    loads = s.groupby('fill').prompt.mean().sort_values()
    xs = np.arange(len(loads))
    width = 0.8 / len(ARMS)
    for k, (arm, _, color, hatch) in enumerate(ARMS):
        r = s[s.arm == arm].set_index('fill').reindex(loads.index)
        ax.bar(xs + (k - (len(ARMS) - 1) / 2) * width, r[y], width,
               yerr=r[y + '_sd'].fillna(0), color=color, hatch=hatch, edgecolor='black', linewidth=0.4,
               error_kw=dict(elinewidth=0.6, capsize=1.2), zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p:.0f}" if p < 1000 else f"{p / 1000:.1f}k" for p in loads],
                       fontsize=FONTSIZE_TICK_LABEL)
    if y == 'ttft':
        ttft_axis(ax, s)
    else:
        ax.set_ylim(0, (s[y] + s[y + '_sd'].fillna(0)).max() * 1.15)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
    ax.tick_params(labelsize=FONTSIZE_TICK_LABEL)
    ax.tick_params(axis='x', pad=1)
    ax.tick_params(axis='y', pad=1 if y != 'ttft' else 3)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    thin_spines(ax)
    better_hint(ax, higher=higher)


# Reading speed as a dashed line; the legend says which side to stay on.
def reading_speed(ax):
    ax.axhline(READING_TPS, color='black', linewidth=0.7, linestyle='--')
    ax.set_ylim(top=max(ax.get_ylim()[1], READING_TPS * 1.6))


# Under each load, in a row of its own below the axis: how long a background
# request takes to complete, from arrival to its last token, under LROS against
# llama.cpp with priorities at that load, what LROS costs the background for serving the
# interactive requests first.
BG_COLOR = '#4D4D4D'


def background_cost(ax, s):
    loads = s.groupby('fill').prompt.mean().sort_values()
    below = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, fill in enumerate(loads.index):
        d = s[s.fill == fill].set_index('arm')
        if 'lros' not in d.index or 'priority-chunked' not in d.index:
            continue
        ratio = d.bgct['lros'] / d.bgct['priority-chunked']
        pct = (ratio - 1) * 100
        text = (f"{ratio:.1f}$\\times$" if ratio >= 2
                else f"{pct:+.0f}\\%" if USETEX else f"{pct:+.0f}%")
        ax.annotate(text, (x, 0), xycoords=below, xytext=(0, -12),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=FONTSIZE_TICK_LABEL, color=BG_COLOR)
    ax.annotate('background\ncompletion time,\nLROS vs llama.cpp\n+ priorities:', (0, 0), xycoords='axes fraction',
                xytext=(-3, -12), textcoords='offset points', ha='right', va='top',
                linespacing=0.9,
                fontsize=FONTSIZE_ANNOTATION, color=BG_COLOR)


# The CSVs in order; an arm measured again in a later one replaces the earlier
# measurement of that arm on that platform.
def load(csvs):
    df = pd.DataFrame()
    for c in csvs:
        new = pd.read_csv(c)
        if not df.empty:
            keys = set(zip(new.platform, new.arm))
            df = df[[(p, a) not in keys for p, a in zip(df.platform, df.arm)]]
        df = pd.concat([df, new], ignore_index=True)
    return df


def plot(out, cpu_csvs, accel_csvs, devices):
    arms = [a for a, _, _, _ in ARMS]
    rows = []
    cpu = load(cpu_csvs)
    if not cpu.empty and 'CPU' in devices:
        rows.append(('CPU', summarise(cpu[cpu.arm.isin(arms)], ['arm', 'fill'])))
    accel = load(accel_csvs)
    for platform, name in DEVICE.items():
        if name in devices and not accel.empty and (accel.platform == platform).any():
            d = accel[(accel.platform == platform) & accel.arm.isin(arms)]
            rows.append((name, summarise(d, ['arm', 'fill'])))
    for name, s in rows:
        print(name)
        print(s.sort_values(['fill', 'arm']).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    fig, axes = plt.subplots(len(rows), len(COLUMNS), squeeze=False,
                             figsize=(figwidth_half, 1.4 * len(rows) + 0.4))
    for r, (name, s) in enumerate(rows):
        for c, (y, ylabel, higher) in enumerate(COLUMNS):
            panel(axes[r, c], s, y, ylabel, higher)
        reading_speed(axes[r, 0])
        background_cost(axes[r, 0], s)
        axes[r, 0].annotate(name, (0, 0.5), xycoords='axes fraction', xytext=(-22, 0),
                            textcoords='offset points', rotation=90, ha='right', va='center',
                            fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    fig.supxlabel('Background prompt (tok)', fontsize=FONTSIZE_AXIS_LABEL, y=0.01)
    fig.tight_layout(pad=0.2, w_pad=0.4, h_pad=1.5, rect=(0.03, 0.02, 1, 0.92 if len(rows) > 1 else 0.84))
    handles = [mpl.patches.Patch(facecolor=c, hatch=h, edgecolor='black', linewidth=0.4, label=l)
               for _, l, c, h in ARMS]
    handles.append(plt.Line2D([], [], color='black', linewidth=0.7, linestyle='--',
                              label='Minimal throughput for reading'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.0, columnspacing=1.0)
    save_path(fig, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--cpu', nargs='*', default=[])
    ap.add_argument('--accel', nargs='*', default=[])
    ap.add_argument('--devices', default='CPU,GPU,NPU', help='rows to draw, of CPU, GPU and NPU')
    a = ap.parse_args()
    plot(a.out, a.cpu, a.accel, a.devices.split(','))
