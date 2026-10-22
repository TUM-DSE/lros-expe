#!/usr/bin/env python3
"""The evaluation's figures, each a full-width row of panels under one legend:

  eval-main.pdf    (a-b) the end-to-end scenario, (c-d) VIA against the native
                   backend, (e-f) weights over a memory quota, (g) KV swap
  eval-sched.pdf   interactive requests into background work, CPU and GPU

Each panel is drawn by the script of its own figure; this one lays them out.

    python3 paper-eval.py <out-dir> <results-dir>
"""
import glob
import importlib
import sys

# Larger text than the single-column figures: these are set at the text width.
import common
common.FONTSIZE = 8.5
common.FONTSIZE_AXIS_LABEL = 8.5
common.FONTSIZE_TICK_LABEL = 8
common.FONTSIZE_LEGEND = 8
common.FONTSIZE_TITLE = 8
common.FONTSIZE_ANNOTATION = 7
common.mpl.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
                            'ytick.labelsize': 8, 'legend.fontsize': 8})
# The scripts below copy these sizes when they import common.
from common import *
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import orch
import sched
accel = importlib.import_module('accel-perf')

# swap.py and kvswap.py parse their own arguments only when run.
import swap
import kvswap

SYSTEMS = [('llama.cpp', LINUX_COLOR, LINUX_HATCH), ('LROS', LROS_COLOR, LROS_HATCH)]


def legend(fig, extra=(), y=1.0, systems=SYSTEMS):
    handles = [Patch(facecolor=c, hatch=h, edgecolor='black', linewidth=0.4, label=l)
               for l, c, h in systems] + list(extra)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y), ncol=len(handles),
               fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.4, handleheight=0.8,
               columnspacing=1.6, handletextpad=0.4, borderaxespad=0)


def caption(ax, text, y):
    panel_caption(ax, text, y=y)


# --- (a, b) the end-to-end scenario ------------------------------------------------

def e2e(ax_tpot, ax_bg, results):
    csvs = [os.path.join(results, 'orch/orangepi/orch.csv')]
    df = orch.load(csvs)
    tokens = orch.load_tokens(csvs, df)
    arms = ['host-fifo', 'lros-adaptive']
    mem = 'budget'
    # The dots a shade darker than the bars, to stay visible at their size.
    orch.ARM_COLOR.update({'host-fifo': darken(darken(LINUX_COLOR)), 'lros-adaptive': darken(darken(LROS_COLOR))})
    orch.draw_tpot(ax_tpot, tokens[tokens.mem == mem], arms, 208.0, compact=True)
    ax_tpot.set_ylabel('TPOT (s)', fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
    better_hint(ax_tpot, higher=False)

    apps = ['summarise', 'reply']
    width = 0.8 / len(arms)
    top = 0.0
    for i, (arm, (_, color, hatch)) in enumerate(zip(arms, SYSTEMS)):
        tps = [orch.background_tps(df, mem, arm, apps=(app,)) for app in apps]
        xs = np.arange(len(apps)) + (i - (len(arms) - 1) / 2) * width
        ax_bg.bar(xs, tps, width, color=color, hatch=hatch, edgecolor='black', linewidth=0.4, zorder=3)
        for x, v in zip(xs, tps):
            ax_bg.annotate(f"{v:.1f}", (x, v), xytext=(0, 1), textcoords='offset points', ha='center',
                           va='bottom', fontsize=FONTSIZE_ANNOTATION)
        top = max(top, np.nanmax(tps))
    ax_bg.set_xticks(np.arange(len(apps)))
    ax_bg.set_xticklabels([orch.APP_LABEL_PAPER[a] for a in apps], fontsize=FONTSIZE_TICK_LABEL)
    ax_bg.tick_params(axis='x', length=0)
    ax_bg.set_xlim(-0.5, len(apps) - 0.5)
    ax_bg.set_ylim(0, top * 1.2)
    ax_bg.set_ylabel('Throughput (tok/s)', fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
    better_hint(ax_bg, higher=True)
    ax_bg.grid(axis='y', linewidth=0.3, alpha=0.5)
    thin_spines(ax_bg)


# --- (c, d) VIA against the native backend ------------------------------------------

def via(ax_ttft, ax_dec, results):
    df = pd.concat([pd.read_csv(c) for c in sorted(glob.glob(os.path.join(results, 'accel-perf/*/accel-perf.csv')))],
                   ignore_index=True)
    df = df[df['pass'] == df['pass'].max()]
    accel.panel(ax_ttft, df, 't_pp', 'TTFT (s)', higher=False)
    accel.panel(ax_dec, df, 's_tg', 'Decode (tok/s)', higher=True)
    for ax in (ax_ttft, ax_dec):
        ax.yaxis.label.set_size(FONTSIZE_AXIS_LABEL)
        ax.yaxis.labelpad = 1


# --- (e, f) weights over a memory quota, (g) KV swap ---------------------------------

def weights(axes, results):
    data = swap.load([os.path.join(results, 'swap/orangepi/swap.csv')], 's_tg', 16)
    of_moe = data['model'] == 'olmoe-1b-7b-0924-instruct-q4_k_m.gguf'
    data = data[~of_moe | data['quota_pct'].astype(float).isin({110, 50, 20, 10})]
    configs = ['linux-mmap', 'lros-tensor-layer']
    for ax, model in zip(axes, ['Llama-3.2-1B-Instruct-f16', 'olmoe-1b-7b-0924-instruct-q4_k_m']):
        sub = data[(data['model'] == model + '.gguf') & (data['disk'] == 'nvme') & data['config'].isin(configs)]
        budgets = np.sort(sub['budget'].unique())[::-1]
        swap.draw_panel(ax, sub, configs, 's_tg', budgets, swap.axis_range(sub, 's_tg', budgets), '',
                        short_cut_label=True)
        # The caption says what the dashed line marks; its note would sit on
        # the first speedup label.
        for t in list(ax.texts):
            if 'not fit' in t.get_text():
                t.remove()
        ax.set_title('')
        ax.set_xlabel(f'Memory budget ({PCT} of model)', fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
        better_hint(ax, higher=True)
    axes[0].set_ylabel('Decode (tok/s)', fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)


def kv(ax, results):
    df = kvswap.load([os.path.join(results, 'kvswap/orangepi/kvswap.csv')])
    sub = df[(df['model'] == 'Llama-3.2-1B-Instruct-f16') & (df['disk'] == 'nvme')]
    configs = kvswap.ordered(pd.concat([sub[sub['keep_pct'] < 0]['config'],
                                        sub[sub['keep_pct'] >= 0]['base']]), kvswap.CONFIG_ORDER)
    # LROS in its own colour, as in every panel.
    kvswap.CONFIG_STYLE['lros-64K'] = dict(kvswap.CONFIG_STYLE['lros-64K'], color=LROS_COLOR)
    kvswap.draw_partial(sub, ['nvme'], [ax], configs, compact=False, split=False)
    ax.set_title('')
    ax.set_xlabel('KV kept resident', fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
    ax.set_ylabel('Round trip (ms)', fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
    better_hint(ax, higher=False)


def main_figure(out, results):
    fig = plt.figure(figsize=(figwidth_full, 2.75))
    gs = gridspec.GridSpec(2, 12, figure=fig, hspace=0.72, wspace=1.9,
                           left=0.058, right=0.995, top=0.89, bottom=0.125)
    top = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(4)]
    bottom = [fig.add_subplot(gs[1, 4 * i:4 * i + 4]) for i in range(3)]
    e2e(top[0], top[1], results)
    via(top[2], top[3], results)
    weights(bottom[:2], results)
    kv(bottom[2], results)
    for ax, t in zip(top, ['(a) Chat TPOT', '(b) Background throughput', '(c) VIA: TTFT', '(d) VIA: decode']):
        caption(ax, t, y=-0.375)
    for ax, t in zip(bottom, ['(e) Weights: Llama 3.2 1B (f16)', '(f) Weights: OLMoE 1B-7B (Q4\\_K\\_M)' if USETEX
                              else '(f) Weights: OLMoE 1B-7B (Q4_K_M)', '(g) KV swap: Llama 3.2 1B']):
        caption(ax, t, y=-0.42)
    legend(fig, extra=[plt.Line2D([], [], color='black', linewidth=0.7, linestyle='--',
                                  label='Chat TPOT target')], y=0.93)
    save_path(fig, out)


# --- eval-sched: one row, CPU and GPU, decode and TTFT -------------------------------

def background_cost(ax, s):
    """Under each load, how long a background request takes to complete under
    LROS against llama.cpp with priorities."""
    loads = s.groupby('fill').prompt.mean().sort_values()
    below = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, fill in enumerate(loads.index):
        d = s[s.fill == fill].set_index('arm')
        if 'lros' not in d.index or 'priority-chunked' not in d.index:
            continue
        ratio = d.bgct['lros'] / d.bgct['priority-chunked']
        pct = round((ratio - 1) * 100)
        text = (f"{ratio:.1f}$\\times$" if ratio >= 2 else f"{pct:+.0f}\\%" if USETEX else f"{pct:+.0f}%")
        ax.annotate(text, (x, 0), xycoords=below, xytext=(0, -11), textcoords='offset points',
                    ha='center', va='top', fontsize=FONTSIZE_ANNOTATION, color=sched.BG_COLOR)


# The board whose CPU the CPU row shows: one board only, as the loads differ.
CPU_PLATFORM = 'rk3588'


def sched_figure(out, results):
    def csvs(*names):
        return [c for n in names for c in sorted(glob.glob(os.path.join(results, n, '*', n + '.csv')))]
    arms = [a for a, _, _, _ in sched.ARMS]
    cpu = sched.load(csvs('sched-cpu', 'sched-cpu-lros', 'sched-cpu-host'))
    acc = sched.load(csvs('sched-accel', 'sched-accel-host', 'sched-accel-lros'))
    rows = [('CPU', sched.summarise(cpu[(cpu.platform == CPU_PLATFORM) & cpu.arm.isin(arms)], ['arm', 'fill'])),
            ('GPU', sched.summarise(acc[(acc.platform == 'orin') & acc.arm.isin(arms)], ['arm', 'fill']))]

    fig, axes = plt.subplots(1, 4, figsize=(figwidth_full, 1.95))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.77, bottom=0.33, wspace=0.48)
    letters = iter('abcd')
    for k, (name, s) in enumerate(rows):
        dec, ttft = axes[2 * k], axes[2 * k + 1]
        sched.panel(dec, s, 'dec', 'Decode (tok/s)', True)
        sched.reading_speed(dec)
        background_cost(dec, s)
        sched.panel(ttft, s, 'ttft', 'TTFT', False)
        caption(dec, f'({next(letters)}) {name}: interactive decode', y=-0.56)
        caption(ttft, f'({next(letters)}) {name}: interactive TTFT', y=-0.56)
    for ax in axes:
        ax.set_xlabel('Background prompt (tok)', fontsize=FONTSIZE_AXIS_LABEL, labelpad=9)
    legend(fig, extra=[plt.Line2D([], [], color='black', linewidth=0.7, linestyle='--',
                                  label='Minimal throughput for reading')], y=0.885,
           systems=[(l, c, h) for _, l, c, h in sched.ARMS])
    save_path(fig, out)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    out_dir, results = sys.argv[1], sys.argv[2]
    main_figure(os.path.join(out_dir, 'eval-main.pdf'), results)
    sched_figure(os.path.join(out_dir, 'eval-sched.pdf'), results)
