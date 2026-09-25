#!/usr/bin/env python3
"""The orchestration experiment, from a `just orch_sweep` run.

Two figures per memory setting, under bench/plots/orch/:

  orch-timeline-<mem>   one panel per arm: every request as a bar from its
                        arrival to its completion, in a lane per app (chat,
                        summarise, reply). A hatched bar missed its deadline;
                        a chat bar is annotated with the turn's TPOT.
  orch-qos-<mem>        what each app got, one panel per metric, a bar per
                        arm: the chat's TPOT (mean, p95 as a tick) against
                        its reading-speed target, its TTFT, and the share of
                        summarise and reply requests that met their deadline.
                        Averaged over the runs, with the spread.

    python3 orch.py <orch.csv> [...] [-o out_dir]
"""
import argparse
import os

from common import *

APPS = ['chat', 'summarise', 'reply']
APP_LABEL = {'chat': 'Chat (fg)', 'summarise': 'Summarise (bg)', 'reply': 'Reply (bg, LoRA)'}
# Fixed order and colour per arm, never cycled.
ARMS = [
    ('host-fifo',     'llama.cpp (Linux)',        LINUX_COLOR),
    ('host-prio',     'vLLM/SGLang-like (Linux)', ALT_COLOR),
    ('slots-fifo',    'llama.cpp (guest)',        lighten(LINUX_COLOR, 0.45)),
    ('slots-prio',    'vLLM/SGLang-like (guest)', lighten(ALT_COLOR, 0.45)),
    ('lros-fixed',    'LROS, fixed',              lighten(LROS_COLOR, 0.45)),
    ('lros-adaptive', 'LROS, adaptive',           LROS_COLOR),
]
ARM_LABEL = {a: l for a, l, _ in ARMS}
ARM_COLOR = {a: c for a, _, c in ARMS}
MEM_LABEL = {'unbounded': 'weights fit', 'budget': 'weights budget'}
PNG = False   # also write a PNG beside each PDF (--png), for a quick look


def save_fig(fig, path):
    save_path(fig, path)
    if PNG:
        fig.savefig(path[:-4] + '.png', dpi=130)


def load(csvs):
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    # `truncated`: a slot arm stopped once its background was done and one
    # more chat turn through; its chat figures cover the turns it made.
    df = df[df.status.isin(['ok', 'truncated'])] if 'status' in df else df
    df['t_arrive_s'] = df.t_done_s - df.e2e_s
    df['met'] = df.deadline.map({'met': 1.0, 'missed': 0.0})
    return df


def summarise(df):
    """Per (mem, arm, run): the QoS figures; then the mean and spread over runs."""
    rows = []
    for (mem, arm, run), d in df.groupby(['mem', 'arm', 'run']):
        chat = d[d.app == 'chat']
        summ = d[d.app == 'summarise']
        reply = d[d.app == 'reply']
        rows.append(dict(
            mem=mem, arm=arm, run=run,
            chat_tpot=chat.tpot_ms.mean(), chat_p95=chat.tpot_p95_ms.max(),
            chat_late=chat.late.sum() / max(1, chat.gaps.sum()) * 100,
            # The buffering a stall-free display would have needed: the worst
            # turn, since the user meets the worst one.
            chat_buffer=chat.buffer_ms.max() if 'buffer_ms' in chat and chat.buffer_ms.notna().any() else np.nan,
            chat_ttft=chat.ttft_s.mean() * 1000,
            summ_met=summ.met.mean() * 100, summ_e2e=summ.e2e_s.mean(),
            reply_met=reply.met.mean() * 100, reply_e2e=reply.e2e_s.mean(),
            bg_tps=(summ.gen.sum() + reply.gen.sum()) / max(1e-9, d.t_done_s.max()),
        ))
    per_run = pd.DataFrame(rows)
    out = per_run.groupby(['mem', 'arm']).agg(['mean', 'std']).drop(columns='run')
    out.columns = [f"{a}_{b}" if b == 'std' else a for a, b in out.columns]
    return per_run, out.reset_index()


def timeline(df, mem, out_dir, ttft_target_s):
    arms = [a for a, _, _ in ARMS if (df.arm == a).any()]
    run = df.run.min()
    d = df[(df.mem == mem) & (df.run == run)]
    fig, axes = plt.subplots(len(arms), 1, sharex=True, squeeze=False,
                             figsize=(figwidth_full, 0.9 * len(arms) + 0.5))
    for ax, arm in zip(axes[:, 0], arms):
        r = d[d.arm == arm]
        for lane, app in enumerate(APPS):
            q = r[r.app == app].sort_values('t_arrive_s')
            # Requests in flight at once go into sub-lanes, first fit.
            ends, subs = [], []
            for _, x in q.iterrows():
                for i, end in enumerate(ends):
                    if end <= x.t_arrive_s:
                        ends[i] = x.t_done_s
                        subs.append(i)
                        break
                else:
                    ends.append(x.t_done_s)
                    subs.append(len(ends) - 1)
            n_sub = max(1, len(ends))
            h = 0.8 / n_sub
            for (_, x), sub in zip(q.iterrows(), subs):
                y = lane - 0.4 + (sub + 0.5) * h
                # A miss: a background deadline, or a chat display that would
                # have stalled at the target rate. A chat TTFT over target
                # gets a black edge.
                stalled = app == 'chat' and pd.notna(x.get('buffer_ms')) and x.buffer_ms > 0
                missed = x.deadline == 'missed' or stalled
                slow_start = app == 'chat' and x.ttft_s > ttft_target_s
                edge = 'black' if slow_start else 'white'
                # The prefill, arrival to first token, lighter; the decode full.
                t_first = x.t_arrive_s + x.ttft_s
                ax.barh(y, x.ttft_s, left=x.t_arrive_s, height=h * 0.9,
                        color=ARM_COLOR[arm], edgecolor=edge, linewidth=0.3 if edge == 'white' else 0.8,
                        alpha=0.35)
                ax.barh(y, x.t_done_s - t_first, left=t_first, height=h * 0.9,
                        color=ARM_COLOR[arm], edgecolor=edge, linewidth=0.3 if edge == 'white' else 0.8,
                        hatch='////' if missed else None, alpha=0.9)
                if app == 'chat':
                    ax.annotate(f"{x.tpot_ms:.0f}·{int(x.gen)}", (x.t_done_s, y), xytext=(2, 0),
                                textcoords='offset points', va='center',
                                fontsize=FONTSIZE_ANNOTATION)
        label = ARM_LABEL[arm]
        if (r.status == 'truncated').any():
            t_stop = r.t_done_s.max()
            ax.axvline(t_stop, color='black', linewidth=0.7, linestyle=':')
            label += f" (stopped at {t_stop:.0f} s)"
        ax.set_yticks(range(len(APPS)))
        ax.set_yticklabels([APP_LABEL[a] for a in APPS], fontsize=FONTSIZE_TICK_LABEL)
        ax.set_ylim(-0.6, len(APPS) - 0.4)
        ax.invert_yaxis()
        ax.annotate(label, (1.0, 1.0), xycoords='axes fraction', xytext=(-2, -2),
                    textcoords='offset points', ha='right', va='top', fontsize=FONTSIZE_TITLE)
        ax.grid(axis='x', linewidth=0.3, alpha=0.5)
        thin_spines(ax)
    axes[-1, 0].set_xlabel('Trace time (s)', fontsize=FONTSIZE_AXIS_LABEL)
    fig.suptitle(f"Requests over the trace, {MEM_LABEL.get(mem, mem)} (run {run}). "
                 "Light: prefill to the first token. Chat labels: TPOT (ms)·tokens generated.\n"
                 "Hatched: deadline missed, or the chat's display would stall at the target rate. "
                 "Black edge: chat TTFT over target. Dotted line: run stopped there.",
                 fontsize=FONTSIZE_TITLE)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    save_fig(fig, os.path.join(out_dir, f"orch-timeline-{mem}.pdf"))


def qos(stats, mem, out_dir, tpot_target):
    s = stats[stats.mem == mem]
    arms = [a for a, _, _ in ARMS if (s.arm == a).any()]
    panels = [
        ('chat_tpot',   'Chat TPOT (ms)',           False),
        ('chat_buffer', 'Chat display buffer (ms)', False),
        ('chat_ttft',   'Chat TTFT (ms)',           False),
        ('summ_met',    'Summarise on time (%)',    True),
        ('reply_met',   'Reply on time (%)',        True),
    ]
    if s.chat_buffer.isna().all():
        panels = [p for p in panels if p[0] != 'chat_buffer']
    fig, axes = plt.subplots(1, len(panels), figsize=(figwidth_full, 1.5), squeeze=False)
    xs = np.arange(len(arms))
    for ax, (key, label, higher) in zip(axes[0], panels):
        r = s.set_index('arm').reindex(arms)
        ax.bar(xs, r[key], 0.7, yerr=r[key + '_std'].fillna(0),
               color=[ARM_COLOR[a] for a in arms], edgecolor='white', linewidth=0.3,
               error_kw=dict(elinewidth=0.6, capsize=1.2))
        if key == 'chat_tpot':
            ax.scatter(xs, r['chat_p95'], marker='_', color='black', s=40, linewidths=0.8,
                       label='p95', zorder=3)
            ax.axhline(tpot_target, color='black', linewidth=0.7, linestyle='--')
            ax.annotate('target', (len(arms) - 0.5, tpot_target), xytext=(0, 2),
                        textcoords='offset points', ha='right', fontsize=FONTSIZE_ANNOTATION)
        if key.endswith('_met'):
            ax.set_ylim(0, 105)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=30, ha='right',
                           fontsize=FONTSIZE_TICK_LABEL)
        ax.set_ylabel(label, fontsize=FONTSIZE_AXIS_LABEL, labelpad=1)
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
        thin_spines(ax)
        better_hint(ax, higher=higher)
    fig.suptitle(MEM_LABEL.get(mem, mem), fontsize=FONTSIZE_TITLE, y=0.99)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9))
    save_fig(fig, os.path.join(out_dir, f"orch-qos-{mem}.pdf"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csvs', nargs='+')
    ap.add_argument('-o', '--out-dir', default=os.path.join(dir_path, '../../bench/plots/orch'))
    ap.add_argument('--tpot-target', type=float, default=208.0, help='ms, the chat target drawn')
    ap.add_argument('--ttft-target', type=float, default=3.0, help='s, the chat TTFT bound marked')
    ap.add_argument('--png', action='store_true', help='also write a PNG beside each PDF')
    a = ap.parse_args()
    global PNG
    PNG = a.png
    os.makedirs(a.out_dir, exist_ok=True)

    df = load(a.csvs)
    per_run, stats = summarise(df)
    cols = ['mem', 'arm', 'chat_tpot', 'chat_p95', 'chat_late', 'chat_buffer', 'chat_ttft',
            'summ_met', 'summ_e2e', 'reply_met', 'reply_e2e', 'bg_tps']
    print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    for mem in df.mem.unique():
        timeline(df, mem, a.out_dir, a.ttft_target)
        qos(stats, mem, a.out_dir, a.tpot_target)


if __name__ == '__main__':
    main()
