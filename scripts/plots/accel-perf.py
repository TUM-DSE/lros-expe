#!/usr/bin/env python3
"""TTFT and decode of the guest through VIA and of the native host library, per
platform; VIA labelled with the share of native it keeps.

    python3 accel-perf.py <out.pdf> <accel-perf.csv> [<accel-perf.csv> ...]
"""
import sys

from common import *

# (csv arm, style key)
ARMS = [('host-native', 'native'), ('guest-viai', 'viai')]
PLATFORMS = [('orin', 'GPU'), ('rk3588', 'NPU')]
MODELS = [('Llama-3.2-1B-Instruct-f16.gguf', 'Llama'),
          ('olmoe-1b-7b-0924-instruct-q8_0.gguf', 'OLMoE')]
# Where the accelerator never ran, so the arms would compare two CPUs: neither
# NPU path runs OLMoE's expert multiplications or its F32 router.
NO_ACCELERATOR = {('rk3588', 'olmoe-1b-7b-0924-instruct-q8_0.gguf')}


def stats(df, metric):
    """the bar groups (one per platform and model) and the mean and std of each
    arm in them, in the order the bars are drawn"""
    groups = [(p, f, f"{pn}\n{mn}") for p, pn in PLATFORMS for f, mn in MODELS
              if ((df.platform == p) & (df.model == f)).any() and (p, f) not in NO_ACCELERATOR]
    mean, err = {}, {}
    for arm, key in ARMS:
        mean[key], err[key] = [], []
        for p, f, _ in groups:
            v = df[(df.platform == p) & (df.model == f) & (df.arm == arm)][metric]
            mean[key].append(v.mean() if len(v) else np.nan)
            err[key].append(v.std() if len(v) > 1 else 0)
    return groups, mean, err


def panel(ax, df, metric, ylabel, higher, log=False):
    groups, mean, err = stats(df, metric)
    keys = [k for _, k in ARMS]
    x = grouped_bars(ax, [g[2] for g in groups], keys, mean, errors=err, annotate=False)
    width = 0.8 / len(keys)
    top = np.nanmax([v + e for k in keys for v, e in zip(mean[k], err[k])])
    vi, na = mean['viai'], mean['native']
    for i, (p, _, name) in enumerate(groups):
        if np.isnan(vi[i]) or np.isnan(na[i]):
            continue
        share = vi[i] / na[i] if higher else na[i] / vi[i]
        off = (keys.index('viai') - len(keys) / 2 + 0.5) * width
        label = f"{share * 100:.0f}\\%" if USETEX else f"{share * 100:.0f}%"
        y = (vi[i] + err['viai'][i]) * 1.15 if log else vi[i] + err['viai'][i] + top * 0.03
        ax.text(x[i] + off, y, label, ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION)
        print(f"{name.replace(chr(10), ' '):22s} {metric:5s} "
              + "  ".join(f"{k}={mean[k][i]:.3f}" for k in keys)
              + f"  viai/native={share * 100:.1f}%")
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL)
    if log:
        # TTFT spans a factor of twenty between the GPU and the NPU.
        ax.set_yscale('log')
        bottom = 10 ** np.floor(np.log10(np.nanmin([v for k in keys for v in mean[k]])))
        ax.set_ylim(bottom, top * 4)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_ylim(0, top * 1.3)
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICK_LABEL)
    better_hint(ax, higher=higher)


def plot(out, csvs):
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df['pass'] == df['pass'].max()]

    fig, (a, b) = plt.subplots(1, 2, figsize=(figwidth_half, 1.5))
    panel(a, df, 't_pp', 'TTFT (s)', higher=False)
    panel(b, df, 's_tg', 'Decode (tok/s)', higher=True)
    panel_caption(a, "(a) TTFT", y=-0.42)
    panel_caption(b, "(b) Decode throughput", y=-0.42)

    keys = [k for _, k in ARMS]
    fig.legend(*legend_handles(keys), loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(keys), fontsize=FONTSIZE_LEGEND, frameon=False, handlelength=1.4,
               handleheight=0.8, columnspacing=1.6, handletextpad=0.4)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    save_path(fig, out)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__.strip())
    plot(sys.argv[1], sys.argv[2:])
