#!/usr/bin/env python3
"""TTFT and decode of the guest on its CPU, the guest through VIAI and the native
host library, per platform; VIAI labelled with the share of native it keeps.

    python3 accel-perf.py <out.pdf> <accel-perf.csv> [<accel-perf.csv> ...]
"""
import sys

from common import *

# (csv arm, style key)
ARMS = [('guest-cpu', 'lros-cpu'), ('guest-viai', 'viai'), ('host-native', 'native')]
PLATFORMS = [('orin', 'Jetson Orin'), ('rk3588', 'RK3588')]
# Where the accelerator never ran, so the arms measure something else. The RKNN
# adapter takes F16 weights only, so it refuses every matmul of a Q4_1 model and
# the guest falls back to its own CPU: the bars would compare two CPUs.
NO_ACCELERATOR = {('rk3588', 'gemma-3-1b-it-Q4_1.gguf')}
MODELS = [('Llama-3.2-1B-Instruct-f16.gguf', 'Llama'),
          ('gemma-3-1b-it-Q4_1.gguf', 'Gemma')]


def stats(df, platform, metric):
    """mean and std per (arm, model), in the order the bars are drawn"""
    d = df[df.platform == platform]
    models = [(f, n) for f, n in MODELS
              if (d.model == f).any() and (platform, f) not in NO_ACCELERATOR]
    mean, err = {}, {}
    for arm, key in ARMS:
        a = d[d.arm == arm]
        g = a.groupby('model')[metric].agg(['mean', 'std'])
        mean[key] = [g.loc[f, 'mean'] if f in g.index else np.nan for f, _ in models]
        err[key] = [0 if f not in g.index or np.isnan(g.loc[f, 'std']) else g.loc[f, 'std']
                    for f, _ in models]
    return models, mean, err


def panel(ax, df, platform, metric, ylabel, higher, clip=False):
    models, mean, err = stats(df, platform, metric)
    keys = [k for _, k in ARMS]
    x = grouped_bars(ax, [n for _, n in models], keys, mean, errors=err, annotate=False)
    width = 0.8 / len(keys)
    top = np.nanmax([v + e for k in keys for v, e in zip(mean[k], err[k])])
    # With a clipped axis the tallest bar is off the top, so labels are spaced
    # against what the reader can actually see.
    limit = np.nanmax([v + e for k in ('viai', 'native')
                       for v, e in zip(mean[k], err[k])]) * 1.45 if clip else None
    top = limit if clip else top
    vi, na = mean['viai'], mean['native']
    for i in range(len(models)):
        if np.isnan(vi[i]) or np.isnan(na[i]):
            continue
        share = vi[i] / na[i] if higher else na[i] / vi[i]
        off = (keys.index('viai') - len(keys) / 2 + 0.5) * width
        label = f"{share * 100:.0f}\\%" if USETEX else f"{share * 100:.0f}%"
        ax.text(x[i] + off, vi[i] + err['viai'][i] + top * 0.03, label,
                ha='center', va='bottom', fontsize=FONTSIZE_ANNOTATION)
        print(f"{platform:7s} {models[i][1]:13s} {metric:5s} "
              + "  ".join(f"{k}={mean[k][i]:.3f}" for k in keys)
              + f"  viai/native={share * 100:.1f}%")
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL)
    if clip:
        # The CPU bar is up to twenty times the accelerated one, which would
        # leave the two bars this panel is about a pixel high. The axis is cut
        # to the accelerated bars and the CPU bar runs off the top, carrying
        # its value instead.
        ax.set_ylim(0, limit)
        off = (keys.index('lros-cpu') - len(keys) / 2 + 0.5) * width
        for i in range(len(models)):
            v = mean['lros-cpu'][i]
            if np.isnan(v) or v <= limit:
                continue
            # Inside the top of the bar it belongs to: above the axes is where
            # the direction hint now lives.
            ax.text(x[i] + off, limit * 0.97, f"{v:.1f} s", ha='center', va='top',
                    fontsize=FONTSIZE_ANNOTATION,
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.9))
    else:
        ax.set_ylim(0, top * 1.25)
    better_hint(ax, higher=higher)


def plot(out, csvs):
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df['pass'] == df['pass'].max()]
    platforms = [(p, n) for p, n in PLATFORMS if (df.platform == p).any()]

    n = 2 * len(platforms)
    fig, axes = plt.subplots(1, n, figsize=(figwidth_full * n / 4, fig_height * 0.95))
    axes = np.atleast_1d(axes)
    letters = 'abcdefgh'
    for j, (p, name) in enumerate(platforms):
        a, b = axes[2 * j], axes[2 * j + 1]
        panel(a, df, p, 't_pp', 'TTFT (s)', higher=False, clip=True)
        panel(b, df, p, 's_tg', 'Decode thrghpt (tok/s)', higher=True)
        panel_caption(a, f"({letters[2 * j]}) {name}: TTFT", y=-0.18)
        panel_caption(b, f"({letters[2 * j + 1]}) {name}: decode thrghpt", y=-0.18)

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
