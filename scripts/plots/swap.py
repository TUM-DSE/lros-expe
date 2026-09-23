#!/usr/bin/env python3
"""Weight paging under memory pressure, from a `just swap_sweep` run.

The real-data counterpart of mock_memory.py's mem-weights-tp.pdf: decode
throughput against the memory quota as a fraction of the model size, one line
per paging configuration (llama.cpp mmap on Linux, the LROS prefetcher with
4K and 2M buffers), one panel per disk the model was read from.

  python3 swap.py [swap.csv ...] [-o OUT_DIR] [--metric s_tg|s_pp|s] [--ntg N]

With no CSV the newest bench/out/swap/<timestamp>/swap.csv is used. Several
CSVs are pooled; each bar is the median over repeated runs, with error bars
from the runs' minimum to maximum. The figure comes in four layouts per model,
under bench/plots/swap/mem-weights-tp/<model>/ unless -o says otherwise, as
PDF and PNG:
  mem-weights-tp.pdf                        both disks, one column per panel
  single-column/mem-weights-tp.pdf          both disks fitted into one column
  single-column/mem-weights-tp_<disk>.pdf   one disk alone, one column
"""
import glob
import sys

from common import *

# The sweep's config names, mapped onto the paper's series identities. Buffer
# sizes differ by hatch; the -nopin arms (--pin-percent 0, S3-FIFO over the
# whole cache instead of a pinned 75% prefix) take a lighter LROS shade.
NOPIN_COLOR = lighten(LROS_COLOR, 0.45)
# Configurations left out of the figures even when the data has them: the
# no-pin arms add nothing over the pinned ones. --configs overrides this.
HIDDEN_CONFIGS = {"lros-128K-nopin", "lros-2M-nopin"}
CONFIG_STYLE = {
    "linux-mmap": ("mmap", {}),
    "lros-4K": ("prefetch", {"label": "LROS (prefetch, 4K)", "hatch": ".."}),
    "lros-128K": ("prefetch", {"label": "LROS (prefetch, 128K)", "hatch": ".."}),
    "lros-2M": ("prefetch", {"label": "LROS (prefetch, 2M)"}),
    "lros-128K-nopin": ("prefetch", {"label": "LROS (128K, no pin)", "hatch": "..", "color": NOPIN_COLOR}),
    "lros-2M-nopin": ("prefetch", {"label": "LROS (2M, no pin)", "color": NOPIN_COLOR}),
}
CONFIG_ORDER = list(CONFIG_STYLE)
DISK_LABEL = {"nvme": "NVMe", "emmc": "eMMC"}
DISK_ORDER = ["nvme", "emmc"]
METRIC_LABEL = {"s_tg": "Decode throughput (tok/s)",
                "s_pp": "Prefill throughput (tok/s)",
                "s": "Throughput (tok/s)"}
QUOTA_LABEL = "Memory quota (% of model size)"
# The sweep's figures get their own folder in bench/plots, apart from the
# raw logs in bench/out/swap/<timestamp>/. (common.py no longer names either
# folder; dir_path is this file's directory.)
result_dir = os.path.join(dir_path, "../../bench/out")
plots_dir = os.path.join(dir_path, "../../bench/plots")
swap_plots_dir = os.path.join(plots_dir, "swap")


def config_style(cfg):
    base, overrides = CONFIG_STYLE.get(cfg, (cfg, {}))
    st = dict(style_for(base))
    st.update(overrides)
    return st


def newest_sweep_csv():
    runs = sorted(glob.glob(os.path.join(result_dir, "swap", "*", "swap.csv")))
    if not runs:
        sys.exit("no bench/out/swap/<timestamp>/swap.csv; run `just swap_sweep` first")
    return runs[-1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", help="swap.csv files to pool (default: newest run)")
    p.add_argument("-o", "--out-dir", help="where to write the figures "
                   f"(default: {os.path.relpath(swap_plots_dir)})")
    p.add_argument("--metric", default="s_tg", choices=sorted(METRIC_LABEL),
                   help="batched-bench column to plot (default: s_tg, decode tok/s)")
    p.add_argument("--ntg", type=int, default=16,
                   help="use the table rows with this many generated tokens (default 16)")
    p.add_argument("--configs", default=None,
                   help="comma-separated configs to plot, in order (default: all present)")
    return p.parse_args()


def load(paths, metric, ntg):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["status"] == "ok"].dropna(subset=[metric])
    if ntg not in set(df["tg"]):
        ntg = int(df["tg"].max())
        print(f"no rows with ntg={ntg}; using ntg={ntg}", file=sys.stderr)
    df = df[df["tg"] == ntg].copy()
    df = df[~df["config"].isin(HIDDEN_CONFIGS)]
    df["budget"] = df["quota_pct"] / 100.0
    return df


def plot_model(df, disks, metric, out_dir, name, only_configs=None, single_column=False):
    """One figure with a panel per disk in `disks`, saved as `name` in
    `out_dir`; single_column fits the panels into one column width."""
    df = df[df["disk"].isin(disks)]
    configs = [c for c in CONFIG_ORDER if c in set(df["config"])] + \
              sorted(set(df["config"]) - set(CONFIG_ORDER))
    if only_configs:
        configs = [c for c in only_configs if c in configs]
        df = df[df["config"].isin(configs)]

    budgets = np.sort(df["budget"].unique())
    fits = budgets >= 1.0
    x = np.arange(len(budgets), dtype=float)
    width = 0.8 / len(configs)
    offset = {cfg: (i - len(configs) / 2 + 0.5) * width for i, cfg in enumerate(configs)}

    fig_w = figwidth_half if single_column else figwidth_half * len(disks)
    fig, axes = plt.subplots(1, len(disks), sharey=True, figsize=(fig_w, fig_height))
    axes = np.atleast_1d(axes)

    # The y axis is sized for the bars under memory pressure. A bar for a quota
    # the model fits in runs off the top of the axes and carries its value as a
    # label instead, so the interesting bars keep their resolution.
    under = df[df["budget"] < 1.0] if (~fits).any() else df
    ylim = float(under[metric].max()) * 1.45

    for ax, disk in zip(axes, disks):
        sub = df[df["disk"] == disk]
        # The cutoff between quotas the model does not fit in and those it does.
        if (~fits).any() and fits.any():
            cut = x[~fits][-1] + 0.5
            ax.axvline(cut, color="0.35", linestyle="--", linewidth=0.7, zorder=2)
            ax.text(cut - 0.08, 0.96, "does not fit" if single_column else "model does not fit",
                    transform=ax.get_xaxis_transform(), ha="right", va="top",
                    fontsize=FONTSIZE_ANNOTATION, color="0.35")

        # Bars are the median over runs, so one disturbed run cannot move a
        # cell; the error bars span the runs' min and max.
        means, tops = {}, {}
        for cfg in configs:
            st = config_style(cfg)
            g = sub[sub["config"] == cfg].groupby("budget")[metric] \
                .agg(["median", "min", "max", "count"]).reindex(budgets)
            vals = g["median"].to_numpy()
            multi = (g["count"] > 1).to_numpy()
            lo_err = np.where(multi, vals - g["min"].to_numpy(), 0)
            hi_err = np.where(multi, g["max"].to_numpy() - vals, 0)
            yerr = np.nan_to_num(np.vstack([lo_err, hi_err]))
            means[cfg] = vals
            tops[cfg] = np.nan_to_num(vals + yerr[1])
            ax.bar(x + offset[cfg], np.nan_to_num(vals), width=width, label=st["label"],
                   color=st["color"], edgecolor="black", linewidth=0.4, hatch=st["hatch"],
                   yerr=yerr, capsize=1.5, error_kw=dict(lw=0.5, capthick=0.5), zorder=3)
            # The bars that leave the axes get their value written on them.
            for k in np.flatnonzero(fits):
                if not np.isnan(vals[k]) and vals[k] + yerr[1, k] > ylim:
                    ax.text(x[k] + offset[cfg], ylim * 0.98, f"{vals[k]:.3g}",
                            ha="center", va="top", rotation=90, zorder=4,
                            fontsize=FONTSIZE_ANNOTATION,
                            bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                                      edgecolor="none", alpha=0.85))

        # The headline: prefetcher (2M) against mmap under memory pressure,
        # placed between the two bars it compares.
        if "lros-2M" in means and "linux-mmap" in means:
            for k in np.flatnonzero(~fits):
                hi, lo = means["lros-2M"][k], means["linux-mmap"][k]
                if np.isnan(hi) or np.isnan(lo) or lo <= 0:
                    continue
                # Above everything the label spans, error bars included.
                i0, i1 = sorted((configs.index("linux-mmap"), configs.index("lros-2M")))
                clear = max(tops[c][k] for c in configs[i0:i1 + 1])
                ax.text(x[k] + (offset["lros-2M"] + offset["linux-mmap"]) / 2,
                        clear + ylim * 0.03, speedup_label(hi / lo),
                        ha="center", va="bottom", fontsize=FONTSIZE_ANNOTATION, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{b * 100:g}%" for b in budgets], fontsize=FONTSIZE_TICK_LABEL)
        ax.set_xlim(-0.6, len(budgets) - 0.4)
        if not single_column:
            ax.set_xlabel(QUOTA_LABEL, fontsize=FONTSIZE_AXIS_LABEL)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        thin_spines(ax)
        ax.set_title(DISK_LABEL.get(disk, disk), fontsize=FONTSIZE_TITLE, pad=2)

    if single_column:
        # One label under both panels; per-panel labels would collide.
        fig.supxlabel(QUOTA_LABEL, fontsize=FONTSIZE_AXIS_LABEL, y=-0.02)
    axes[0].set_ylim(0, ylim)
    axes[0].set_ylabel(METRIC_LABEL[metric], fontsize=FONTSIZE_AXIS_LABEL)
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    better_hint(axes[0], higher=True, xy=(0.03, 0.86))

    handles, labels = axes[0].get_legend_handles_labels()
    # One row across a wide figure; a column holds three entries per row.
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0 if len(disks) == 1 else 0.98),
               ncol=min(3, len(labels)) if single_column else len(labels),
               fontsize=FONTSIZE_LEGEND, frameon=False,
               handlelength=1.4, handleheight=0.8, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.1)
    save(fig, name + ".pdf", out_dir, png=True)


args = parse_args()
paths = args.csv or [newest_sweep_csv()]
out_dir = args.out_dir or swap_plots_dir
data = load(paths, args.metric, args.ntg)
if data.empty:
    sys.exit("no usable rows (status ok with the requested metric)")
models = sorted(data["model"].unique())
only = args.configs.split(",") if args.configs else None
# Four layouts of the figure per model, under mem-weights-tp/<model>/: both
# disks side by side at one column per panel, both fitted into one column,
# and each disk alone in one column (the last two in single-column/).
FIGURE = "mem-weights-tp"
for m in models:
    sub = data[data["model"] == m]
    disks = [d for d in DISK_ORDER if d in set(sub["disk"])] + \
            sorted(set(sub["disk"]) - set(DISK_ORDER))
    model_dir = os.path.join(out_dir, FIGURE, os.path.splitext(m)[0])
    column_dir = os.path.join(model_dir, "single-column")
    plot_model(sub, disks, args.metric, model_dir, FIGURE, only_configs=only)
    if len(disks) > 1:
        plot_model(sub, disks, args.metric, column_dir, FIGURE, only_configs=only, single_column=True)
    for d in disks:
        plot_model(sub, [d], args.metric, column_dir, f"{FIGURE}_{d}", only_configs=only, single_column=True)
