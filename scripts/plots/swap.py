#!/usr/bin/env python3
"""Weight paging under memory pressure, from a `just swap_sweep` run.

The real-data counterpart of mock_memory.py's mem-weights-tp.pdf: decode
throughput against the memory quota as a fraction of the model size, one bar
per paging configuration (llama.cpp mmap on Linux; LROS with a buffer per
tensor piece, with and without layer prefetch; LROS with fixed buffers), one
panel per disk the model was read from.

  python3 swap.py [swap.csv ...] [-o OUT_DIR] [--metric s_tg|s_pp|s] [--ntg N]

With no CSV the newest bench/out/swap/<timestamp>/swap.csv is used. Several
CSVs are pooled; each bar is the median over repeated runs, with error bars
from the runs' minimum to maximum. The figure comes in four layouts per model,
under bench/plots/swap/mem-weights-tp/<model>/ unless -o says otherwise, as
PDF and PNG:
  mem-weights-tp.pdf                        both disks, one column per panel
  single-column/mem-weights-tp.pdf          both disks fitted into one column
  single-column/mem-weights-tp_<disk>.pdf   one disk alone, one column
and with --pair A,B one figure of two models side by side on one disk (NVMe
unless --pair-disk says otherwise), each on its own y axis, under
mem-weights-tp/pairs/<A>__<B>.pdf, and fitted into one column under
mem-weights-tp/pairs/single-column/<A>__<B>.pdf. --quotas MODEL=110,50,20,10
draws only those quotas of MODEL.

A sweep run with SWAP_FRONTEND=cli (llama-cli on real text, one row per run)
is told apart by having no ntg=1 rows; all its rows are used, whatever --ntg
says. Where a model has rows from both frontends, the real-text ones are
plotted: batched-bench's single repeated token hides a mixture of experts'
routing.
"""
import glob
import sys

from common import *

# The sweep's config names, mapped onto the paper's series identities. LROS
# is never hatched (only the baselines are): the headline arm, per tensor with
# layer prefetch, is plain LROS; per tensor alone a deeper blue; the -nopin
# arms (--pin-percent 0, S3-FIFO over the whole cache) a lighter shade; the
# fixed buffer sizes of older sweeps differ by hatch still.
NOPIN_COLOR = lighten(LROS_COLOR, 0.45)
_h, _s, _v = rgb_to_hsv(to_rgb(LROS_COLOR))
DEEP_COLOR = hsv_to_rgb((_h, _s, _v * 0.6))
# Configurations left out of the figures even when the data has them; older
# sweeps carry arms the current one dropped. --configs overrides this.
HIDDEN_CONFIGS = {"lros-128K-nopin"}
CONFIG_STYLE = {
    "linux-mmap": ("mmap", {}),
    # The arm the speedup is printed for comes right after mmap, so that the
    # label sits between the two bars it compares.
    "lros-tensor-layer": ("prefetch", {"label": "LROS (per tensor, layer prefetch)"}),
    "lros-tensor": ("prefetch", {"label": "LROS (per tensor)", "color": DEEP_COLOR}),
    "lros-4K": ("prefetch", {"label": "LROS (prefetch, 4K)", "hatch": ".."}),
    "lros-128K": ("prefetch", {"label": "LROS (prefetch, 128K)", "hatch": ".."}),
    "lros-2M": ("prefetch", {"label": "LROS (prefetch, 2M)"}),
    "lros-128K-nopin": ("prefetch", {"label": "LROS (128K, no pin)", "hatch": "..", "color": NOPIN_COLOR}),
    "lros-2M-nopin": ("prefetch", {"label": "LROS (2M, no pin)", "color": NOPIN_COLOR}),
}
CONFIG_ORDER = list(CONFIG_STYLE)
# The arm the speedup over mmap is printed for: the first of these the data has.
HEADLINE_CONFIGS = ["lros-tensor-layer", "lros-tensor", "lros-2M"]
DISK_LABEL = {"nvme": "NVMe", "emmc": "eMMC"}
DISK_ORDER = ["nvme", "emmc"]
METRIC_LABEL = {"s_tg": "Decode throughput (tok/s)",
                "s_pp": "Prefill throughput (tok/s)",
                "s": "Throughput (tok/s)"}
QUOTA_LABEL = f"Memory quota ({PCT} of model size)"
# Panel titles of the pair figure, by model file name without .gguf.
MODEL_LABEL = {
    "Llama-3.2-1B-Instruct-f16": "Llama 3.2 1B (f16)",
    "Llama-3.2-1B-Instruct-Q4_K_M": "Llama 3.2 1B (Q4_K_M)",
    "qwen2.5-1.5b-instruct-fp16": "Qwen2.5 1.5B (f16)",
    "olmoe-1b-7b-0924-instruct-q4_k_m": "OLMoE 1B-7B (Q4_K_M)",
}
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
    p.add_argument("--height", type=float, default=fig_height,
                   help=f"figure height in inches (default {fig_height:g}, as every plot)")
    p.add_argument("--pair", action="append", default=[], metavar="A,B",
                   help="also draw models A (left) and B (right) side by side, file names "
                        "without .gguf; may be given more than once")
    p.add_argument("--pair-disk", default="nvme", help="the disk of the pair figures (default nvme)")
    p.add_argument("--models", default=None,
                   help="comma-separated models for the per-model figures, file names without "
                        ".gguf (default: all)")
    p.add_argument("--layouts", default="wide,column,disk",
                   help="comma-separated layouts of the per-model figures: wide (both disks), "
                        "column (both disks in one column), disk (each disk alone in one column), "
                        "or none, e.g. for the --pair figures alone (default: all)")
    p.add_argument("--no-png", action="store_true", help="write the PDFs only")
    p.add_argument("--quotas", action="append", default=[], metavar="MODEL=P,P,...",
                   help="draw only these quotas (percent) of MODEL (file name without .gguf), "
                        "in every figure; may be given once per model")
    return p.parse_args()


def load(paths, metric, ntg):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # batched-bench writes an ntg=1 row for every run, llama-cli never; per
    # sweep (its timestamp), as one CSV may hold several sweeps.
    bench_sweeps = set(df.loc[df["tg"] == 1, "timestamp"])
    df["frontend"] = np.where(df["timestamp"].isin(bench_sweeps), "batched-bench", "cli")
    df = df[df["status"] == "ok"].dropna(subset=[metric])
    bench = df["frontend"] == "batched-bench"
    if bench.any() and ntg not in set(df.loc[bench, "tg"]):
        print(f"no batched-bench rows with ntg={ntg}; using ntg={int(df.loc[bench, 'tg'].max())}",
              file=sys.stderr)
        ntg = int(df.loc[bench, "tg"].max())
    df = df[~bench | (df["tg"] == ntg)].copy()
    # Real text where a model has it.
    cli_models = set(df.loc[df["frontend"] == "cli", "model"])
    for m in sorted(cli_models & set(df.loc[df["frontend"] == "batched-bench", "model"])):
        print(f"{m}: plotting the llama-cli rows, not batched-bench's", file=sys.stderr)
    df = df[(df["frontend"] == "cli") | ~df["model"].isin(cli_models)]
    df = df[~df["config"].isin(HIDDEN_CONFIGS)]
    df["budget"] = df["quota_pct"] / 100.0
    return df


def axis_range(df, metric, budgets):
    """The y axis is sized for the bars under memory pressure. A bar for a
    quota the model fits in runs off the top of the axes and carries its value
    as a label instead, so the interesting bars keep their resolution."""
    under = df[df["budget"] < 1.0] if (budgets < 1.0).any() else df
    return float(under[metric].max()) * 1.45


def draw_panel(ax, sub, configs, metric, budgets, ylim, title, short_cut_label=False):
    """One panel's bars over `budgets`, largest quota on the left, on an axis
    of height `ylim`. Returns the x of the cutoff between the quotas the model
    fits in and those it does not, or None."""
    fits = budgets >= 1.0
    x = np.arange(len(budgets), dtype=float)
    width = 0.8 / len(configs)
    offset = {cfg: (i - len(configs) / 2 + 0.5) * width for i, cfg in enumerate(configs)}

    # The labels of the right-hand side start just after the cutoff.
    cut = x[fits][-1] + 0.5 if (~fits).any() and fits.any() else None
    if cut is not None:
        ax.axvline(cut, color="0.35", linestyle="--", linewidth=0.7, zorder=2)
        # A line down, under the "better" hint in the top right corner.
        ax.annotate("does not fit" if short_cut_label else "model does not fit",
                    xy=(cut + 0.08, 1.0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -FONTSIZE_ANNOTATION - 4), textcoords="offset points",
                    ha="left", va="top", fontsize=FONTSIZE_ANNOTATION, color="0.35")

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

    # The headline: the best LROS arm the data has against mmap under
    # memory pressure, placed between the two bars it compares.
    best = next((c for c in HEADLINE_CONFIGS if c in means), None)
    if best and "linux-mmap" in means:
        for k in np.flatnonzero(~fits):
            hi, lo = means[best][k], means["linux-mmap"][k]
            if np.isnan(hi) or np.isnan(lo) or lo <= 0:
                continue
            # Above everything the label spans, error bars included.
            i0, i1 = sorted((configs.index("linux-mmap"), configs.index(best)))
            clear = max(tops[c][k] for c in configs[i0:i1 + 1])
            ax.text(x[k] + (offset[best] + offset["linux-mmap"]) / 2,
                    clear + ylim * 0.03, speedup_label(hi / lo),
                    ha="center", va="bottom", fontsize=FONTSIZE_ANNOTATION, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b * 100:g}{PCT}" for b in budgets], fontsize=FONTSIZE_TICK_LABEL)
    ax.set_xlim(-0.6, len(budgets) - 0.4)
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=2)
    return cut


def plot_panels(panels, metric, out_dir, name, only_configs=None, single_column=False,
                height=fig_height, sharey=True, png=True):
    """One figure of `panels`, (rows, title) each, saved as `name` in
    `out_dir`. sharey: one y axis and one set of quotas for all panels (the
    disks of one model); otherwise each panel its own (different models).
    single_column fits the panels into one column width."""
    df = pd.concat([p for p, _ in panels])
    configs = [c for c in CONFIG_ORDER if c in set(df["config"])] + \
              sorted(set(df["config"]) - set(CONFIG_ORDER))
    if only_configs:
        configs = [c for c in only_configs if c in configs]
    panels = [(p[p["config"].isin(configs)], t) for p, t in panels]
    df = df[df["config"].isin(configs)]

    fig_w = figwidth_half if single_column else figwidth_half * len(panels)
    fig, axes = plt.subplots(1, len(panels), sharey=sharey, figsize=(fig_w, height))
    axes = np.atleast_1d(axes)

    # The largest quota on the left, down to the smallest.
    all_budgets = np.sort(df["budget"].unique())[::-1]
    for ax, (sub, title) in zip(axes, panels):
        budgets = all_budgets if sharey else np.sort(sub["budget"].unique())[::-1]
        ylim = axis_range(df if sharey else sub, metric, budgets)
        draw_panel(ax, sub, configs, metric, budgets, ylim, title,
                   short_cut_label=single_column)
        if not single_column:
            ax.set_xlabel(QUOTA_LABEL, fontsize=FONTSIZE_AXIS_LABEL)

    if single_column:
        # One label under both panels; per-panel labels would collide.
        fig.supxlabel(QUOTA_LABEL, fontsize=FONTSIZE_AXIS_LABEL, y=-0.02)
    axes[0].set_ylabel(METRIC_LABEL[metric], fontsize=FONTSIZE_AXIS_LABEL)
    for ax in axes:
        better_hint(ax, higher=True, inside=True)

    handles, labels = axes[0].get_legend_handles_labels()
    # One row across a wide figure; a column holds three entries per row,
    # four as two rows of two.
    column_ncol = 2 if len(labels) == 4 else min(3, len(labels))
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0 if len(panels) == 1 else 0.98),
               ncol=column_ncol if single_column else len(labels),
               fontsize=FONTSIZE_LEGEND, frameon=False,
               handlelength=1.4, handleheight=0.8, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.1)
    save(fig, name + ".pdf", out_dir, png=png)


def main():
    args = parse_args()
    paths = args.csv or [newest_sweep_csv()]
    out_dir = args.out_dir or swap_plots_dir
    data = load(paths, args.metric, args.ntg)
    for spec in args.quotas:
        model, _, pcts = spec.partition("=")
        keep = {float(p) for p in pcts.split(",")}
        of_model = data["model"] == model + ".gguf"
        data = data[~of_model | data["quota_pct"].astype(float).isin(keep)]
    if data.empty:
        sys.exit("no usable rows (status ok with the requested metric)")
    models = sorted(data["model"].unique())
    if args.models:
        models = [m for m in models if os.path.splitext(m)[0] in args.models.split(",")]
    layouts = set(args.layouts.split(",")) - {"none"}
    if layouts - {"wide", "column", "disk"}:
        sys.exit("--layouts takes wide, column, disk or none")
    png = not args.no_png
    only = args.configs.split(",") if args.configs else None
    # Up to four layouts of the figure per model, under mem-weights-tp/<model>/:
    # both disks side by side at one column per panel (wide), both fitted into
    # one column (column), and each disk alone in one column (disk; the last two
    # in single-column/).
    FIGURE = "mem-weights-tp"
    for m in models if layouts else []:
        sub = data[data["model"] == m]
        disks = [d for d in DISK_ORDER if d in set(sub["disk"])] + \
                sorted(set(sub["disk"]) - set(DISK_ORDER))
        by_disk = [(sub[sub["disk"] == d], DISK_LABEL.get(d, d)) for d in disks]
        model_dir = os.path.join(out_dir, FIGURE, os.path.splitext(m)[0])
        column_dir = os.path.join(model_dir, "single-column")
        if "wide" in layouts:
            plot_panels(by_disk, args.metric, model_dir, FIGURE, only_configs=only, height=args.height,
                        png=png)
        if "column" in layouts and len(disks) > 1:
            plot_panels(by_disk, args.metric, column_dir, FIGURE, only_configs=only, single_column=True,
                        height=args.height, png=png)
        if "disk" in layouts:
            for panel, d in zip(by_disk, disks):
                plot_panels([panel], args.metric, column_dir, f"{FIGURE}_{d}", only_configs=only,
                            single_column=True, height=args.height, png=png)

    # Two models side by side on one disk, each on its own y axis.
    for pair in args.pair:
        names = pair.split(",")
        if len(names) != 2:
            sys.exit(f"--pair {pair}: two model names, comma-separated")
        panels = []
        for n in names:
            rows = data[(data["model"] == n + ".gguf") & (data["disk"] == args.pair_disk)]
            if rows.empty:
                sys.exit(f"--pair: no rows for {n} on {args.pair_disk}")
            panels.append((rows, MODEL_LABEL.get(n, n)))
        pair_dir = os.path.join(out_dir, FIGURE, "pairs")
        plot_panels(panels, args.metric, pair_dir, "__".join(names),
                    only_configs=only, height=args.height, sharey=False, png=png)
        plot_panels(panels, args.metric, os.path.join(pair_dir, "single-column"), "__".join(names),
                    only_configs=only, single_column=True, height=args.height, sharey=False, png=png)


if __name__ == "__main__":
    main()
