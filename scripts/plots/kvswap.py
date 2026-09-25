#!/usr/bin/env python3
"""KV-cache swap against recompute, from a `just kvswap_sweep` run.

Four figures per model, each with one panel per disk the KV was swapped to:

  mem-kv-cost     the time each strategy costs against the context length N:
                  recomputing (the prefill of N tokens, dashed) and the swap
                  round trip (out + in, solid), one line per configuration.
                  Log-log, so a fixed per-token cost is a line of slope 1.
  mem-kv-ratio    recompute / swap for each configuration; above 1 swapping
                  is the cheaper strategy, below 1 recomputing is, and where
                  a line crosses 1 is the context length at which the answer
                  flips.
  mem-kv-legs     the swap round trip (ms) as stacked bars, the swap out
                  below and the swap in on top, one bar per configuration,
                  grouped by N.
  mem-kv-partial  the partial swap out (`just kvswap_partial`): the same
                  stacked bars grouped by how much of the KV stayed resident,
                  at the experiment's one context length, with the state file
                  (which always moves the whole KV) in every group and the
                  recompute cost as a note. Drawn from the rows that have a
                  keep_pct; the state-file bar and the recompute come from
                  the ordinary rows at the same N, pooled in from the full
                  sweep's CSV or measured in the same run.
  mem-kv-legs-total, mem-kv-partial-total
                  the same two with one solid bar of the round trip instead
                  of its two legs.

Each figure comes in four layouts, under bench/plots/kvswap/<figure>/<model>/:
  <figure>.pdf                         both disks side by side, full width
  single-column/<figure>.pdf           both disks fitted into one column
  single-column/<figure>_<disk>.pdf    one disk alone, one column
all as PDF and PNG.

The swap round trip of a bulk restore is t_out + t_in; of a lazy restore
(nothing read ahead) it is t_out plus what the first decode step after the
swap cost beyond a decode step with everything resident, t_dec2 - t_dec.

  python3 kvswap.py [kvswap.csv ...] [-o OUT_DIR] [--configs a,b,...]

With no CSV the newest bench/out/kvswap/<timestamp>/kvswap.csv is used; several
CSVs are pooled. Each point is the median over repetitions and runs, with a
band from the minimum to the maximum. The crossover of every ratio line is
also printed.
"""
import functools
import glob
import sys

from common import *

LAZY_COLOR = lighten(LROS_COLOR, 0.45)
BIG_PAGE_COLOR = darken(LROS_COLOR)
# The sweep's config names, mapped onto the paper's series identities: Linux
# grey, LROS blue, the lazy restore a lighter LROS shade and the bigger page a
# darker one, so the bars tell them apart; the lines also differ by marker.
CONFIG_STYLE = {
    "linux-file":   {"label": "Userspace (state file)", "color": LINUX_COLOR, "marker": "s"},
    "lros-4K":      {"label": "LROS (paged, 4K)", "color": LROS_COLOR, "marker": "o"},
    "lros-64K":     {"label": "LROS (paged, 64K)", "color": BIG_PAGE_COLOR, "marker": "^"},
    "lros-2M":      {"label": "LROS (paged, 2M)", "color": darken(BIG_PAGE_COLOR), "marker": "D"},
    "lros-4K-lazy": {"label": "LROS (4K, lazy restore)", "color": LAZY_COLOR, "marker": "o"},
    "lros-64K-lazy": {"label": "LROS (64K, lazy restore)", "color": lighten(BIG_PAGE_COLOR, 0.45), "marker": "^"},
}
CONFIG_ORDER = list(CONFIG_STYLE)
# Configurations left out of every figure even when the data has them: the
# 4K bulk paging is a page-size point the 64K one supersedes (its lazy variant
# stays, being the fault-driven restore). --configs overrides this.
HIDDEN_CONFIGS = {"lros-4K"}
DISK_LABEL = {"nvme": "NVMe", "emmc": "eMMC"}
DISK_ORDER = ["nvme", "emmc"]
# The raw logs live in bench/out/kvswap/<timestamp>/ and the figures in
# bench/plots/kvswap/; common.py no longer names either folder, dir_path is
# this file's directory.
result_dir = os.path.join(dir_path, "../../bench/out")
plots_dir = os.path.join(dir_path, "../../bench/plots")
kvswap_plots_dir = os.path.join(plots_dir, "kvswap")


def config_style(cfg):
    return dict(CONFIG_STYLE.get(cfg, {"label": cfg, "color": "gray", "marker": "x"}))


def newest_sweep_csv():
    runs = sorted(glob.glob(os.path.join(result_dir, "kvswap", "*", "kvswap.csv")))
    if not runs:
        sys.exit("no bench/out/kvswap/<timestamp>/kvswap.csv; run `just kvswap_sweep` first")
    return runs[-1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", help="kvswap.csv files to pool (default: newest run)")
    p.add_argument("-o", "--out-dir", help="where to write the figures "
                   "(default: bench/plots/kvswap/)")
    p.add_argument("--configs", default=None,
                   help="comma-separated configs to plot, in this order (default: all)")
    return p.parse_args()


def load(paths):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"].dropna(subset=["n", "t_pp_ms"]).copy()
    df["n"] = df["n"].astype(int)
    # How much of the KV stayed resident on the swap out, for the rows of the
    # partial experiment; -1 for the ordinary runs, which swap everything in
    # kv-bench's ordinary layout (and for a CSV from before the column).
    if "keep_pct" not in df.columns:
        df["keep_pct"] = np.nan
    df["keep_pct"] = df["keep_pct"].fillna(-1).astype(int)
    # lros-4K-keep50 and lros-4K are the same mechanism at two settings.
    df["base"] = df["config"].str.replace(r"-keep\d+$", "", regex=True)
    df = df[~df["base"].isin(HIDDEN_CONFIGS)]
    # The first prefill of a run carries a one-off first-touch cost that the
    # frontend's warm-up does not absorb (7 s against 0.65 s for 64 tokens in
    # the guest), so the first repetition of a run's smallest length is left
    # out; the later repetitions are what the run measures.
    run_key = ["timestamp", "run", "model", "disk", "config"]
    first_n = df.groupby(run_key)["n"].transform("min")
    df = df[~((df["n"] == first_n) & (df["rep"] == 1))].copy()
    df["model"] = df["model"].str.replace(".gguf", "", regex=False)
    lazy = df["restore"] == "lazy"
    df["in_ms"] = np.where(lazy, (df["t_dec2_ms"] - df["t_dec_ms"]).clip(lower=0), df["t_in_ms"])
    df["swap_ms"] = df["t_out_ms"] + df["in_ms"]
    df["recompute_ms"] = df["t_pp_ms"]
    df["ratio"] = df["recompute_ms"] / df["swap_ms"]
    df["out_per_tok"] = df["t_out_ms"] / df["n"]
    df["in_per_tok"] = df["in_ms"] / df["n"]
    df["pp_per_tok"] = df["t_pp_ms"] / df["n"]
    return df


def crossover(ns, ratios):
    """The N at which the median ratio crosses 1, by log-log interpolation,
    or a word when it never does."""
    ns, ratios = np.asarray(ns, float), np.asarray(ratios, float)
    if (ratios > 1).all():
        return "swap cheaper at every N"
    if (ratios < 1).all():
        return "recompute cheaper at every N"
    for i in range(len(ns) - 1):
        a, b = ratios[i], ratios[i + 1]
        if (a - 1) * (b - 1) <= 0 and a != b:
            la, lb = np.log(ns[i]), np.log(ns[i + 1])
            t = (np.log(1) - np.log(a)) / (np.log(b) - np.log(a))
            n_star = np.exp(la + t * (lb - la))
            side = "swap cheaper above" if b > a else "recompute cheaper above"
            return "%s N = %.0f" % (side, n_star)
    return "crosses 1 more than once"


def agg(sub, col):
    g = sub.groupby("n")[col].agg(["median", "min", "max"])
    return g.index.to_numpy(), g["median"].to_numpy(), g["min"].to_numpy(), g["max"].to_numpy()


def ordered(values, order):
    present = set(values)
    return [v for v in order if v in present] + sorted(present - set(order))


def leg_patches():
    """Legend entries for the two legs of a stacked bar: the lower, solid
    part is the swap out, the hatched part on top the swap in."""
    return ([mpatches.Patch(facecolor="0.5", edgecolor="black", linewidth=0.4),
             mpatches.Patch(facecolor="0.85", edgecolor="black", linewidth=0.4, hatch="//")],
            ["swap out (lower)", "swap in (upper, hatched)"])


def stacked_bars(ax, x, off, width, out, inn, st, label, split=True):
    """A bar per x for the swap round trip: its two legs stacked when
    `split`, the swap out below and the swap in hatched on top; one solid
    bar of the total otherwise."""
    if not split:
        ax.bar(x + off, out + inn, width=width, color=st["color"],
               edgecolor="black", linewidth=0.4, label=label, zorder=3)
        return
    ax.bar(x + off, out, width=width, color=st["color"],
           edgecolor="black", linewidth=0.4, label=label, zorder=3)
    ax.bar(x + off, inn, width=width, bottom=out, color=lighten(st["color"], 0.6),
           edgecolor="black", linewidth=0.4, hatch="//", zorder=3)


def style_axes(ax, disk, xlabel, title=None, compact=False):
    """`compact` is two panels in one column: only the disk as the title and
    a short x label, or the two panels' text collides."""
    if compact:
        xlabel = xlabel.replace("Context length N (tokens)", "N (tokens)")
        title = DISK_LABEL.get(disk, disk)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_title(title or "KV swapped to %s" % DISK_LABEL.get(disk, disk), fontsize=FONTSIZE_TITLE)
    ax.tick_params(labelsize=FONTSIZE_TICK_LABEL)
    thin_spines(ax)


# --- the four figures: each draws its panels onto the axes it is given -------
# A drawer takes (df, disks, axes, configs, compact) and returns the legend's
# (handles, labels); the layout, the legend and the saving are shared below.

def draw_cost(df, disks, axes, configs, compact=False):
    for ax, disk in zip(axes, disks):
        sub = df[df["disk"] == disk]
        # The recompute cost does not depend on the disk or the page size:
        # one dashed line per system (the guest and the host prefill differ).
        for system, color, label in (("lros", LROS_COLOR, "recompute (LROS prefill)"),
                                     ("linux", LINUX_COLOR, "recompute (userspace prefill)")):
            s = sub[sub["config"].str.startswith(system)]
            if s.empty:
                continue
            ns, med, lo, hi = agg(s, "recompute_ms")
            ax.plot(ns, med, linestyle="--", color=color, linewidth=1.0, label=label, zorder=3)
        for cfg in configs:
            s = sub[sub["config"] == cfg]
            if s.empty:
                continue
            st = config_style(cfg)
            ns, med, lo, hi = agg(s, "swap_ms")
            ax.fill_between(ns, lo, hi, color=st["color"], alpha=0.15, linewidth=0, zorder=2)
            ax.plot(ns, med, color=st["color"], marker=st["marker"], markersize=3,
                    linewidth=1.0, label=st["label"] + " swap", zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
        style_axes(ax, disk, "Context length N (tokens)", compact=compact)
    axes[0].set_ylabel("Time (ms)", fontsize=FONTSIZE_AXIS_LABEL)
    return axes[0].get_legend_handles_labels()


def draw_ratio(df, disks, axes, configs, compact=False):
    for ax, disk in zip(axes, disks):
        sub = df[df["disk"] == disk]
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.7, zorder=2)
        ax.text(0.98, 1.0, "swap cheaper above", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=FONTSIZE_ANNOTATION, color="0.35")
        for cfg in configs:
            s = sub[sub["config"] == cfg]
            if s.empty:
                continue
            st = config_style(cfg)
            ns, med, lo, hi = agg(s, "ratio")
            ax.fill_between(ns, lo, hi, color=st["color"], alpha=0.15, linewidth=0, zorder=2)
            ax.plot(ns, med, color=st["color"], marker=st["marker"], markersize=3,
                    linewidth=1.0, label=st["label"], zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
        style_axes(ax, disk, "Context length N (tokens)", compact=compact)
    axes[0].set_ylabel("Recompute time / swap time", fontsize=FONTSIZE_AXIS_LABEL)
    return axes[0].get_legend_handles_labels()


def draw_legs(df, disks, axes, configs, compact=False, split=True):
    """The swap round trip against N as bars, one per configuration, split
    into its two legs or (split=False) one solid bar of the total."""
    ns = np.sort(df["n"].unique())
    x = np.arange(len(ns), dtype=float)
    width = 0.8 / len(configs)
    for ax, disk in zip(axes, disks):
        sub = df[df["disk"] == disk]
        for i, cfg in enumerate(configs):
            st = config_style(cfg)
            g = sub[sub["config"] == cfg].groupby("n")[["t_out_ms", "in_ms"]] \
                .median().reindex(ns)
            off = (i - len(configs) / 2 + 0.5) * width
            stacked_bars(ax, x, off, width, g["t_out_ms"].to_numpy(), g["in_ms"].to_numpy(),
                         st, st["label"], split)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns], fontsize=FONTSIZE_TICK_LABEL)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5, zorder=0)
        style_axes(ax, disk, "Context length N (tokens)", compact=compact)
    axes[0].set_ylabel("Swap round trip (ms)", fontsize=FONTSIZE_AXIS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    if split:
        h, l = leg_patches()
        handles, labels = handles + h, labels + l
    return handles, labels


def draw_partial(df, disks, axes, configs, compact=False, split=True):
    """The partial swap out: the round trip of the KV when 0, 10, 50 or 90%
    of it may stay resident, so a bar's height against the 0% bar is what
    keeping that share saves; the legs split or (split=False) one solid bar.
    Each disk on its own y axis: eMMC is an order of magnitude slower, and on
    a shared axis the NVMe bars would be slivers."""
    partial = df[df["keep_pct"] >= 0]
    keeps = np.sort(partial["keep_pct"].unique())
    ns = sorted(partial["n"].unique())
    # The state file has no partial form: it always moves the whole KV, so
    # its full-swap rows at the same N stand in every group.
    ref = df[(df["keep_pct"] < 0) & df["n"].isin(ns)]
    file_ref = ref[ref["config"] == "linux-file"]
    bases = [c for c in ordered(partial["base"], CONFIG_ORDER) if c in configs or c not in CONFIG_STYLE]
    if not file_ref.empty:
        bases = ["linux-file"] + bases

    x = np.arange(len(keeps), dtype=float)
    width = 0.8 / len(bases)
    for ax, disk in zip(axes, disks):
        sub = partial[partial["disk"] == disk]
        top = 0.0   # the tallest bar of this panel; its axis is sized to it
        for i, base in enumerate(bases):
            st = config_style(base)
            if base == "linux-file":
                r = file_ref[file_ref["disk"] == disk][["t_out_ms", "in_ms"]].median()
                out = np.full(len(keeps), r["t_out_ms"])
                inn = np.full(len(keeps), r["in_ms"])
            else:
                g = sub[sub["base"] == base].groupby("keep_pct")[["t_out_ms", "in_ms"]] \
                    .median().reindex(keeps)
                out, inn = g["t_out_ms"].to_numpy(), g["in_ms"].to_numpy()
            top = max(top, float(np.nanmax(out + inn)))
            off = (i - len(bases) / 2 + 0.5) * width
            stacked_bars(ax, x, off, width, out, inn, st, st["label"], split)

        # The recompute, the same whatever the quota: a full prefill of N.
        # Far above every bar, so it is stated rather than drawn.
        notes = []
        for system, label in (("lros", "LROS"), ("linux", "userspace")):
            rows = pd.concat([sub, ref[ref["disk"] == disk]])
            rows = rows[rows["config"].str.startswith(system)]
            if not rows.empty:
                notes.append("%s %.1f" % (label, rows["t_pp_ms"].median() / 1000.0))
        if notes:
            text = ("recompute:\n%s s" % ",\n".join(notes)) if compact \
                else "recompute: %s s" % ", ".join(notes)
            ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top",
                    fontsize=FONTSIZE_ANNOTATION, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels(["%d%%" % k for k in keeps], fontsize=FONTSIZE_TICK_LABEL)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5, zorder=0)
        style_axes(ax, disk, "KV kept resident", compact=compact)
        # Room above the tallest bar for the recompute note, three lines of
        # it in the compact layout.
        ax.set_ylim(0, top * (1.45 if compact else 1.2))
    axes[0].set_ylabel("Swap round trip (ms)", fontsize=FONTSIZE_AXIS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    if split:
        h, l = leg_patches()
        handles, labels = handles + h, labels + l
    return handles, labels


FIGURES = {
    # name: (drawer, panels share a y axis). The names are the paper's, with
    # the mem-kv- prefix that groups them next to mem-weights-tp. The -total
    # variants draw one solid bar of the round trip instead of its two legs.
    "mem-kv-cost":          (draw_cost, True),
    "mem-kv-ratio":         (draw_ratio, True),
    "mem-kv-legs":          (draw_legs, True),
    "mem-kv-legs-total":    (functools.partial(draw_legs, split=False), True),
    "mem-kv-partial":       (draw_partial, False),
    "mem-kv-partial-total": (functools.partial(draw_partial, split=False), False),
}


def render(kind, df, model, out_dir, configs):
    """One figure in its four layouts: both disks full width, both fitted
    into one column, and each disk alone in one column."""
    drawer, sharey = FIGURES[kind]
    disks = ordered(df["disk"], DISK_ORDER)
    kind_dir = os.path.join(out_dir, kind, model)
    layouts = [(disks, False, kind_dir, kind)]
    if len(disks) > 1:
        layouts.append((disks, True, os.path.join(kind_dir, "single-column"), kind))
    for disk in disks:
        layouts.append(([disk], True, os.path.join(kind_dir, "single-column"), "%s_%s" % (kind, disk)))

    for panels, single_column, directory, name in layouts:
        fig_w = figwidth_half if single_column else figwidth_half * len(panels)
        fig, axes = plt.subplots(1, len(panels), sharey=sharey, figsize=(fig_w, fig_height))
        axes = np.atleast_1d(axes)
        compact = single_column and len(panels) > 1
        handles, labels = drawer(df[df["disk"].isin(panels)], panels, axes, configs, compact)
        # The wide layout takes the whole legend in one row of four; a column
        # holds two entries per row.
        ncol = 2 if single_column else min(4, len(labels))
        rows = -(-len(labels) // ncol)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0 + 0.02 * rows),
                   ncol=ncol, fontsize=FONTSIZE_LEGEND, frameon=False)
        save(fig, name + ".pdf", directory, png=True)


def main():
    args = parse_args()
    paths = args.csv or [newest_sweep_csv()]
    df = load(paths)
    if df.empty:
        sys.exit("no successful rows in %s" % ", ".join(paths))
    out_dir = args.out_dir or kvswap_plots_dir
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        configs = ordered(sub[sub["keep_pct"] < 0]["config"], CONFIG_ORDER)
        if args.configs:
            configs = [c for c in args.configs.split(",")]
        full = sub[(sub["keep_pct"] < 0) & sub["config"].isin(configs)]
        if not full.empty:
            print("%s:" % model)
            for disk in ordered(full["disk"], DISK_ORDER):
                for cfg in configs:
                    s = full[(full["disk"] == disk) & (full["config"] == cfg)]
                    if not s.empty:
                        ns, med, lo, hi = agg(s, "ratio")
                        print("  %-5s %-16s %s" % (disk, cfg, crossover(ns, med)))
            for kind in ("mem-kv-cost", "mem-kv-ratio", "mem-kv-legs", "mem-kv-legs-total"):
                render(kind, full, model, out_dir, configs)
        # All rows: the partial figure takes its state-file bar and the
        # recompute reference from the ordinary runs at the same N.
        if (sub["keep_pct"] >= 0).any():
            render("mem-kv-partial", sub, model, out_dir, configs)
            render("mem-kv-partial-total", sub, model, out_dir, configs)


if __name__ == "__main__":
    main()
