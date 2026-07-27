#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import StringIO

import pandas as pd
from pandas import DataFrame

from common import *

import os
from datetime import datetime


def get_latest_run_dir(directory):
    latest_dir = None
    latest_time = None

    for name in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, name)):
            continue

        try:
            # Run dirs are named by their ISO 8601 start time
            dt = datetime.fromisoformat(name)
        except ValueError:
            continue  # Skip non-run dirs

        if latest_time is None or dt > latest_time:
            latest_time = dt
            latest_dir = name

    if latest_dir is None:
        raise FileNotFoundError(
            f"No timestamped run dirs under {directory}; run `just paging` first.")
    return os.path.join(directory, latest_dir)


def load_data() -> pd.DataFrame:
    eval_path = get_latest_run_dir(os.path.join(result_dir, "paging"))
    dfs = []
    specs = {
        "mmap": "mmap",
        "read": "read",
        "lrosprefetch": "LROS",
    }

    variants = {
        "16G": "16G",
        "2G": "2G",
        "512M": "512M",
    }

    for spec in specs.keys():
        df1s = []
        for variant in variants.keys():
            variant_out_path = os.path.join(eval_path, f"bench_paging_{variant}_{spec}.out.txt")
            variant_err_path = os.path.join(eval_path, f"bench_paging_{variant}_{spec}.err.txt")
            if os.path.exists(variant_out_path) and os.path.exists(variant_err_path):
                with open(variant_out_path) as f_out, open(variant_err_path) as f_err:
                    table = "".join(l.replace(" ", "") for l in f_out if l.startswith("|") and not l.startswith("|-"))
                    ls = [l for l in f_err if l.startswith("llama_perf_context_print")]
                    timing = {x.strip(): float(y.strip().split("ms")[0]) / 1000 for x, y in
                              (l.split(":", maxsplit=1)[1].strip().split("=") for l in ls)}
                if table:
                    df = pd.read_table(StringIO(table), sep="|", header=0, skipinitialspace=True, dtype=float,
                                       on_bad_lines="skip").dropna(axis=1, how='all').head(1)
                else:
                    df = pd.DataFrame()

                if "load time" in timing:
                    df["load_time"] = timing["load time"]
                    df["total_time"] = timing["total time"]
                # df["system"]=spec
                df1s.append(df)
            else:
                df1s.append(pd.DataFrame())
        dfs.append(pd.concat(df1s, keys=variants.values(), axis=0))
    return pd.concat(dfs, keys=specs.values(), axis=0).reset_index()


def main():
    data = load_data()
    data["prompt_time_cum"] = data["load_time"] + data["T_PPs"]
    data["tg_time_cum"] = data["prompt_time_cum"] + data["T_TGs"]

    fig, (sax1, sax2) = plt.subplots(1, 2, figsize=(figwidth_full, fig_height), layout="constrained")
    make_ttft_plot(sax1, data)
    sax1.set_title("(a) " + sax1.get_title())

    fig1, ax = plt.subplots(figsize=(figwidth_third, 1.6))
    make_ttft_plot(ax, data)
    # Extra headroom so the two legends fit above the bars at third width
    ax.set_ylim(0,65)
    fig1.tight_layout(pad=0.1)
    legend = fig1.legend()
    legend.remove()
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.get_legend()
    leg1=ax.legend(handles[:-2], labels[:-2], loc="upper left")
    for patch in leg1.legend_handles:
        patch.set_hatch('')
    ax.add_artist(leg)
    fig1.savefig(os.path.join(plots_dir, f"paging-ttft.pdf"), format="pdf")

    # -------------
    make_throughput_plot(sax2, data)
    sax2.set_title("(b) " + sax2.get_title())

    fig1, ax = plt.subplots(figsize=(figwidth_third, 1.6))
    make_throughput_plot(ax, data)
    ax.legend(fontsize=FONTSIZE, loc='upper right', ncol=1)
    fig1.tight_layout(pad=0.1)
    fig1.savefig(os.path.join(plots_dir, f"paging-tp.pdf"), format="pdf")

    fig1, ax = plt.subplots(figsize=(figwidth_third, 1.6))
    make_throughput_plot(ax, data, log=True, labels=True)
    ax.legend(fontsize=FONTSIZE, loc='upper right', ncol=1)
    fig1.tight_layout(pad=0.1)
    fig1.savefig(os.path.join(plots_dir, f"paging-tp-log.pdf"), format="pdf")

    fig1 = make_throughput_broken_fig(data)
    fig1.savefig(os.path.join(plots_dir, f"paging-tp-broken.pdf"), format="pdf")

    handles, labels = sax2.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=FONTSIZE, loc='outside lower center', ncol=4)
    # fig.tight_layout()
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=2 / 72, hspace=0,
                                wspace=0.1)
    fig.savefig(os.path.join(plots_dir, f"paging.pdf"), format="pdf")


def label_bars(ax):
    for c in ax.containers:
        heights = [b.get_height() for b in c]
        ax.bar_label(c, labels=[f"{h:.2f}" if h == h else "" for h in heights],
                     fontsize=FONTSIZE - 1, padding=1)


def make_throughput_plot(ax, data: DataFrame, log=False, labels=False):
    plot = sns.barplot(ax=ax, data=data, x="level_1", y="S_TGt/s",
                       hue="level_0"  # , style = "level_0"
                       , palette=palette[:3]
                       , zorder=-2
                       , edgecolor="black", linewidth=0.5,
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    if log:
        plot.set_yscale("log")
        # Bottom just below the smallest bar (mmap out-of-memory ~0.09),
        # top leaves headroom for bar labels and the legend
        ax.set_ylim(0.05, 30)
    if labels:
        label_bars(ax)
    # ax.legend(fontsize=FONTSIZE, loc='upper right', ncol=2)
    ax.get_legend().remove()
    ax.set_xlabel(None)
    ax.set_ylabel("Throughput (tk/s)")
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    ax.set_title(f"{higher_better_str}", fontsize=FONTSIZE, color="navy")


def make_throughput_broken_fig(data: DataFrame):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True,
                                         figsize=(figwidth_third, 1.6),
                                         height_ratios=[1, 1], layout="constrained")
    for ax in (ax_top, ax_bot):
        sns.barplot(ax=ax, data=data, x="level_1", y="S_TGt/s",
                    hue="level_0"
                    , palette=palette[:3]
                    , zorder=-2
                    , edgecolor="black", linewidth=0.5,
                    )
        ax.get_legend().remove()
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    ax_top.set_ylim(5.5)
    ax_bot.set_ylim(0, 0.7)

    # Axis break: hide the facing spines and draw diagonal break marks
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    d = 0.5
    break_kw = dict(marker=[(-1, -d), (1, d)], markersize=5, linestyle="none",
                    color="black", mec="black", mew=0.8, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **break_kw)

    ax_top.legend(fontsize=FONTSIZE, loc='upper right', ncol=1)
    ax_top.set_title(f"{higher_better_str}", fontsize=FONTSIZE, color="navy")
    fig.supylabel("Throughput (tk/s)", fontsize=FONTSIZE)
    # tight_layout doesn't reserve space for supylabel; constrained layout does
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, hspace=0.08)
    return fig


def make_ttft_plot(ax, data: DataFrame):
    plot = sns.barplot(ax=ax, data=data, x="level_1", y="load_time",
                       hue="level_0"  # , style = "level_0"
                       , edgecolor="black", linewidth=0.5
                       , palette=palette[:3]
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    for x in plot.patches:
        if x.get_hatch() is None:
            x.set_hatch(hatch_def[0])
            x.set_label("_")
    plot = sns.barplot(ax=ax, data=data, x="level_1", y="prompt_time_cum",
                       hue="level_0"  # , style = "level_0"
                       , palette=palette[:3]
                       , zorder=-1
                       , edgecolor="black", linewidth=0.5,
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    for x in plot.patches:
        if x.get_hatch() is None: x.set_hatch(hatch_def[5])

    # l1 = ax.legend(fontsize=FONTSIZE, loc='upper left')

    # Hatch legend

    hidden = [ax.bar(0, 0, color="gray", hatch=hatch_def[0], label="Load"),
              ax.bar(0, 0, color="gray", hatch=hatch_def[5], label="Prompt"),
              # ax.bar(0, 0, color="gray", hatch=hatch_def[2], label="TextGen")
              ]

    plot.legend(handles=hidden, loc='upper right', fontsize=FONTSIZE)
    # ax.add_artist(l1)

    ax.set_xlabel(None)
    ax.set_ylabel("TTFT (s)")
    ax.set_ylim(0, 48)
    ax.set_title(f"{lower_better_str}", fontsize=FONTSIZE, color="navy")


if __name__ == "__main__":
    main()
