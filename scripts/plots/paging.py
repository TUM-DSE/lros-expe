#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import StringIO

import pandas as pd

from common import *

import os
from datetime import datetime


def get_latest_file(directory, prefix, identifier, variant, spec, flavor):
    latest_file = None
    latest_time = None

    for filename in os.listdir(directory):
        parts = filename.rsplit('_', 4)
        if len(parts) != 5:
            continue  # Doesn't match expected format

        file_prefix, file_id, file_variant, file_spec, date_str = parts

        if file_prefix != prefix or file_id != identifier or file_variant != variant or file_spec != spec:
            continue

        split = date_str.split('.')

        if len(split) != 3:
            continue  # Doesn't match expected format

        # Clean date string (remove file extension if any)
        date_str, file_flavor, file_ext = split

        if file_flavor != flavor:
            continue

        try:
            # Parse ISO 8601 datetime
            dt = datetime.fromisoformat(date_str)
        except ValueError:
            continue  # Skip malformed dates

        if latest_time is None or dt > latest_time:
            latest_time = dt
            latest_file = filename

    return latest_file


def load_data() -> pd.DataFrame:
    eval_path = os.path.join(result_dir, "paging")
    dfs = []
    specs = ["prefetch", "mmap", "read", "lros"]
    variants = ["4G", "2G"]
    file = get_latest_file(eval_path, "bench", "paging", variants[0], specs[0], "out")
    for spec in specs:
        spec_file = file.replace(specs[0], spec)
        spec_file.replace("out", "err")
        df1s = []
        for variant in variants:
            variant_file_out = spec_file.replace(variants[0], variant)
            variant_file_err = variant_file_out.replace("out", "err")
            variant_out_path = os.path.join(eval_path, variant_file_out)
            variant_err_path = os.path.join(eval_path, variant_file_err)
            if os.path.exists(variant_out_path) and os.path.exists(variant_err_path):
                with open(variant_out_path) as f_out, open(variant_err_path) as f_err:
                    table = "".join(l.replace(" ", "") for l in f_out if l.startswith("|") and not l.startswith("|-"))
                    ls = [l for l in f_err if l.startswith("llama_perf_context_print")]
                    timing = {x.strip():float(y.strip().split("ms")[0])/1000 for x,y in (l.split(":", maxsplit=1)[1].strip().split("=") for l in ls)}
                if table:
                    df = pd.read_table(StringIO(table), sep="|", header=0, skipinitialspace=True, dtype=float,
                                   on_bad_lines="skip").dropna(axis=1, how='all')
                else:
                    df=pd.DataFrame()

                if "load time" in timing:
                    df["load_time"]=timing["load time"]
                    df["total_time"]=timing["total time"]
                # df["system"]=spec
                df1s.append(df)
            else:
                df1s.append(pd.DataFrame())
        dfs.append(pd.concat(df1s, keys=variants, axis=0))
    return pd.concat(dfs, keys=specs, axis=0).reset_index()


def main():
    data = load_data()
    print(data)
    data["prompt_time_cum"]=data["load_time"]+data["T_PPs"]
    data["tg_time_cum"]=data["prompt_time_cum"]+data["T_TGs"]

    fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
    plot = sns.barplot(ax=ax, data=data, x="level_1", y="load_time",
                       hue="level_0"  # , style = "level_0"
                        , edgecolor = "black", linewidth = 0.5
                        , palette = palette
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    for x in plot.patches:
        if x.get_hatch() is None:
            x.set_hatch(hatch_def[0])
            x.set_label("_")
    plot = sns.barplot(ax=ax, data=data, x="level_1", y="prompt_time_cum",
                       hue="level_0"  # , style = "level_0"
                       , palette = palette
                       , zorder = -1
                       ,edgecolor = "black", linewidth = 0.5,
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    for x in plot.patches:
        if x.get_hatch() is None: x.set_hatch(hatch_def[1])
    plot = sns.barplot(ax=ax, data=data, x="level_1", y="tg_time_cum",
                       hue="level_0"  # , style = "level_0"
                       , palette = palette
                       , zorder = -2
                       ,edgecolor = "black", linewidth = 0.5,
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    for x in plot.patches:
        if x.get_hatch() is None:
            x.set_hatch(hatch_def[2])
            x.set_label("_")
    l1=ax.legend(fontsize=FONTSIZE)

    # Hatch legend

    hidden = [ax.bar(0, 0, color="gray", hatch=hatch_def[0], label="Load"),
        ax.bar(0, 0, color="gray", hatch=hatch_def[1], label="Prompt"),
        ax.bar(0, 0, color="gray", hatch=hatch_def[2], label="TextGen")]

    plot.legend(handles=hidden, loc='upper right',fontsize=FONTSIZE)
    ax.add_artist(l1)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Memory size")

    plt.savefig(os.path.join(plots_dir, f"paging-new.pdf"), format="pdf", pad_inches=0, bbox_inches="tight")



if __name__ == "__main__":
    main()
