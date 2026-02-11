#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import StringIO

import matplotlib
import pandas as pd

from common import *

import os
from datetime import datetime


def get_latest_file(directory, prefix, identifier, flavor, variant, batch_size):
    latest_file = None
    latest_time = None

    for filename in os.listdir(directory):
        parts = filename.rsplit('_', 3)
        if len(parts) != 4:
            continue  # Doesn't match expected format

        file_prefix, file_id, file_variant, batch_date_str = parts

        if file_prefix != prefix or file_id != identifier or file_variant != variant:
            continue

        split = batch_date_str.split('-',1)
        if len(split) != 2:
            continue  # Doesn't match expected format

        file_batch_size, date_str = split

        if file_batch_size != str(batch_size):
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
    dfs = []
    specs = {

        "process": [
            # "cpu",
            # "cpu-novec",
            "cuda",
            "rknn"
        ],
        "vm": [
            # "cpu",
            "cpu-novec"
        ],
        "lros": [
            # "cpu",
            "cpu-novec",
            # "vaccel",
            "vaccel-novec"
        ],
    }
    conv={
        "lros": {
            "cpu-novec" : "LROS: CPU",
            "vaccel-novec" : "LROS: vAccel"
        },
        "process":{
            "cuda": "CUDA",
            "rknn": "RKNN"
        },
        "vm":{
            "cpu-novec": "Linux VM"
        }
    }
    path = os.path.join(result_dir, "main")
    batch_sizes = [1,4]
    for spec, variants in specs.items():
        for var in variants:
            for b in batch_sizes:
                file = get_latest_file(path, "bench", spec, "out", var, b)
                if file:
                    with open(os.path.join(path, file)) as f:
                        table = "".join(l.replace(" ", "") for l in f if l.startswith("|") and not l.startswith("|-"))
                    df = pd.read_table(StringIO(table), sep="|", header=0, skipinitialspace=True, dtype=float,
                                       on_bad_lines="skip").dropna(axis=1, how='all')

                    df["Spec"] = conv[spec][var]
                    df["variant"]=var
                    df["system"]=spec
                    dfs.append(df)
                else:
                    dfs.append(pd.DataFrame())
    return pd.concat(dfs, axis=0).reset_index()


def main():

    # matplotlib.use('TkAgg')
    data = load_data()
    print(data)

    data["ConfigPP"] = data["B"].astype(int).astype(str)+", "+data["PP"].astype(int).astype(str)
    data["ConfigTG"] = data["B"].astype(int).astype(str)+", "+data["TG"].astype(int).astype(str)

    # TTFT

    # for val in data["B"].unique():
    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))
    plot = sns.barplot(ax=ax, data=data, x="ConfigPP", y="T_PPs",
                       hue="Spec"  # , style = "level_0"
                       , palette = palette
                       , edgecolor="black", linewidth=0.5
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    plot.set_yscale("log")
    # plot.set_xticks([0, 100, 200, 300])
    # plot.set_xticklabels([0, 100, 200, 300], fontsize=FONTSIZE)
    # plot.set_yticks([0, 50_000, 100_000])
    # plot.set_yticklabels([0, 50, 100], fontsize=FONTSIZE)
    ax.set_ylabel("TTFT (s)")
    ax.set_xlabel("Batch size, Prompt length (token)")
    ax.legend(loc="upper left", title=None, fontsize=FONTSIZE,
              # bbox_to_anchor=(0.3, 0.6),
              ncol=4,
              )
    ax.set_ylim(0,300)
    ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
    # plt.suptitle(f"Batch size: {val}", fontsize=FONTSIZE)
    # plt.grid()
    # plt.show()
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(plots_dir, f"bench-ttft.pdf"), format="pdf")

    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))
    plot = sns.barplot(ax=ax, data=data, x="ConfigTG", y="S_TGt/s",
                       hue="Spec"  # , style = "level_0"
                       , palette = palette
                       , edgecolor="black", linewidth=0.5
                       #  hue_order = get_order('vmcache')
                       # marker="H"
                       )
    plot.set_yscale("log")
    # plot.set_xticks([0, 100, 200, 300])
    # plot.set_xticklabels([0, 100, 200, 300], fontsize=FONTSIZE)
    # plot.set_yticks([0, 50_000, 100_000])
    # plot.set_yticklabels([0, 50, 100], fontsize=FONTSIZE)
    ax.set_ylabel("Throughput (token/s)")
    ax.set_xlabel("Batch size, Generation length (token)")
    ax.legend(loc="upper left", title=None, fontsize=FONTSIZE,
              # bbox_to_anchor=(0.3, 0.6),
              ncol=2,
              )
    ax.set_title(higher_better_str, fontsize=FONTSIZE, color="navy")
    # plt.suptitle(f"Batch size: {val}", fontsize=FONTSIZE)
    #plt.grid()
    # plt.show()
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(plots_dir, f"bench-throughput.pdf"), format="pdf")


if __name__ == "__main__":
    main()
