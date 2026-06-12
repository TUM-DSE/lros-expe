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
            "cpu",
            # "cpu-novec",
            # "cuda",
            "rknn"
        ],
        "vm": [
            "cpu",
            # "cpu-novec"
        ],
        "lros": [
            "cpu",
            # "cpu-novec",
            # "vaccel",
            # "vaccel-novec"
        ],
    }
    conv={
        "lros": {
            "cpu"         : "LROS",
            "cpu-novec"   : "LROS (no-vec)",
            "vaccel-novec": "LROS: vAccel"
        },
        "process":{
            "cpu" : "Linux process",
            "cuda": "CUDA",
            "rknn": "RKNN"
        },
        "vm":{
            "cpu"      : "Linux VM",
            "cpu-novec": "Linux VM (no-vec)"
        }
    }
    path = os.path.join(result_dir, "main")
    batch_sizes = [1]
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

    data["ConfigPP"] = data["PP"].astype(int).astype(str)
    data["ConfigTG"] = data["TG"].astype(int).astype(str)

    n_hues = data["Spec"].nunique()

    # TTFT
    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))
    sns.barplot(ax=ax, data=data, x="ConfigPP", y="T_PPs",
                hue="Spec", palette=palette,
                edgecolor="black", linewidth=0.5)
    ax.set_ylabel("TTFT (s)")
    ax.set_xlabel("Prompt length (token)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=n_hues, title=None, fontsize=FONTSIZE, frameon=False)
    ax.annotate(lower_better_str, xy=(1, 1), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=FONTSIZE - 1, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bench-ttft.pdf"), format="pdf", bbox_inches="tight")

    # Throughput
    fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))
    sns.barplot(ax=ax, data=data, x="ConfigTG", y="S_TGt/s",
                hue="Spec", palette=palette,
                edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Throughput (token/s)")
    ax.set_xlabel("Generation length (token)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=n_hues, title=None, fontsize=FONTSIZE, frameon=False)
    ax.annotate(higher_better_str, xy=(1, 1), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=FONTSIZE - 1, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"bench-throughput.pdf"), format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
