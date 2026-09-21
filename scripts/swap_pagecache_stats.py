#!/usr/bin/env python3
"""Collect miniosv's page-cache counters from a swap_sweep run.

Every LROS console log ends with a "page cache" block when the image is
built with conf_pagecache_stats=1. This gathers those blocks into
<run dir>/pagecache.csv, one row per run, joined with the run's identity
and its decode throughput from swap.csv, and prints a per-cell summary
(medians over runs).

  python3 swap_pagecache_stats.py [bench/out/swap/<timestamp>]

Derived columns:
  passes        MiB read in / model size: how many times the model's bytes
                came off the disk (a lower bound on what the benchmark
                needs is about 20: warm-up, two prefills, 17 decode steps)
  adopted_pct   prefetches the consumer waited for while still in flight,
                as a share of all prefetches; the rest landed unconsumed
  evict_x       bytes evicted / bytes read in
  *_pct         share of total fault time, as printed by the guest
"""
import glob
import os
import re
import sys

import pandas as pd

RE_LINE1 = re.compile(r"(\d+) faults, (\d+) found resident, (\d+) MiB read in")
RE_LINE2 = re.compile(r"(\d+) buffers evicted \((\d+) MiB\); (\d+) prefetched, (\d+) published on "
                      r"completion, (\d+) adopted by a fault; (\d+) faults found no memory")
RE_LINE3 = re.compile(r"(\d+) us per fault, (\d+) trips round a spin loop")
RE_PHASE = re.compile(r"^\s*(whole fault|claim and submit|make room|wait for the device|"
                      r"install a prefetch|spin for another cpu|start the prefetches)"
                      r"\s+([\d.]+) ms\s+([\d.]+)%")
PHASE_KEY = {"whole fault": "whole", "claim and submit": "claim", "make room": "room",
             "wait for the device": "device", "install a prefetch": "adopt",
             "spin for another cpu": "spin", "start the prefetches": "prefetch"}


def parse_block(path):
    text = open(path, errors="replace").read().replace("\r", "")
    i = text.rfind("######## page cache")
    if i < 0:
        return None
    block = text[i:]
    row = {}
    for rx, keys in ((RE_LINE1, ["faults", "hits", "read_mib"]),
                     (RE_LINE2, ["evicted", "evicted_mib", "prefetched", "completed", "adopted", "nomem"]),
                     (RE_LINE3, ["us_per_fault", "spins"])):
        m = rx.search(block)
        if not m:
            return None
        row.update(zip(keys, map(int, m.groups())))
    for line in block.splitlines():
        m = RE_PHASE.match(line)
        if m:
            k = PHASE_KEY[m.group(1)]
            row[f"{k}_ms"] = float(m.group(2))
            row[f"{k}_pct"] = float(m.group(3))
    return row


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "bench", "out", "swap", "*", "swap.csv")))[-1])
    run_dir = os.path.realpath(run_dir)
    sweep = pd.read_csv(os.path.join(run_dir, "swap.csv"))
    sweep = sweep[sweep["config"].str.startswith("lros")]
    tg16 = sweep[sweep["tg"] == 16].set_index(["model", "disk", "config", "quota_pct", "run"])
    sizes = pd.read_csv(os.path.join(run_dir, "calibration.csv")).set_index("model")["model_bytes"]

    rows = []
    for (model, disk, config, quota, run), r in sweep.groupby(["model", "disk", "config", "quota_pct", "run"]):
        stats = parse_block(os.path.join(run_dir, "logs", model[:-5], disk, f"q{quota}",
                                         config, f"run{run}.out.txt"))
        if stats is None:
            continue
        stats.update(model=model, disk=disk, config=config, quota_pct=quota, run=run,
                     status=r["status"].iloc[0],
                     s_tg=tg16["s_tg"].get((model, disk, config, quota, run), float("nan")))
        rows.append(stats)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no page cache blocks found; was the image built with conf_pagecache_stats=1?")
    df["passes"] = df["read_mib"] * 2**20 / df["model"].map(sizes)
    df["adopted_pct"] = 100 * df["adopted"] / df["prefetched"].clip(lower=1)
    df["evict_x"] = df["evicted_mib"] / df["read_mib"].clip(lower=1)
    out = os.path.join(run_dir, "pagecache.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} runs with counters)\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["s_tg", "passes", "adopted_pct", "evict_x", "us_per_fault", "spin_pct", "device_pct",
            "room_pct", "adopt_pct"]
    med = df.groupby(["model", "disk", "quota_pct", "config"])[cols].median()
    order = ["lros-128K", "lros-2M", "lros-128K-nopin", "lros-2M-nopin"]
    for model, g in med.groupby(level=0):
        print(f"=== {model} (medians over runs; status!=ok runs included where counters exist)")
        t = g.droplevel(0).reset_index()
        t["config"] = pd.Categorical(t["config"], order)
        t = t.sort_values(["disk", "quota_pct", "config"], ascending=[False, False, True])
        print(t.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
        print()


if __name__ == "__main__":
    main()
