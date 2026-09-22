#!/usr/bin/env python3
"""Numbers out of a run's console log: parse.py <mode> <log>, where mode is
serve (lros-serve totals, ok=1 once finished), requests (a row per finished
request) or bench (a row per llama-batched-bench table line)."""

import re
import sys

RE_TOTALS = re.compile(r"Total gen tokens:\s+(?P<gen>\d+), speed:\s+(?P<tps>[\d.]+) t/s")
RE_DONE = re.compile(
    r"\[t=\s*(?P<t>[\d.]+) s\] req\s+(?P<id>\d+) DONE\s+prompt\s+(?P<prompt>\d+) t, "
    r"gen\s+(?P<gen>\d+) t,\s+(?P<secs>[\d.]+) s,\s+(?P<tps>[\d.]+) t/s, "
    r"waited\s+(?P<waited>[\d.]+) s(?:, ttft\s+(?P<ttft>[\d.]+) s)?")
RE_DONE_SLOT = re.compile(
    r"\[t=\s*(?P<t>[\d.]+) s\] Client\s+(?P<slot>\d+): DONE\s+req\s+(?P<id>\d+) "
    r"\(priority (?P<prio>\d+)\) - prompt\s+(?P<prompt>\d+) t, gen\s+(?P<gen>\d+) t, "
    r"total\s+(?P<secs>[\d.]+) s, speed\s+(?P<tps>[\d.]+) t/s, "
    r"wait\s+(?P<waited>[\d.]+) s, preempted \d+x [\d.]+ s(?:, ttft\s+(?P<ttft>[\d.]+) s)?")
RE_STARTED = re.compile(
    r"\[t=\s*[\d.]+ s\] req\s+(?P<id>\d+) STARTED\s+seq\s+\d+ prio (?P<prio>\d+), "
    r"waited [\d.]+ s, task (?P<task>\d+) on (?P<cores>\d+) core\(s\)")
RE_BENCH = re.compile(
    r"^\|\s*(?P<pp>\d+)\s*\|\s*(?P<tg>\d+)\s*\|\s*(?P<b>\d+)\s*\|\s*(?P<nkv>\d+)\s*"
    r"\|\s*(?P<t_pp>[\d.]+)\s*\|\s*(?P<s_pp>[\d.]+)\s*\|\s*(?P<t_tg>[\d.]+)\s*"
    r"\|\s*(?P<s_tg>[\d.]+)\s*\|\s*(?P<t>[\d.]+)\s*\|\s*(?P<s>[\d.]+)\s*\|", re.M)

def serve(text):
    m = RE_TOTALS.search(text)
    out = {"gen_tokens": int(m["gen"]), "tps": float(m["tps"])} if m else {}
    out["ok"] = 1 if m else 0
    return [f"{k}={v}" for k, v in out.items()]


# The slots frontend's DONE line carries the priority; the lros one is joined
# with the request's STARTED line for it.
def requests(text):
    rows = []
    started = {int(m["id"]): m for m in RE_STARTED.finditer(text)}
    for m in RE_DONE_SLOT.finditer(text):
        rows.append((m["id"], m["prio"], -1, -1, m["t"], m["prompt"], m["gen"], m["secs"],
                     m["tps"], m["waited"], m["ttft"] or -1))
    if not rows:
        for m in RE_DONE.finditer(text):
            s = started.get(int(m["id"]))
            rows.append((m["id"], s["prio"] if s else -1, s["task"] if s else -1,
                         s["cores"] if s else -1, m["t"], m["prompt"], m["gen"], m["secs"],
                         m["tps"], m["waited"], m["ttft"] or -1))
    return [",".join(str(v) for v in r) for r in rows]


def bench(text):
    keys = ("pp", "tg", "b", "nkv", "t_pp", "s_pp", "t_tg", "s_tg", "t", "s")
    return [",".join(m[k] for k in keys) for m in RE_BENCH.finditer(text)]


if __name__ == "__main__":
    modes = {"serve": serve, "requests": requests, "bench": bench}
    if len(sys.argv) != 3 or sys.argv[1] not in modes:
        sys.exit(__doc__.strip())
    with open(sys.argv[2], "rb") as f:
        text = f.read().decode("utf-8", "replace")
    print("\n".join(modes[sys.argv[1]](text)))
