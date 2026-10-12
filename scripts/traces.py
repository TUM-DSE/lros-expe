"""The scheduling traces, one per device and background prompt length.

    python3 traces.py <out dir> <paragraphs per background prompt, comma separated>

A length of 0 writes the preemption traces instead: background prefills one at
a time, an interactive request arriving at a random point inside each
(<dev>-k0.trace), and the same interactive requests alone (<dev>-k0-alone.trace).
"""
import random
import sys

out, fills = sys.argv[1], [int(x) for x in sys.argv[2].split(",") if x]

FILL = ("The history of computing is long and its lessons are repetitive. "
        "Each generation rediscovers that memory is slow, that scheduling "
        "is a policy question, and that the cost of moving data usually "
        "exceeds the cost of computing on it.")   # about 47 tokens
INSTR = "Summarise the following passage in a few sentences."
ASK = "What is the capital of France, and why?"

BG_GEN, BG_PRIO = 16, 2
INT_START, INT_PERIOD, INT_N, INT_GEN = 4000, 5000, 6, 32

# Background period (ms) and count per device, so that the lengths run from
# under what the device drains to well over it: about 100 tok/s on the CPU,
# 500 on the GPU and 70 on the NPU.
DEVICES = {'cpu': (1500, 20), 'gpu': (1000, 30), 'npu': (3000, 10)}

# The preemption trace: the same background prompt on every device, one at a
# time with a period longer than its prefill and decode; the interactive
# request arrives uniformly within the first 80% of the prefill (ms).
PREEMPT_PARAS, PREEMPT_N, PREEMPT_GEN = 16, 10, 4
PREEMPT = {'cpu': (12000, 6000), 'gpu': (5000, 800), 'npu': (20000, 11000)}


def write(path, reqs):
    with open(path, "w") as f:
        f.write("# <t_ms> <model> <priority> <max_tokens> [prompt]\n")
        for t, prio, gen, p in sorted(reqs, key=lambda r: r[0]):
            f.write(f"{t} 0 {prio} {gen} {p}\n")


arrivals = [(INT_START + i * INT_PERIOD, 0, INT_GEN, ASK) for i in range(INT_N)]
for dev, (period, n) in DEVICES.items():
    for k in fills:
        if k == 0:
            every, prefill = PREEMPT[dev]
            rng = random.Random(1)
            bg = " ".join([INSTR] + [FILL] * PREEMPT_PARAS)
            background, interactive = [], []
            for i in range(PREEMPT_N):
                background.append((i * every, BG_PRIO, PREEMPT_GEN, bg))
                interactive.append((i * every + rng.randint(100, int(0.8 * prefill)), 0, PREEMPT_GEN, ASK))
            write(f"{out}/{dev}-k0.trace", background + interactive)
            write(f"{out}/{dev}-k0-alone.trace", interactive)
            continue
        prompt = " ".join([INSTR] + [FILL] * k)
        write(f"{out}/{dev}-k{k}.trace", arrivals + [(i * period, BG_PRIO, BG_GEN, prompt) for i in range(n)])
