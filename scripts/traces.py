"""The scheduling traces, one per device and background prompt length.

    python3 traces.py <out dir> <paragraphs per background prompt, comma separated>
"""
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

arrivals = [(INT_START + i * INT_PERIOD, 0, INT_GEN, ASK) for i in range(INT_N)]
for dev, (period, n) in DEVICES.items():
    for k in fills:
        prompt = " ".join([INSTR] + [FILL] * k)
        reqs = arrivals + [(i * period, BG_PRIO, BG_GEN, prompt) for i in range(n)]
        with open(f"{out}/{dev}-k{k}.trace", "w") as f:
            f.write("# <t_ms> <model> <priority> <max_tokens> [prompt]\n")
            for t, prio, gen, p in sorted(reqs, key=lambda r: r[0]):
                f.write(f"{t} 0 {prio} {gen} {p}\n")
