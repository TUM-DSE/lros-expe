#!/usr/bin/env python3
"""One interactive request arriving into background streams on a llama-server;
--n-bg 0 is the uncontended floor. Prints CSV rows: arm,trial,ttft_ms,
tpot_mean_ms,tpot_p95_ms,late_frac,int_tokens,bg_before,bg_during"""
import argparse, http.client, json, threading, time, sys

def sse(port, payload, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/completion", json.dumps(payload), {"Content-Type": "application/json"})
    return c, c.getresponse()


class Background(threading.Thread):
    """A stream that keeps generating, recording when each token arrives."""

    def __init__(self, port, slot, tokens, prompt):
        super().__init__(daemon=True)
        self.port, self.slot, self.tokens, self.prompt = port, slot, tokens, prompt
        self.stamps = []
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                c, r = sse(self.port, {
                    "prompt": self.prompt, "n_predict": self.tokens, "stream": True,
                    "id_slot": self.slot, "cache_prompt": False, "ignore_eos": True,
                    "temperature": 0})
            except OSError:
                return
            while not self.stop:
                line = r.readline()
                if not line:
                    break
                if line.startswith(b"data: ") and b'"content"' in line:
                    self.stamps.append(time.monotonic())
            try:
                c.close()
            except OSError:
                pass


def start_bg(a):
    s = [Background(a.port, i + 1, a.bg_tokens,
                    f"Document {i}. " + "alpha " * a.bg_prompt + "Summarise it.")
         for i in range(a.n_bg)]
    for t in s:
        t.start()
    return s


def stop_bg(streams):
    for t in streams:
        t.stop = True
    time.sleep(2.0)


def wait_decoding(streams, timeout):
    """All background slots past their own prompt and producing tokens."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if all(t.stamps for t in streams):
            return True
        time.sleep(0.5)
    return False


def rate(streams, base, t0, t1):
    n = sum(sum(1 for s in t.stamps[b:] if t0 <= s <= t1) for t, b in zip(streams, base))
    return n / (t1 - t0) if t1 > t0 else 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--arrival", choices=["steady", "prefill"], default="steady")
ap.add_argument("--n-bg", type=int, default=7, help="0 measures the uncontended floor")
ap.add_argument("--int-prompt", type=int, default=512)
ap.add_argument("--int-tokens", type=int, default=256)
ap.add_argument("--bg-prompt", type=int, default=512)
ap.add_argument("--bg-tokens", type=int, default=16384)
ap.add_argument("--deadline-ms", type=float, default=208.0, help="4.8 tok/s reading speed")
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--settle", type=float, default=8.0, help="steady: after all slots decode")
ap.add_argument("--lead", type=float, default=1.5, help="prefill: into the background prompt")
ap.add_argument("--ready-timeout", type=float, default=180.0)
ap.add_argument("--gap", type=float, default=8.0)
a = ap.parse_args()

bg = start_bg(a)
if bg and not wait_decoding(bg, a.ready_timeout):
    print(f"# not all {a.n_bg} background streams reached decode", file=sys.stderr)

print("arm,trial,ttft_ms,tpot_mean_ms,tpot_p95_ms,late_frac,int_tokens,bg_before,bg_during")
prompt = "Here is a note. " + "beta " * a.int_prompt + "What is the capital of France?"
for trial in range(a.trials):
    if bg:
        if a.arrival == "prefill":
            stop_bg(bg)
            bg = start_bg(a)
            time.sleep(a.lead)
        else:
            if not wait_decoding(bg, a.ready_timeout):
                print(f"# trial {trial}: background not all decoding", file=sys.stderr)
            time.sleep(a.settle)

    base = [len(t.stamps) for t in bg]
    t_probe = time.monotonic()
    time.sleep(1.0)
    bg_before = rate(bg, base, t_probe, time.monotonic()) if bg else 0.0

    base = [len(t.stamps) for t in bg]
    t0 = time.monotonic()
    try:
        c, r = sse(a.port, {"prompt": prompt, "n_predict": a.int_tokens, "stream": True,
                            "id_slot": 0, "cache_prompt": False, "ignore_eos": True,
                            "temperature": 0})
    except OSError as e:
        print(f"# interactive request failed: {e}", file=sys.stderr)
        continue
    stamps = []
    while True:
        line = r.readline()
        if not line:
            break
        if line.startswith(b"data: ") and b'"content"' in line:
            stamps.append(time.monotonic())
    try:
        c.close()
    except OSError:
        pass
    if len(stamps) < 2:
        print(f"# interactive request returned {len(stamps)} tokens", file=sys.stderr)
        continue

    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
    gaps_sorted = sorted(gaps)
    late = sum(1 for g in gaps if g > a.deadline_ms) / len(gaps)
    bg_during = rate(bg, base, stamps[0], stamps[-1]) if bg else 0.0
    print(f"{a.arm},{trial},{(stamps[0] - t0) * 1000:.1f},"
          f"{sum(gaps) / len(gaps):.2f},{gaps_sorted[int(0.95 * (len(gaps) - 1))]:.2f},"
          f"{late:.3f},{len(stamps)},{bg_before:.2f},{bg_during:.2f}", flush=True)
    time.sleep(a.gap)

stop_bg(bg)
