#!/usr/bin/env python3
"""The orchestration trace: three applications on two models sharing one
machine, and the events a policy has to react to.

    python3 orch_traces.py <out.trace> --sharegpt <ShareGPT json> [options]

Applications (the @app lines; the model indices follow --models of lros-serve):
  chat        the foreground: one conversation of several turns on the chat
              model, closed loop -- a turn arrives think time after the previous
              one completed. QoS: reading-speed TPOT and a TTFT bound.
  summarise   background: a periodic stream of long-prompt summarisations on the
              background model. QoS: a completion deadline per request.
  reply       background, bursty: smart replies to short messages on the
              background model with its adapter, in bursts. QoS: a shorter
              deadline per request.

Phases, in order:
  1. background only: summarise (and the first reply burst, if placed there)
     have the machine to themselves
  2. the chat arrives and takes its turns
  3. the chat goes idle for a long think time while the background fills
     memory; a reply burst lands in the pause
  4. the chat returns; a reply burst lands during one of its turns, beside a
     summarisation

The chat's turns are the human turns of one ShareGPT conversation; the reply
prompts are short ShareGPT opening messages; the summarisation prompts are
synthetic, like the scheduling traces (traces.py).
"""
import argparse
import json
import random
import re

FILL = ("The history of computing is long and its lessons are repetitive. "
        "Each generation rediscovers that memory is slow, that scheduling "
        "is a policy question, and that the cost of moving data usually "
        "exceeds the cost of computing on it.")   # about 47 tokens
SUMM_INSTR = "Summarise the following passage in a few sentences."
REPLY_INSTR = "Suggest a short reply to this message:"


def clean(text):
    # ShareGPT's markdown escapes and the "2 / 2" regeneration marks are not
    # what a user typed.
    text = re.sub(r"\\([\\*_#`\[\]()~>-])", r"\1", text)
    text = re.sub(r"\s*\d+ / \d+$", "", text)
    text = " ".join(text.split())
    if not text.isascii():
        return None
    if any(c in text for c in '{}"\''):
        return None
    # Not what a figure should quote.
    if re.search(r"\b(sex|porn|nude|naked|kill|erotic)", text, re.I):
        return None
    return text


def load_sharegpt(path, seed, n_turns, chat_len, reply_len, n_reply):
    """One conversation with n_turns usable human turns, and n_reply short
    opening messages from other conversations."""
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    order = list(range(len(data)))
    rng.shuffle(order)

    chat, replies = None, []
    for i in order:
        conv = data[i].get("conversations") or []
        humans = [clean(t.get("value", "")) for t in conv if t.get("from") == "human"]
        humans = [h for h in humans if h]
        if chat is None:
            # The first n_turns human turns, so that they read as one
            # conversation; every one of them within the length bounds.
            turns = humans[:n_turns]
            if len(turns) == n_turns and all(chat_len[0] <= len(h) <= chat_len[1] for h in turns):
                chat = turns
                continue
        if len(replies) < n_reply and humans and reply_len[0] <= len(humans[0]) <= reply_len[1]:
            replies.append(humans[0])
        if chat is not None and len(replies) >= n_reply:
            break
    if chat is None:
        raise SystemExit(f"no ShareGPT conversation with {n_turns} human turns of {chat_len} chars")
    if len(replies) < n_reply:
        raise SystemExit(f"only {len(replies)} reply prompts of {reply_len} chars")
    return chat, replies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--sharegpt", required=True, help="ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--seed", type=int, default=1)
    # --models <A>,<A>+<lora>,<B> of lros-serve: 0 the background model, 1 the
    # same with the reply adapter, 2 the chat model.
    ap.add_argument("--chat-model", type=int, default=2)
    ap.add_argument("--summ-model", type=int, default=0)
    ap.add_argument("--reply-model", type=int, default=1)
    # chat
    ap.add_argument("--chat-start", type=int, default=30000, help="ms: when the first turn arrives")
    ap.add_argument("--chat-turns", type=int, default=8)
    ap.add_argument("--think", type=int, default=10000, help="ms between a turn's completion and the next")
    ap.add_argument("--idle-after", type=int, default=4, help="turn after which the user goes idle")
    ap.add_argument("--idle", type=int, default=40000, help="ms: the long think time")
    # The target stays the reading speed (4.8 tok/s, 208 ms) whatever the
    # model: if a model meets it too easily, the experiment moves to a larger
    # model rather than to a stricter target.
    ap.add_argument("--chat-gen", type=int, default=96)
    ap.add_argument("--chat-tpot", type=int, default=208, help="ms: 4.8 tok/s reading speed")
    # First token: a base plus a term per prompt token, so a longer turn is
    # allowed longer. 60 ms/token is two thirds of the 3B model's raw
    # four-core prefill rate (24 t/s), the second covers the first decode
    # step and the scheduling.
    ap.add_argument("--chat-ttft", type=int, default=1000, help="ms, base")
    ap.add_argument("--chat-ttft-tok", type=int, default=60, help="ms per prompt token")
    ap.add_argument("--chat-len", type=int, nargs=2, default=(20, 300), help="chars per turn")
    # summarise
    # Sized (2026-09-24) for a 3B Q8 chat model beside the 1B Q8 background
    # model: the background is meant to finish on time when the weights fit
    # under every engine, and to show a trade-off under the budget, where it
    # streams at 0.35-0.5 s per token in the chat's pauses.
    # Long prompts through the whole chat period, with prefill that a chat
    # turn has to displace; spaced so that one request is done before the
    # next arrives under the budget (2026-09-25: 553 tokens every 45 s took
    # 55-70 s each on one core beside the chat and queued up, every engine
    # missing the deadline).
    ap.add_argument("--summ-period", type=int, default=55000, help="ms")
    ap.add_argument("--summ-n", type=int, default=6)
    ap.add_argument("--summ-fill", type=int, default=8, help="paragraphs per prompt (~47 tok each)")
    ap.add_argument("--summ-gen", type=int, default=16)
    # Deadlines (2026-09-25) cover what a request costs under the budget when
    # a chat turn and a reply burst land on it, measured with the adaptive
    # LROS arm: a summarisation 73-87 s, a burst 38-46 s; alone they take
    # 12-34 s and 7-8 s.
    ap.add_argument("--summ-deadline", type=int, default=90000)
    ap.add_argument("--summ-prio", type=int, default=2)
    # reply bursts
    # One burst during an early turn, one in the chat's pause, one during a
    # late turn beside a streaming summarisation.
    ap.add_argument("--burst-at", type=int, nargs="*", default=[60000, 150000, 250000], help="ms")
    ap.add_argument("--burst-n", type=int, default=8)
    ap.add_argument("--burst-spread", type=int, default=3000, help="ms over which a burst arrives")
    ap.add_argument("--reply-gen", type=int, default=8)
    ap.add_argument("--reply-deadline", type=int, default=60000)
    ap.add_argument("--reply-prio", type=int, default=1)
    ap.add_argument("--reply-len", type=int, nargs=2, default=(40, 200), help="chars per message")
    a = ap.parse_args()

    n_reply = a.burst_n * len(a.burst_at)
    chat, replies = load_sharegpt(a.sharegpt, a.seed, a.chat_turns, tuple(a.chat_len),
                                  tuple(a.reply_len), n_reply)

    lines = []
    lines.append(f"@app chat model={a.chat_model} prio=0 role=foreground "
                 f"tpot_ms={a.chat_tpot} ttft_ms={a.chat_ttft} ttft_tok_ms={a.chat_ttft_tok}")
    lines.append(f"@app summarise model={a.summ_model} prio={a.summ_prio} role=background "
                 f"deadline_ms={a.summ_deadline}")
    lines.append(f"@app reply model={a.reply_model} prio={a.reply_prio} role=background "
                 f"deadline_ms={a.reply_deadline}")

    reqs = []   # (t_ms or None for relative, text)
    summ_prompt = " ".join([SUMM_INSTR] + [FILL] * a.summ_fill)
    for i in range(a.summ_n):
        reqs.append((i * a.summ_period, f"summarise {a.summ_gen} {summ_prompt}"))
    k = 0
    for t0 in a.burst_at:
        for j in range(a.burst_n):
            t = t0 + (j * a.burst_spread) // max(1, a.burst_n - 1)
            reqs.append((t, f"reply {a.reply_gen} {REPLY_INSTR} {replies[k]}"))
            k += 1
    reqs.sort(key=lambda r: r[0])

    turns = [(a.chat_start, chat[0])]
    for i in range(1, a.chat_turns):
        think = a.idle if i == a.idle_after else a.think
        turns.append((think, chat[i]))

    with open(a.out, "w") as f:
        f.write("# Orchestration trace; see scripts/orch_traces.py for the scenario.\n")
        f.write(f"# seed {a.seed}; chat every {a.think} ms think, idle {a.idle} ms after turn "
                f"{a.idle_after}; summarise every {a.summ_period} ms x{a.summ_n}; "
                f"reply bursts of {a.burst_n} at {a.burst_at} ms\n")
        f.write("\n".join(lines) + "\n")
        f.write("# <t_ms> <app>[:<session>] <max_tokens> <prompt>\n")
        for t, text in reqs:
            f.write(f"{t} {text}\n")
        f.write("# The chat: the first turn by the clock, the rest think time after the previous one.\n")
        for i, (t, text) in enumerate(turns):
            prefix = f"{t}" if i == 0 else f"+{t}"
            f.write(f"{prefix} chat:1 {a.chat_gen} {text}\n")
    print(f"wrote {a.out}: {len(reqs)} background requests, {len(turns)} chat turns")


if __name__ == "__main__":
    main()
