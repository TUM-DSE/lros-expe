#!/usr/bin/env python3
"""Mock figures for the "End-to-end Performance" evaluation subsection.

Synthetic numbers, only meant to show what the figures will look like:
  end-to-end-perf-ttft-<model>.pdf  TTFT vs. prompt length, one per model
  end-to-end-perf-tpot1.pdf         TPOT per model
  e2e-latency-cdf.pdf               end-to-end request latency distribution
"""
from common import *

OUT_DIR = parse_out_dir(description=__doc__)

RNG = np.random.default_rng(20260903)

MODELS = ["Llama3.1-1B", "Gemma3-1B", "Qwen2.5-1.5B"]
# per-model prefill cost: seconds per prompt token and fixed setup cost
PREFILL = {"Llama3.1-1B": (0.152, 0.9), "Gemma3-1B": (0.118, 0.7),
           "Qwen2.5-1.5B": (0.229, 1.2)}
# per-model decode cost on the baseline, seconds per output token
DECODE = {"Llama3.1-1B": 0.205, "Gemma3-1B": 0.163, "Qwen2.5-1.5B": 0.284}

N_REQUESTS = 55


def request_trace():
    """A ShareGPT-like mix of prompt and response lengths."""
    prompt = np.clip(RNG.lognormal(mean=4.9, sigma=0.62, size=N_REQUESTS), 30, 780)
    output = np.clip(RNG.lognormal(mean=5.0, sigma=0.45, size=N_REQUESTS), 40, 480)
    return np.round(prompt).astype(int), np.round(output).astype(int)


def lros_gain(n_tokens):
    """LROS pulls further ahead as prompts grow: prefetching and the
    accelerator path both pay off more once the prefill dominates."""
    return 1.24 + 0.42 * np.clip(n_tokens, 0, 800) / 800.0


prompt_tokens, output_tokens = request_trace()

rows = []
for model in MODELS:
    slope, base = PREFILL[model]
    noise = RNG.normal(1.0, 0.045, size=N_REQUESTS)
    linux_ttft = (base + slope * prompt_tokens) * noise
    lros_ttft = linux_ttft / lros_gain(prompt_tokens) * RNG.normal(1.0, 0.03, size=N_REQUESTS)

    linux_tpot = DECODE[model] * RNG.normal(1.0, 0.05, size=N_REQUESTS)
    lros_tpot = linux_tpot / RNG.normal(1.11, 0.03, size=N_REQUESTS)

    for system, ttft, tpot in (("linux", linux_ttft, linux_tpot),
                               ("lros", lros_ttft, lros_tpot)):
        rows.append(pd.DataFrame({
            "model": model, "system": system,
            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "ttft": ttft, "tpot": tpot,
            "latency": ttft + tpot * output_tokens,
        }))
data = pd.concat(rows, ignore_index=True)


# --------------------------- (1) TTFT scatter ------------------------------
# One narrow panel per model so the three fit across a figure*.
for i, model in enumerate(MODELS):
    fig, ax = plt.subplots(figsize=(figwidth_third, fig_height))
    sub = data[data["model"] == model]
    for key in ("linux", "lros"):
        st = style_for(key)
        s = sub[sub["system"] == key]
        ax.scatter(s["prompt_tokens"], s["ttft"], s=7, marker=st["marker"],
                   facecolor=st["color"], edgecolor=darken(st["color"]),
                   linewidth=0.3, label=st["label"], zorder=3)
        # a trend line makes the diverging slopes readable at this size
        fit = np.polyfit(s["prompt_tokens"], s["ttft"], 1)
        xs = np.linspace(0, sub["prompt_tokens"].max() * 1.05, 50)
        ax.plot(xs, np.polyval(fit, xs), color=st["color"], linewidth=0.9,
                linestyle="--", zorder=2)

    ax.set_xlim(0, sub["prompt_tokens"].max() * 1.05)
    ax.set_ylim(0, sub["ttft"].max() * 1.30)
    ax.set_ylabel("TTFT [s]", fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_xlabel("Prompt tokens", fontsize=FONTSIZE_AXIS_LABEL)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    thin_spines(ax)
    better_hint(ax)
    ax.legend(loc="lower right", fontsize=FONTSIZE_LEGEND, frameon=False,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.25)
    panel_caption(ax, "(%s) %s" % ("abc"[i], model), y=-0.34)
    fig.tight_layout(pad=0.1)
    save(fig, f"end-to-end-perf-ttft-{model}.pdf", OUT_DIR)


# ------------------------------ (2) TPOT -----------------------------------
fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
means = {k: [data[(data.model == m) & (data.system == k)]["tpot"].mean() for m in MODELS]
         for k in ("linux", "lros")}
errs = {k: [data[(data.model == m) & (data.system == k)]["tpot"].std() for m in MODELS]
        for k in ("linux", "lros")}
grouped_bars(ax, MODELS, ["linux", "lros"], means, errors=errs, baseline="linux")
ax.set_ylabel("TPOT [s]", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylim(0, max(means["linux"]) * 1.30)
better_hint(ax)
ax.legend(*legend_handles(["linux", "lros"]), loc="lower center",
          bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=FONTSIZE_LEGEND,
          frameon=False, handlelength=1.4, handleheight=0.8, columnspacing=1.0,
          handletextpad=0.4)
fig.tight_layout(pad=0.1)
save(fig, "end-to-end-perf-tpot1.pdf", OUT_DIR)


# ------------------------- (3) end-to-end latency --------------------------
fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
for key in ("linux", "lros"):
    st = style_for(key)
    xs, ys = cdf(data[data.system == key]["latency"])
    ax.plot(xs, ys, color=st["color"], linewidth=1.2, label=st["label"], zorder=3)
    # p99 marker: the tail is what the memory limit hurts most
    p99 = np.percentile(data[data.system == key]["latency"], 99)
    ax.plot([p99], [0.99], marker=st["marker"], markersize=3.5,
            color=st["color"], markeredgecolor="black", markeredgewidth=0.3, zorder=4)
    ax.annotate("p99", xy=(p99, 0.99), xytext=(3, -7), textcoords="offset points",
                fontsize=FONTSIZE_ANNOTATION, color=st["color"])

ax.set_xlabel("End-to-end request latency [s]", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylabel("CDF", fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylim(0, 1.02)
ax.set_xlim(0, data["latency"].max() * 1.05)
ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.35, zorder=0)
ax.set_axisbelow(True)
thin_spines(ax)
ax.annotate(left_better_str, color="blue", xy=(0.02, 0.90),
            xycoords="axes fraction", fontsize=FONTSIZE_ANNOTATION)
ax.legend(loc="center left", fontsize=FONTSIZE_LEGEND, frameon=False,
          handletextpad=0.4, labelspacing=0.3)
fig.tight_layout(pad=0.1)
save(fig, "e2e-latency-cdf.pdf", OUT_DIR)
