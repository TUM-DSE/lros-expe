import os.path
import string

import pandas as pd

from common import *


def generate_input_data(type: tuple[str, float, float], models: list[str],
                        model_ttft: list[list[float]], model_tpot: list[float]) -> pd.DataFrame:
    output_token_count = np.array([184, 220, 81, 301, 236, 255, 356, 92, 118, 152])
    return pd.concat(pd.DataFrame({
        "prompt_token_count": [41, 48, 52, 56, 101, 155, 174, 190, 239, 420],
        "TTFT": np.array(ttft) / type[1],
        "T_TG": output_token_count / (tpot * type[2]),
        "output_token_count": output_token_count,
        "type": [type[0]] * 10,
        "Model": [model] * 10
    }) for model, ttft, tpot in zip(models, model_ttft, model_tpot))


data = pd.concat([
    generate_input_data(("Linux", 1, 1), ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1],
                         [9.1, 10.9, 11.8, 12.5, 22.9, 34.7, 39.4, 43.0, 55.1, 100.4],
                         [4.8, 5.6, 6.0, 6.5, 11.8, 18.1, 21.2, 23.1, 30.0, 54.5]], [3.2, 2.8, 4.3]),
    generate_input_data(("LROS", 1.5, 1.08), ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1],
                         [9.1, 10.9, 11.8, 12.5, 22.9, 34.7, 39.4, 43.0, 55.1, 100.4],
                         [4.8, 5.6, 6.0, 6.5, 11.8, 18.1, 21.2, 23.1, 30.0, 54.5]], [3.2, 2.8, 4.3]),
])
data["TPOT"] = data["T_TG"] / data["output_token_count"]

for m, x in zip(["Llama3.1-1B", "Gemma3-1B", "Qwen2.5-1.5B"], string.ascii_lowercase):
    fig, ax = plt.subplots(figsize=(figwidth_third, fig_height))
    sns.scatterplot(data=data[data["Model"] == m], x="prompt_token_count", y="TTFT", ax=ax, hue="type", style="type",
                    markers=True,
                    # markeredgewidth=0.1,
                    linewidth=0.1,
                    # dashes=False,
                    palette=palette[:2])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("TTFT [s]")
    ax.set_xlabel("Input token count")
    ax.set_title(f"({x}) {lower_better_str}", fontsize=FONTSIZE, color="navy")

    fig.tight_layout(pad=0.1)
    # fig.tight_layout(pad=0.1, rect=(0, 0.1, 1, 1))
    # ax.text(
    #     0.5, -0.35, m,  # Adjust -0.35 as needed
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="top",
    #     clip_on=False,
    # )

    # fig.subplots_adjust(bottom=0.3)
    ax.legend(title=None)
    fig.savefig(os.path.join(plots_dir, "mock", f"end-to-end-perf-ttft-{m}.pdf"))

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
sns.scatterplot(data=data, x="prompt_token_count", y="TTFT", ax=ax, hue="type", style="type", markers=True,
             palette=palette[:2])
ax.set_ylim(bottom=0)
ax.set_ylabel("TTFT [s]")
ax.set_xlabel("Input token count")
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "end-to-end-perf-ttft.pdf"))

fig, ax = plt.subplots(figsize=(figwidth_full, fig_height))
sns.lineplot(data=data, x="output_token_count", y="TPOT", ax=ax, hue="type", style="type", markers=True,
             dashes=False, palette=palette[:2])
ax.set_ylim(bottom=0, top=0.4)
ax.set_ylabel("TPOT [s]")
ax.set_xlabel("Output token count")
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "end-to-end-perf-tpot.pdf"))

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
sns.barplot(data=data, x="Model", y="TPOT", ax=ax, hue="type", palette=palette[:2])
ax.set_ylim(bottom=0, top=0.4)
ax.set_ylabel("TPOT [s]")
ax.set_xlabel(None)
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "end-to-end-perf-tpot1.pdf"))
