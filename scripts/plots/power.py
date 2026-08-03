import os.path

import pandas as pd

from common import *


def generate_input_data(type: str, ttft_ratio: float, tpot_ratio: float, prefill_efficiency: float,
                        decode_efficiency: float, models: list[str],
                        model_ttfts: list[list[float]], model_tpots: list[float]) -> pd.DataFrame:
    prompt_token_count = np.array([41, 48, 52, 56, 101, 155, 174, 190, 239, 420])
    output_token_count = np.array([184, 220, 81, 301, 236, 255, 356, 92, 118, 152])
    return pd.concat(pd.DataFrame({
        "prompt_token_count": prompt_token_count,
        "TTFT": np.array(model_ttft) / ttft_ratio,
        "prompt_energy": np.array(model_ttft) / prefill_efficiency,
        "output_token_count": output_token_count,
        "T_TG": output_token_count / (model_tpot * tpot_ratio),
        "textgen_energy": output_token_count / (model_tpot * decode_efficiency),
        "type": [type] * 10,
        "Model": [model] * 10
    }) for model, model_ttft, model_tpot in zip(models, model_ttfts, model_tpots))


data = pd.concat([
    generate_input_data("Linux (no throttling)", 1, 1, 1, 1, ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1],
                         [9.1, 10.9, 11.8, 12.5, 22.9, 34.7, 39.4, 43.0, 55.1, 100.4],
                         [4.8, 5.6, 6.0, 6.5, 11.8, 18.1, 21.2, 23.1, 30.0, 54.5]], [3.2, 2.8, 4.3]),
    generate_input_data("LROS (throttle X1)", 1.5, 0.8, 1.2, 1.4, ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1],
                         [9.1, 10.9, 11.8, 12.5, 22.9, 34.7, 39.4, 43.0, 55.1, 100.4],
                         [4.8, 5.6, 6.0, 6.5, 11.8, 18.1, 21.2, 23.1, 30.0, 54.5]], [3.2, 2.8, 4.3]),
    generate_input_data("LROS (throttle X2)", 1.5, 0.5, 1.25, 1.5, ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1],
                         [9.1, 10.9, 11.8, 12.5, 22.9, 34.7, 39.4, 43.0, 55.1, 100.4],
                         [4.8, 5.6, 6.0, 6.5, 11.8, 18.1, 21.2, 23.1, 30.0, 54.5]], [3.2, 2.8, 4.3]),
])
data["TPOT"] = data["T_TG"] / data["output_token_count"]
data["prefill_efficiency"] = data["prompt_energy"] / data["prompt_token_count"]
data["decode_efficiency"] = data["textgen_energy"] / data["output_token_count"]

# for m in ["Llama3.1-1B", "Gemma3-1B", "Qwen2.5-1.5B"]:
#     fig, ax = plt.subplots(figsize=(figwidth_third, fig_height))
#     sns.lineplot(data=data[data["Model"] == m], x="prompt_token_count", y="TTFT", ax=ax, hue="type", style="type",
#                  markers=True,
#                  markeredgewidth=0.1,
#                  dashes=False, palette=palette[:2])
#     ax.set_ylim(bottom=0)
#     ax.set_ylabel("TTFT [s]")
#     ax.set_xlabel("Input token count")
#     ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
#
#     fig.tight_layout(pad=0.1, rect=(0, 0.1, 1, 1))
#     ax.text(
#         0.5, -0.35, m,  # Adjust -0.35 as needed
#         transform=ax.transAxes,
#         ha="center",
#         va="top",
#         clip_on=False,
#     )
#
#     # fig.subplots_adjust(bottom=0.3)
#     ax.legend(title=None)
#     fig.savefig(os.path.join(plots_dir, "mock", f"end-to-end-perf-ttft-{m}.pdf"))

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
sns.barplot(data=data, x="Model", y="prefill_efficiency", ax=ax, hue="type", palette=palette[:3])
# ax.set_ylim(bottom=0, top=0.4)
ax.set_ylabel("Prefill efficiency [J/Token]")
ax.set_xlabel(None)
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, "mock", "energy-prefill.pdf"))

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
sns.barplot(data=data, x="Model", y="decode_efficiency", ax=ax, hue="type", palette=palette[:3])
# ax.set_ylim(bottom=0, top=0.4)
ax.set_ylabel("Decode efficiency [J/Token]")
ax.set_xlabel(None)
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout()
fig.savefig(os.path.join(plots_dir, "mock", "energy-decode.pdf"))
