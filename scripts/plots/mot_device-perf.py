from fontTools.cffLib import parseBlendList

import glob

from common import *


def generate_input_data(type: str, ttft_ratio: float, tpot_ratio: float,
                        models: list[str], model_ttfts: list[list[float]], model_tpots: list[float]) -> pd.DataFrame:
    prompt_token_count = np.array([16, 32, 64, 128, 256])
    output_token_count = np.array([184, 220, 81, 301, 236])
    return pd.concat(pd.DataFrame({
        "prompt_token_count": prompt_token_count,
        "TTFT": np.array(model_ttft) / ttft_ratio,
        "prompt_energy": np.array(model_ttft),
        "output_token_count": output_token_count,
        "T_TG": output_token_count / (model_tpot * tpot_ratio),
        "textgen_energy": output_token_count / model_tpot,
        "type": [type] * 5,
        "Model": [model] * 5
    }) for model, model_ttft, model_tpot in zip(models, model_ttfts, model_tpots))


data = pd.concat([
    generate_input_data("CPU", 1, 1, ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[1.3, 2.6, 5.1, 10.4, 21.2],
                         [1.8, 3.6, 7.2, 14.5, 28.7],
                         [1.0, 1.9, 3.8, 7.6, 15.4]], [3.2, 2.8, 4.3]),
    generate_input_data("vAccel", 2, 1.5, ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[1.3, 2.6, 5.1, 10.4, 21.2],
                         [1.8, 3.6, 7.2, 14.5, 28.7],
                         [1.0, 1.9, 3.8, 7.6, 15.4]], [3.2, 2.8, 4.3]),
    generate_input_data("Native", 2.2, 1.7, ["Llama3.1-1B", "Qwen2.5-1.5B", "Gemma3-1B"],
                        [[1.3, 2.6, 5.1, 10.4, 21.2],
                         [1.8, 3.6, 7.2, 14.5, 28.7],
                         [1.0, 1.9, 3.8, 7.6, 15.4]], [3.2, 2.8, 4.3]),
])


data["TPOT"] = data["T_TG"] / data["output_token_count"]

# First plot: real TTFT vs prompt length from the most recent device_perf sweep
# (produced by `just bench_device_perf`, one timestamped dir per run).
# vacceloh/rknnoh are measured but not plotted here.
device_perf_runs = sorted(glob.glob(
    os.path.join(result_dir, "device_perf", "*", "device_perf.csv")))
if not device_perf_runs:
    raise FileNotFoundError(
        "No device_perf runs found under bench/out/device_perf/; "
        "run `just bench_device_perf` first.")
device_perf = pd.read_csv(device_perf_runs[-1])
type_labels = {"cpu": "CPU", "vaccel": "vAccel", "rknn": "Native"}
first = device_perf[device_perf["type"].isin(type_labels)].copy()
first["type"] = pd.Categorical(first["type"].map(type_labels),
                               categories=["CPU", "vAccel", "Native"], ordered=True)
first = first.sort_values(["type", "prompt_token_count"])

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
sns.barplot(first, x="prompt_token_count", y="ttft_s",
            ax=ax,
            hue="type", palette=palette[:3])
ax.set_ylabel("TTFT [s]")
ax.set_xlabel("Prompt token length")
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mot-device-perf.pdf"))

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
sns.barplot(data[data["prompt_token_count"]==256], x="Model", y="TTFT",
            ax=ax,
            hue="type", palette=palette[:3])
ax.set_ylabel("TTFT [s]")
ax.set_xlabel(None)
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mot-device-perf1.pdf"))