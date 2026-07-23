import glob

from common import *

# TTFT/TPOT per (model, quantization) from an aggregated device_perf sweep.
# Unlike mot_device-perf.py (single model, three backends), this reads the
# combined CSV produced by concatenating several `just bench_device_perf` runs
# across models/quants.  Two outputs:
#   - per-model figures with the three primary configs (cpu/vacceloh/rknnoh,
#     labeled CPU/vAccel/RKNN), untitled so they can be captioned externally
#   - one grid figure with all five backends for every model x metric
combined = sorted(glob.glob(
    os.path.join(result_dir, "device_perf", "*combined.csv")))
if not combined:
    raise FileNotFoundError(
        "No *combined.csv found under bench/out/device_perf/; "
        "aggregate the per-run device_perf.csv files first.")
df = pd.read_csv(combined[-1])

# The aggregate concatenated per-run CSVs verbatim, so header rows recur as data.
df = df[df["type"] != "type"].copy()
for col in ["prompt_token_count", "n_gen", "ttft_s", "tpot_s"]:
    df[col] = pd.to_numeric(df[col])

# Drop the prompt-length-1 point; it is just a warmup, not a real data point.
df = df[df["prompt_token_count"] != 1]

out_dir = os.path.join(plots_dir, "device_perf")
os.makedirs(out_dir, exist_ok=True)

metrics = [("ttft_s", "TTFT [s]", "ttft"), ("tpot_s", "TPOT [s]", "tpot")]


def strip_gguf(model):
    return model[:-5] if model.endswith(".gguf") else model


def bar_plot(ax, mdata, y, ylabel, order):
    sns.barplot(mdata, x="prompt_token_count", y=y, ax=ax,
                hue="backend", hue_order=order, palette=palette[:len(order)])
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Prompt token length")
    ax.legend(title=None)


# ── Per-model figures: primary configs only, OH dropped from the labels ──────
main_labels = {"cpu": "CPU", "vacceloh": "vAccel", "rknnoh": "RKNN"}
main_order = ["CPU", "vAccel", "RKNN"]
main = df[df["type"].isin(main_labels)].copy()
main["backend"] = pd.Categorical(main["type"].map(main_labels),
                                 categories=main_order, ordered=True)

for model, mdata in main.groupby("model"):
    name = strip_gguf(model)
    mdata = mdata.sort_values(["backend", "prompt_token_count"])
    for y, ylabel, tag in metrics:
        fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))
        bar_plot(ax, mdata, y, ylabel, main_order)
        ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(out_dir, f"device-perf-{tag}-{name}.pdf"))
        plt.close(fig)

# ── Grid figure: all five backends, one row per model, TTFT | TPOT columns ───
full_labels = {
    "cpu": "CPU",
    "vaccel": "vAccel",
    "vacceloh": "vAccel-OH",
    "rknn": "RKNN",
    "rknnoh": "RKNN-OH",
}
full_order = list(full_labels.values())
full = df[df["type"].isin(full_labels)].copy()
full["backend"] = pd.Categorical(full["type"].map(full_labels),
                                 categories=full_order, ordered=True)

models = sorted(full["model"].unique())
fig, axes = plt.subplots(len(models), len(metrics),
                         figsize=(figwidth_full, len(models) * fig_height))
for row, model in zip(axes, models):
    mdata = full[full["model"] == model].sort_values(
        ["backend", "prompt_token_count"])
    for ax, (y, ylabel, _) in zip(row, metrics):
        bar_plot(ax, mdata, y, ylabel, full_order)
        ax.set_title(strip_gguf(model), fontsize=FONTSIZE)
        ax.legend(title=None, ncol=2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(out_dir, "device-perf-full-grid.pdf"))
plt.close(fig)

print(f"Wrote {2 * main['model'].nunique()} plots and the full grid to {out_dir}")