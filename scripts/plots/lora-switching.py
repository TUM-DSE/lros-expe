import pandas as pd

from common import *


def generate_input_data(type: tuple[str, float, float], lora_count: list[int],
                        ttft_factors: list[float]) -> pd.DataFrame:
    return pd.concat(pd.DataFrame({
        "prompt_token_count": [41, 48, 52, 56, 101, 155, 174, 190, 239, 420],
        "TTFT": factor * np.array([6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1]) / type[1],
        "T_TG": [128 / type[2]] * 10,
        "output_token_count": [128] * 10,
        "type": [type[0]] * 10,
        "lora_count": count
    }) for count, factor in zip(lora_count, ttft_factors))


data = pd.concat([
    generate_input_data(("Native", 1, 3.2), [1, 2, 4, 8], [1, 1.3, 1.35, 1.4]),
    #generate_input_data(("LROS", 1.5, 3.4), [1, 2, 4, 8], [1, 1, 1.05, 1.1, 1.3, 1.5]),
])

data["TTFT_norm"] = data["TTFT"] / data["prompt_token_count"]

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))

sns.barplot(data=data, x="lora_count", y="TTFT_norm", ax=ax,
            #hue="lora_count", legend=False,
            color=palette[0])
ax.set_ylim(bottom=0)
ax.set_ylabel("Normalized TTFT")
ax.set_xlabel("Number of different LoRA used")
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
#ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "lora-ttft.pdf"))
