import pandas as pd

from common import *


def generate_input_data(type: tuple[str, float, float], rates: list[float], ttft_factors: list[float]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({
        "prompt_token_count": [41, 48, 52, 56, 101, 155, 174, 190, 239, 420],
        "TTFT": factor * np.array([6.4, 7.4, 8.1, 8.7, 15.8, 24.5, 28.0, 31.0, 39.1, 70.1]) / type[1],
        "T_TG": [128 / type[2]] * 10,
        "output_token_count": [128] * 10,
        "type": [type[0]] * 10,
        "request_rate": rate
    }) for rate, factor in zip(rates, ttft_factors)])


data = pd.concat([
    generate_input_data(("Native", 1, 3.2), [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], [1, 1, 1.1, 1.3, 2, 2.5]),
    generate_input_data(("LROS", 1.5, 3.4), [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], [1, 1, 1.05, 1.1, 1.3, 1.5]),
])

data["TTFT_norm"] = data["TTFT"] / data["prompt_token_count"]

fig, ax = plt.subplots(figsize=(figwidth_half, fig_height))

sns.barplot(data=data, x="request_rate", y="TTFT_norm", ax=ax, hue="type",
            # style="type", markers=True,
            # dashes=False,
            palette=palette[:2])
ax.set_ylim(bottom=0)
ax.set_ylabel("Normalized TTFT")
ax.set_xlabel("Request rate [1/min]")
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
ax.legend(title=None)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "sched-ttft.pdf"))
