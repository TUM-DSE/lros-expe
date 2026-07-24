import pandas as pd

from common import *

def generate_input_data(type: str, prefill_efficiency: float,
                        decode_efficiency: float, ttft_ratio: float, tpot_ratio:float) -> pd.DataFrame:
    prompt_token_count = np.array([24]*5)
    output_token_count = np.array([32]*5)
    return pd.DataFrame({
        "prompt_token_count": prompt_token_count,
        "TTFT": prompt_token_count / ttft_ratio,
        # Scaled so the topmost point (the Linux baseline) sits at 1.2 J/Tk
        "prompt_energy": prompt_token_count * 1.2 / prefill_efficiency,
        "output_token_count": output_token_count,
        "T_TG": output_token_count / tpot_ratio,
        "textgen_energy": output_token_count / decode_efficiency,
        "type": [type] * 5,
    })


data = pd.concat([
    generate_input_data("Linux (no throttling)", 1, 1, 1, 1),
    generate_input_data("LROS (throttle 20%)", 1.35, 1.4, 0.83, 0.83),
    generate_input_data("LROS (throttle 50%)", 1.6, 1.65, 0.65, 0.65),
    generate_input_data("LROS (throttle 100%)", 1.8, 1.85, 0.5, 0.5),
    generate_input_data("LROS (throttle 150%)", 1.6, 1.65, 0.4, 0.4),
])
#data["TPOT"] = data["T_TG"] / data["output_token_count"]
data["prefill_efficiency"] = data["prompt_energy"] / data["prompt_token_count"]
data["decode_efficiency"] = data["textgen_energy"] / data["output_token_count"]



fig, ax = plt.subplots(figsize=(figwidth_third, 1.6))
sns.scatterplot(data, x="TTFT", y="prefill_efficiency", ax=ax)
# Headroom past the last x tick so its centered label stays inside the axes
# box; y ticks kept to one decimal
ax.set_xlim(left=0, right=70)
#ax.set_xticks([0, 20, 40, 60])
ax.set_ylim(bottom=0, top=1.3)
#ax.set_yticks(np.arange(0, 1.3, 0.4))
ax.set_ylabel("Prefill energy [J/Tk]")
ax.set_xlabel("TTFT [s]")
ax.set_title(lower_better_str, fontsize=FONTSIZE, color="navy")
# Called twice: the first pass can work with stale text metrics, the second
# fixes up the remaining clipping
fig.tight_layout(pad=0.1)
fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "mot-energy.pdf"))