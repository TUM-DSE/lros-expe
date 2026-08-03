import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common import *


df = pd.DataFrame({
    "threads": [1, 2, 3, 4, 1, 2, 3, 4],
    "runtime": [1.8, 1.9, 2.0, 2.1,
                2.9, 6.3, 10.2, 13.9],
    "series": ["Limited"] * 4 + ["Unlimited"] * 4
})


fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))

sns.barplot(
    data=df,
    x="threads",
    y="runtime",
    hue="series",
    palette = palette,
    ax=ax
    , edgecolor="black", linewidth=0.5,
)

ax.set_xlabel("Number of threads")
ax.set_ylabel("Throughput [T/s]")
ax.legend(title=None)

plt.tight_layout()
fig.savefig(os.path.join(plots_dir, "res-mgmt-overhead.pdf"))
fig.savefig(os.path.join(plots_dir, "res-mgmt-overhead.png"), dpi=400)