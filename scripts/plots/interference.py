import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import *


# Read CSV
df = pd.read_csv(os.path.join(result_dir,"interference","data.csv"))

# Extract values by experiment name
stressor = df.loc[df["Experiment"] == "stressor", "Seconds"].iloc[0]
inference = df.loc[df["Experiment"] == "inference", "Seconds"].iloc[0]
parallel = df.loc[df["Experiment"] == "parallel", "Seconds"].iloc[0]
cgroup = df.loc[df["Experiment"] == "cgroup", "Seconds"].iloc[0]

fig, ax = plt.subplots(figsize=(figwidth_full_thesis, fig_height))

# Base bars
plot_df = pd.DataFrame({
    "Bar": ["Sequential"],
    "Seconds": [stressor]
})

x1=sns.barplot(
    data=plot_df,
    x="Bar",
    y="Seconds",
    color=palette[0],
    ax=ax
    , edgecolor="black", linewidth=0.5, label="Stressor"
)

bars = ax.patches

# Add second part of stacked bar
ax.bar(
    bars[0].get_x() + bars[0].get_width()/2,
    inference,
    bottom=stressor,
    width=bars[0].get_width(),
    color=palette[3],
    align="center"
    , edgecolor="black", linewidth=0.5, label="Inference"
)

sns.barplot(
    data=pd.DataFrame({
        "Bar": ["Parallel", "CGroup"],
        "Seconds": [parallel, cgroup]
    }),
    x="Bar",
    y="Seconds",
    color=palette[4],
    ax=ax
    , edgecolor="black", linewidth=0.5, label="Both"
)



ax.set_ylim((0,50))
ax.set_ylabel("Runtime [s]")
ax.set_xlabel("")
ax.legend(ncol= 3, loc="upper center")

fig.tight_layout()
fig.savefig(os.path.join(plots_dir, "interference.pdf"))
fig.savefig(os.path.join(plots_dir, "interference.png"), dpi=400)