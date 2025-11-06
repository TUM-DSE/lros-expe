#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common import *

def load_data() -> pd.DataFrame:
    df = pd.read_csv(result_dir+'/perf_isolation/perf_isolation.csv')
    df['interference'] = df['type_interference'] + '-' + df['nb_thread_interference'].astype(str)
    df = df.assign(time=round(df['time']/1000, 2))
    df = df[['setup', 'interference', 'time']]
    overhead = df.pivot_table(index='setup', columns='interference', values='time')
    columns = [ c for c in df['interference'].unique() if c != "none-0" ]
    overhead = overhead[columns].div(overhead['none-0'], axis=0)
    overhead = overhead.reset_index().melt(id_vars='setup', var_name='interference', value_name='overhead')
    return df, overhead


def main():
    data, overhead = load_data()
    print(data)
    print(overhead)

    plot_order = [ c for c in data['interference'].unique() ]
    fig, ax = plt.subplots(1, 2, figsize=(figwidth_full, fig_height), sharey=True)
    sns.barplot(ax=ax[0], data=data[data.setup=="process"], x = "interference", y = "time", edgecolor="black",
                color=palette[0], order=plot_order)
    sns.barplot(ax=ax[1], data=data[data.setup=="vm"], x = "interference", y = "time", edgecolor="black",
                color=palette[1], order=plot_order)
  
    for bar in ax[0].patches:
        interference = data[(data.setup == 'process') & (data.time == bar.get_height())]['interference'].values[0]
        if interference != 'none-0':
            ax[0].text(bar.xy[0]+bar.get_width()/6, bar.get_height()*1.15, 
                'x'+str(round(
                    overhead[(overhead.setup == 'process') & (overhead.interference == interference)]['overhead'].values[0], 2)), 
                color = darken(palette[3]), fontsize=FONTSIZE-1)
    for bar in ax[1].patches:
        interference = data[(data.setup == 'vm') & (data.time == bar.get_height())]['interference'].values[0]
        if interference != 'none-0':
            ax[1].text(bar.xy[0]+bar.get_width()/6, bar.get_height()*1.15, 
                'x'+str(round(
                    overhead[(overhead.setup == 'vm') & (overhead.interference == interference)]['overhead'].values[0], 2)), 
                color = darken(palette[3]), fontsize=FONTSIZE-1)
    ax[0].set_xticklabels(data[data.setup=="process"]['interference'], fontsize=FONTSIZE)
    ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=FONTSIZE)
    ax[1].set_xticklabels(data[data.setup=="process"]['interference'], fontsize=FONTSIZE)
    ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=FONTSIZE)

    ax[0].set_xlabel("Type of interference")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_xlabel("Type of interference")
    ax[1].set_ylabel("Time (s)")
    #ax.legend(loc="lower right", title=None,fontsize=FONTSIZE
    #    bbox_to_anchor=(0.5, -0.15),
    #    ncol=2,
    #)
    
    ax[0].set_title("Process", fontsize=FONTSIZE, color="black")
    ax[1].set_title("VM", fontsize=FONTSIZE, color="black")

    #plt.suptitle(lower_better_str, fontsize=FONTSIZE, color="navy", y=0.85, x=0.53)
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()
    plt.savefig(os.path.join(plots_dir, "perf_isolation.pdf"), format="pdf", pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    main()
