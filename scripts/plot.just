# Weight paging under memory pressure (the real mem-weights-tp figure): decode
# throughput vs. memory quota from one or more `just swap_sweep` runs.
# csv: swap.csv file(s), space separated; empty picks the newest run.
# out_dir: where the figures go (PDF + PNG, full width and single-column/);
# empty means bench/plots/swap/.
plot_swap csv="" out_dir="" args="":
	#!/usr/bin/env bash
	set -euo pipefail
	csvs=()
	for c in {{csv}}; do csvs+=("$(cd "{{invocation_directory()}}" && realpath "$c")"); done
	out=(); [ -n "{{out_dir}}" ] && out=(-o "$(cd "{{invocation_directory()}}" && realpath -m "{{out_dir}}")")
	cd {{source_directory()}}/plots
	export PLOT_USETEX=1   # the paper's figures are typeset by LaTeX
	python3 swap.py "${csvs[@]}" "${out[@]}" {{args}}

# KV-cache swap vs. recompute: the cost of each against the context length,
# from one or more `just kvswap_sweep` runs. csv: kvswap.csv file(s), space
# separated; empty picks the newest run. out_dir: where the figures go (PDF +
# PNG); empty means bench/plots/kvswap/.
plot_kvswap csv="" out_dir="" args="":
	#!/usr/bin/env bash
	set -euo pipefail
	csvs=()
	for c in {{csv}}; do csvs+=("$(cd "{{invocation_directory()}}" && realpath "$c")"); done
	out=(); [ -n "{{out_dir}}" ] && out=(-o "$(cd "{{invocation_directory()}}" && realpath -m "{{out_dir}}")")
	cd {{source_directory()}}/plots
	export PLOT_USETEX=1   # the paper's figures are typeset by LaTeX
	python3 kvswap.py "${csvs[@]}" "${out[@]}" {{args}}
