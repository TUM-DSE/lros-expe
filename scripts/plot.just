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
	python3 kvswap.py "${csvs[@]}" "${out[@]}" {{args}}
