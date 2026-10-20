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

# Every figure of the paper from results/, typeset by LaTeX, as flat PDFs in
# out (default ../lros_paper/plots, where the motivation, sched and accel plot
# recipes write): those recipes' figures (accel-perf only, not accel-loc) and
# the evaluation's weight paging pair, KV swap, orchestration and unikernel
# figures. The unikernel figure reads bench/out/boottime/, which stays local.
[doc('every figure of the paper, into ../lros_paper/plots')]
plot_paper out=(source_directory() / "../../lros_paper/plots"):
	#!/usr/bin/env bash
	set -euo pipefail
	out=$(cd "{{invocation_directory()}}" && realpath -m "{{out}}")
	R={{source_directory()}}/../results
	tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
	mkdir -p "$out"
	j="just -f {{justfile()}}"
	$j motivation::plot "$out"
	$j sched::plot "$out"
	# The recipes below draw many figures into nested directories; the
	# paper's ones are rendered into $tmp and copied out under flat names.
	$j plot_swap $R/swap/orangepi/swap.csv "$tmp/swap" "--no-png --height 1.5 \
	  --configs linux-mmap,lros-tensor-layer --quotas olmoe-1b-7b-0924-instruct-q4_k_m=110,50,20,10 \
	  --pair Llama-3.2-1B-Instruct-f16,olmoe-1b-7b-0924-instruct-q4_k_m --layouts none"
	cp "$tmp/swap/mem-weights-tp/pairs/single-column/Llama-3.2-1B-Instruct-f16__olmoe-1b-7b-0924-instruct-q4_k_m.pdf" \
	  "$out/mem-weights-tp-dense-moe-nvme.pdf"
	$j plot_kvswap $R/kvswap/orangepi/kvswap.csv "$tmp/kvswap" "--no-png --height 1.5 \
	  --figures mem-kv-partial-total --models Llama-3.2-1B-Instruct-f16 --layouts disk"
	cp "$tmp/kvswap/mem-kv-partial-total/Llama-3.2-1B-Instruct-f16/single-column/mem-kv-partial-total_nvme.pdf" \
	  "$out/mem-kv-partial-total_nvme.pdf"
	$j plot_orch $R/orch/orangepi/orch.csv "$tmp/orch" "--arms host-fifo,lros-adaptive"
	cp "$tmp/orch/orch-tpot-bg-budget.pdf" "$out/orch-tpot-bg-budget.pdf"
	# No recipe draws accel-perf alone (accel::plot adds accel-loc, which
	# the paper does not use) or the unikernel figure.
	cd {{source_directory()}}/plots
	export PLOT_USETEX=1   # the paper's figures are typeset by LaTeX
	python3 accel-perf.py "$out/accel-perf.pdf" $R/accel-perf/*/accel-perf.csv
	# The evaluation's two full-width figures, from the panels of the ones above.
	python3 paper-eval.py "$out" $R
	python3 unikernel.py -o "$tmp/unikernel"
	cp "$tmp/unikernel/unikernel-micro.pdf" "$out/unikernel-micro.pdf"
	echo "Wrote the paper's figures to $out"
