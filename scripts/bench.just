import "run.just"

LROS_IMAGE_NAME := "llama.cpp_qemu-arm64"
LROS_PATH := proot + "/lros/"

LLAMA_CPP_TARGETS := "llama-batched-bench"

OUT_DIR := proot + "/bench/out"
parallel_img := LROS_PATH + "/.unikraft/build-parallel/parallel_qemu-arm64"

MODEL := proot + "/models/Llama-3.2-1B-Instruct-f16.gguf"
LLAMACPP_BIN := proot + "/lros/llama.cpp-rknn/build/bin/llama-batched-bench"

EVAL_ARGS := "-npp 16,32 -ntg 1,128,256 -t 1 -ngl 0 --no-mmap --no-perf"
EVAL_ARGS_SINGLE := "-npp 16 -ntg 128 -npl 1 -t 1 -ngl 0 --no-mmap --no-perf"

cpu_type := "big"
AFFINITY := if cpu_type == "big" { "4-4" } else { "0-0" }

_plugins := if path_exists("/sys/devices/platform/fdab0000.npu") == "true" { env("VACCEL_PLUGINS_RKNN", "") } else if path_exists("/proc/driver/nvidia/") == "true" { env("VACCEL_PLUGINS_CUDA", "") } else {""}

export VACCEL_PLUGINS := _plugins

bench spec variant="all" model=MODEL size_mem="10G" log_power="false" batch_size="1": #(build spec variant)
	#!/usr/bin/env bash

	set -ex

	mkdir -p {{OUT_DIR}}

	unset VACCEL_BACKENDS

	# Detect if we are on NPU or GPU system
	if [ -d /sys/devices/platform/fdab0000.npu ]; then
		powerlogger="measure_orangepi"
	elif [ -d /proc/driver/nvidia/ ]; then
		export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
		powerlogger="measure_jetson 0.1"
	fi

	if [ ! -f {{proot}}/models/Llama-3.2-1B-Instruct-f16.gguf ]; then
		echo "Please download the Llama-3.2-1B-Instruct-f16.gguf model." >&2
		exit 1
	fi

	log_power={{log_power}}
	batch_size={{batch_size}}
	d="$batch_size-$(date -Iseconds)"

	case {{lowercase(spec)}} in
		lros)
			if [ ! -f {{proot}}/config/mat_kernel_size_$batch_size.json ]; then
				{{proot}}/scripts/mat_kernel_size_generator.py {{proot}}/config/mat_kernel_size_$batch_size.json -b $batch_size
			fi
			cp -fl {{proot}}/config/mat_kernel_size_$batch_size.json {{proot}}/config/mat_kernel_size.json
			case {{lowercase(variant)}} in
				vaccel | vaccel-novec | vacceloh | vacceloh-novec | cpu | cpu-novec )
					variants={{lowercase(variant)}} ;;
				all )
					variants="vaccel vaccel-novec vacceloh vacceloh-novec cpu cpu-novec" ;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					variants="" ;;
			esac

			for v in $variants; do
				if [ $log_power = true ]; then just -f {{source_directory()}}/power.just $powerlogger > {{OUT_DIR}}/bench_lros_${v}_$d.power.txt & fi
				just -f {{justfile()}} run_lros \
					{{LROS_PATH}}/.unikraft/build-${v}/bench-${v}_qemu-arm64 \
					{{OUT_DIR}}/bench_lros_${v}_$d.out.txt \
					{{OUT_DIR}}/bench_lros_${v}_$d.err.txt \
					{{ size_mem }} {{ batch_size }}
				if [ $log_power = true ]; then pkill -P $!; fi
			done
		;;

		linux-vm | vm)
			case {{lowercase(variant)}} in
				cpu | cpu-novec )
					variants={{lowercase(variant)}} ;;
				all )
					variants="cpu cpu-novec" ;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					variants="" ;;
			esac

			for v in $variants; do
				just -f {{justfile()}} EVAL_ARGS="{{EVAL_ARGS}} -npl {{batch_size}}" run_vm \
					{{OUT_DIR}}/bench_vm_${v}_$d.out.txt \
					{{OUT_DIR}}/bench_vm_${v}_$d.err.txt \
					build-${v} {{size_mem}}
			done
		;;

		linux | process)
			requested={{lowercase(variant)}}
			case $requested in
				cpu | cpu-novec | cuda | rknn | rknnoh )
					variants=$requested ;;
				all )
					variants="cpu cpu-novec cuda rknn rknnoh" ;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					variants="" ;;
			esac

			for v in $variants; do
				# For `all`, silently skip variants whose hardware is absent;
				# for an explicit request, report the missing device.
				case $v in
					cuda )
						if [ ! -d /proc/driver/nvidia/ ]; then
							[ "$requested" = all ] && continue
							echo "Cannot execute CUDA on this device. Could not find GPU driver." >&2
							exit 1
						fi
					;;
					rknn | rknnoh )
						if [ ! -d /sys/devices/platform/fdab0000.npu ]; then
							[ "$requested" = all ] && continue
							echo "Cannot execute RKNN on this device. Could not find NPU." >&2
							exit 1
						fi
					;;
				esac

				taskset -c {{AFFINITY}} \
				{{proot}}/lros/llama.cpp-rknn/build-${v}/bin/llama-batched-bench {{EVAL_ARGS}} -npl {{batch_size}} \
					-m {{model}} \
					2> >(tee {{OUT_DIR}}/bench_process_${v}_$d.err.txt >&2) \
					| tee {{OUT_DIR}}/bench_process_${v}_$d.out.txt
			done
		;;
	esac

hugepage: (build "linux" "cpu") (build "linux" "cpu-thp")
	#!/usr/bin/env bash

	set -ex

	mkdir -p {{OUT_DIR}}/thp

	d=$(date -Iseconds)

	perf stat -e l1d_tlb,l1d_tlb_refill,l2d_tlb,l2d_tlb_refill,dtlb_walk,l1i_tlb,l1i_tlb_refill,itlb_walk -j \
		-o {{OUT_DIR}}/thp/perf_base_$d.json \
		{{LLAMACPP_BIN}} {{EVAL_ARGS}} -m {{MODEL}} \
		2> >(tee {{OUT_DIR}}/thp/bench_base_$d.err.txt >&2) \
		| tee {{OUT_DIR}}/thp/bench_base_$d.out.txt

	perf stat -e l1d_tlb,l1d_tlb_refill,l2d_tlb,l2d_tlb_refill,dtlb_walk,l1i_tlb,l1i_tlb_refill,itlb_walk -j \
		-o {{OUT_DIR}}/thp/perf_thp_$d.json \
		{{proot}}/lros/llama.cpp-rknn/build-thp/bin/llama-batched-bench {{EVAL_ARGS}} -m {{MODEL}} \
		2> >(tee {{OUT_DIR}}/thp/bench_thp_$d.err.txt >&2) \
		| tee {{OUT_DIR}}/thp/bench_thp_$d.out.txt

paging: #(build "linux" "cpu") (build "lros" "cpu")
	#!/usr/bin/env bash

	set -euo pipefail

	mkdir -p {{OUT_DIR}}/paging

	# Shared arguments (16 prompt token, 8 text gen token, 1 batch, 1 thread, no detailed integrated timings)
	args="-npp 16 -ntg 8 -npl 1 -t 1 --no-perf"

	for m in "16G" "2G"; do
		# Linux process (host) — cgroup MemoryMax=$m matches the VM memory size
		systemd-run --user --scope -p MemoryMax="$m" -p MemorySwapMax=0 --quiet -- \
			taskset -c {{AFFINITY}} \
			{{LLAMACPP_BIN}} $args --no-mmap \
			-m {{MODEL}} \
			> {{OUT_DIR}}/paging/bench_paging_"$m"_read.out.txt \
			2> {{OUT_DIR}}/paging/bench_paging_"$m"_read.err.txt
		systemd-run --user --scope -p MemoryMax="$m" -p MemorySwapMax=0 --quiet -- \
			taskset -c {{AFFINITY}} \
			{{LLAMACPP_BIN}} $args \
			-m {{MODEL}} \
			> {{OUT_DIR}}/paging/bench_paging_"$m"_mmap.out.txt \
			2> {{OUT_DIR}}/paging/bench_paging_"$m"_mmap.err.txt

		# LROS
		just -f {{justfile()}} EVAL_ARGS="$args --no-mmap" run_lros \
			{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosread.out.txt \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosread.errout.txt \
			$m
		# Fix merged output
		grep "llama_perf_context_print" {{OUT_DIR}}/paging/bench_paging_"$m"_lrosread.out.txt \
			> {{OUT_DIR}}/paging/bench_paging_"$m"_lrosread.err.txt || test $? = 1;

		if [[ "$m" == "16G" ]]; then
			buf_size=4096
		else
			buf_size=64
		fi

		just -f {{justfile()}} EVAL_ARGS="libukvmem.prefetch_size=$buf_size -- $args" run_lros \
			{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch.out.txt \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch.errout.txt \
			$m
		# Fix merged output
		grep "llama_perf_context_print" {{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch.out.txt \
			> {{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch.err.txt || test $? = 1;
	done

paging_motiv:
	#!/usr/bin/env bash

	set -euo pipefail
	
	to_csv() {
		awk -F'|' -v mem="$1" -v backend="$2" '
			/^\|[[:space:]]*[0-9]/ {
				gsub(/[[:space:]]/, "", $6); gsub(/[[:space:]]/, "", $7)
				gsub(/[[:space:]]/, "", $8); gsub(/[[:space:]]/, "", $9)
				gsub(/[[:space:]]/, "", $10); gsub(/[[:space:]]/, "", $11)
				print mem "," backend "," $6 "," $7 "," $8 "," $9 "," $10 "," $11
			}'
	}

	run_bench() {
		local mem="$1" backend="$2"; shift 2
		local out ec=0
		out=$(systemd-run --user --scope -p MemoryMax="$mem" -p MemorySwapMax=0 --quiet -- \
			taskset -c 4 "$@" 2>/dev/null) || ec=$?
		if [ "$ec" -eq 0 ]; then
			echo "$out" | to_csv "$mem" "$backend"
		else
			echo "$mem,$backend,0,0,0,0,0,0"
		fi
	}

	echo "mem_limit,backend,t_pp_s,s_pp_t_s,t_tg_s,s_tg_t_s,t_s,s_t_s"

	for m in "4096M" "3072M" "2816M" "2560M" "2048M"; do
		sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
		run_bench "$m" "mmap" \
			{{LLAMACPP_BIN}} \
			-npp 16 -ntg 32 -npl 1 -t 1 --no-perf \
			-m {{MODEL}}
		sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"
		run_bench "$m" "read" \
			{{LLAMACPP_BIN}} \
			-npp 16 -ntg 32 -npl 1 -t 1 --no-mmap --no-perf \
			-m {{MODEL}}
	done
	just -f {{justfile()}} run_lros_stdout {{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 2048 "llama-batched-bench libukvmem.prefetch_size=64 -- -npp 16 -ntg 32 -npl 1 -t 1 --no-perf -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" | to_csv "2048" "lros"

perf_isolation: (build "linux" "cpu")
	#!/usr/bin/env bash

	set -euo pipefail
	mkdir -p {{OUT_DIR}}/perf_isolation
	csv="{{OUT_DIR}}/perf_isolation/perf_isolation.csv"
	echo "setup,type_interference,nb_thread_interference,time" > "$csv"
	type_interference="cpu vm io"

	# Runners: emit a "total time" line the measurement harness parses.
	run_vm() {
		just ssh_local "/root/scripts/run_in_vm.sh /root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf"
	}
	run_process() {
		taskset -c 4 {{LLAMACPP_BIN}} {{EVAL_ARGS_SINGLE}} -m {{MODEL}}
	}

	# measure <setup-label> <runner-fn>: baseline (no interference) then, for each
	# interference type, 7-thread (leaving core 4 free) and 8-thread (all cores) runs.
	measure() {
		local setup="$1" runner="$2" ti

		echo -n "$setup,none,0," >> "$csv"
		sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
		"$runner" 2>&1 | grep "total time" | awk '{print $5}' >> "$csv"

		for ti in $type_interference; do
			echo -n "$setup,$ti,7," >> "$csv"
			sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
			taskset -c 0-3,5-7 stress --$ti 7 > /dev/null 2> /dev/null &
			"$runner" 2>&1 | grep "total time" | awk '{print $5}' >> "$csv"
			sudo kill -9 $(pidof stress)

			echo -n "$setup,$ti,8," >> "$csv"
			sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
			taskset -c 0-7 stress --$ti 8 > /dev/null 2> /dev/null &
			"$runner" 2>&1 | grep "total time" | awk '{print $5}' >> "$csv"
			sudo kill -9 $(pidof stress)
		done
	}

	just vm 1 4G big > /dev/null 2> /dev/null &
	sleep 40
	measure vm run_vm
	just ssh_local "poweroff"
	sleep 40

	measure process run_process

# ─── Parallel sweep (iii): avg. throughput vs. #apps ────────────────────────
#
# Sweeps n_parallel from 1 to n_parallel_max.  All n_parallel requests arrive
# at t=0 (ns=np so no staggered arrivals, no preemption) giving a clean N-way
# parallel execution for each data point.  Per-request throughput is extracted
# from DONE lines and written to a CSV for plotting.
#
# KV cache for Llama-3.2-1B-Instruct-f16 (n_layer=16, n_head_kv=8, head_dim=64, f16):
#   32 KB per token  →  n_ctx = (2·N+1)·tokens_per_seq  →  sized per run below.
#
# Run:   just bench_parallel_sweep
# Sweeps n_parallel=1..n_parallel_max, repeated n_runs times per point.
# Each run exercises ns=np requests so all slots are active, then captures
# per-request throughput and latency (from DONE lines) and memory allocations
# (from llama.cpp's init output) into a single CSV.
#
# CSV columns: n_parallel, n_ctx, run_id, seq_id, latency_s, speed_t_per_s,
#			  model_mib, kv_mib, compute_mib, output_mib
#   — memory columns are constant for all rows sharing the same n_parallel
#
# Plot throughput:  just plot_parallel_sweep <csv>
# Plot latency:	 just plot_latency_sweep <csv>
# Plot KV memory:   just plot_kv_memory <csv>
bench_parallel_sweep n_parallel_max="12" n_predict="20" n_runs="5" tokens_per_seq="512" size_mem="8G":
	#!/usr/bin/env bash
	set -e
	mkdir -p {{OUT_DIR}}/parallel_sweep
	out_csv="{{OUT_DIR}}/parallel_sweep/sweep.csv"
	echo "n_parallel,n_ctx,run_id,seq_id,latency_s,speed_t_per_s,model_mib,kv_mib,compute_mib,output_mib" > "$out_csv"

	for np in $(seq 1 {{n_parallel_max}}); do
		n_ctx=$(( (2 * np + 1) * {{tokens_per_seq}} ))
		append_args="llama-parallel -m /root/models/Llama-3.2-1B-Instruct-f16.gguf -np $np -ns $np -n {{n_predict}} -c $n_ctx -t 1 -ngl 0"

		for run_id in $(seq 1 {{n_runs}}); do
			log="{{OUT_DIR}}/parallel_sweep/run_np${np}_run${run_id}.log"
			echo "=== n_parallel=$np run=$run_id/${{n_runs}} (n_ctx=$n_ctx) ===" | tee "$log"
			just -f {{justfile()}} _qemu {{parallel_img}} "$append_args" {{size_mem}} \
				2>&1 | tee -a "$log"

			model_mib=$(grep -oP 'model buffer size\s*=\s*\K[\d.]+' "$log" | head -1)
			kv_mib=$(grep -oP 'KV self size\s*=\s*\K[\d.]+' "$log" | head -1)
			compute_mib=$(grep -oP 'compute buffer size\s*=\s*\K[\d.]+' "$log" | head -1)
			output_mib=$(grep -oP 'output buffer size\s*=\s*\K[\d.]+' "$log" | head -1)

			if [ -z "$kv_mib" ]; then
				echo "WARNING: could not parse memory lines for np=$np run=$run_id" >&2
			fi

			# One row per completed request; memory columns are constant for this np
			grep ": DONE" "$log" | while IFS= read -r line; do
				seq_id=$(echo "$line" | grep -oP 'seq\s+\K\d+(?=/)')
				latency=$(echo "$line" | grep -oP 'total\s+\K[\d.]+(?=\s+s)')
				speed=$(echo "$line" | grep -oP 'speed\s+\K[\d.]+')
				if [ -n "$seq_id" ] && [ -n "$speed" ]; then
					echo "$np,$n_ctx,$run_id,$seq_id,${latency:-0},$speed,${model_mib:-0},${kv_mib:-0},${compute_mib:-0},${output_mib:-0}" >> "$out_csv"
				fi
			done
		done
	done

	echo "Sweep complete. Results: $out_csv"


# ─── Context-switch latency (i) ──────────────────────────────────────────────
#
# Runs the parallel benchmark with np=2 slots, ns=8 requests staggered every
# arrival_interval_ms ms with cycling priorities 0→1→2, which guarantees
# several preemptions and resumes alongside the normal completions.
#
# Each KV-cache switch emits a line:
#   [t=  X.XX s] CTX_SWITCH type=<complete|preempt|resume>  n_tokens=NNN elapsed_us=NNN
#
# The recipe captures those lines into a CSV for plot_ctx_switch.
# Run:   just bench_ctx_switch
# Plot:  just plot_ctx_switch <csv>
bench_ctx_switch n_parallel="2" n_seq="8" n_predict="20" arrival_interval_ms="8000" size_mem="8G":
	#!/usr/bin/env bash
	set -e
	mkdir -p {{OUT_DIR}}/ctx_switch
	log="{{OUT_DIR}}/ctx_switch/run.log"
	out_csv="{{OUT_DIR}}/ctx_switch/ctx_switch.csv"

	# Context sized for (2·np+1) × 512 tokens (enough for 20-token requests)
	n_ctx=$(( (2 * {{n_parallel}} + 1) * 512 ))
	append_args="llama-parallel -m /root/models/Llama-3.2-1B-Instruct-f16.gguf -np {{n_parallel}} -ns {{n_seq}} -n {{n_predict}} -c $n_ctx -t 1 -ngl 0 --arrival-interval-ms {{arrival_interval_ms}}"

	just -f {{justfile()}} _qemu {{parallel_img}} "$append_args" {{size_mem}} \
		2>&1 | tee "$log"

	echo "type,n_tokens,elapsed_us" > "$out_csv"
	while IFS= read -r line; do
		sw_type=$(echo "$line" | grep -oP 'type=\K\S+'	   || true)
		n_tok=$(echo "$line"   | grep -oP 'n_tokens=\s*\K\d+' || true)
		elapsed=$(echo "$line" | grep -oP 'elapsed_us=\K\d+'  || true)
		if [ -n "$sw_type" ] && [ -n "$n_tok" ] && [ -n "$elapsed" ]; then
			echo "$sw_type,$n_tok,$elapsed" >> "$out_csv"
		fi
	done < <(grep "CTX_SWITCH" "$log" || true)

	echo "Done. CSV: $out_csv"


# just plot_parallel_sweep bench/out/parallel_sweep/sweep.csv && just plot_latency_sweep bench/out/parallel_sweep/sweep.csv && just plot_kv_memory bench/out/parallel_sweep/sweep.csv && just plot_ctx_switch bench/out/ctx_switch/ctx_switch.csv
