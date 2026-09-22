# KV-cache swapping sweep: when a request has to give its context up, is it
# cheaper to swap the KV cache out and back in, or to drop it and recompute it
# from the prompt? Both are a time that grows with the context length N; the
# crossover is where the two curves meet.
#
# One kv-bench run (examples/kv-bench in app/llama.cpp) covers every length:
# for each N it builds a context of N tokens, prefills it (the recompute cost),
# times a decode step with everything resident, swaps the KV out and back in
# with both legs timed, and times the first decode step after that.
#
# Manual form (the guest):
#   python3 miniosv/scripts/run.py --arch aarch64 -p kvm -c 4 -m 8G --nogdb \
#     --image-path miniosv/build/cpu.aarch64/loader.img \
#     --emulated-nvme /scratch/orangepi/models/Llama-3.2-1B-Instruct-f16.gguf \
#     --emulated-nvme /scratch/orangepi/kvswap/area.img --nvme-cache none \
#     --args "--kv-swap nvme:2 --kv-page-size 4K kv-bench -m nvme:1 -fa -t 4 --lros \
#             --lengths 128,256,512 --reps 2 --swap page --restore bulk"
# and the host:
#   llama-kv-bench -m model.gguf -fa -t 4 --lengths 128,256,512 --reps 2 \
#     --swap file --swap-file /scratch/orangepi/kvswap/state.bin
#
# Dimensions (all swept, innermost last):
#   run     repetition of the whole boot / process
#   model   KVSWAP_MODELS, always read from KVSWAP_MODELS_DIR (the model's
#           own disk is not a dimension here: it is resident throughout)
#   disk    where the swapped KV goes: nvme = KVSWAP_NVME_DIR, emmc =
#           KVSWAP_EMMC_DIR (under this checkout, on the eMMC)
#   config  linux-file      llama-kv-bench on the host; the KV of the sequence
#                           is written with llama_state_seq_save_file, fsync'ed,
#                           dropped from the page cache and read back with
#                           llama_state_seq_load_file. What a slot engine does.
#           lros-<page>     miniosv, the KV cache in a page-cache window over a
#                           raw NVMe area (--kv-swap nvme:2) with <page> as the
#                           fault/eviction granularity (4K, 64K, ...). The swap
#                           out squeezes the window to one page (every dirty
#                           page is written); the swap in reads every page that
#                           left, in bulk, before the next decode.
#           lros-<page>-lazy
#                           the same, but nothing is read back ahead: the next
#                           decode step faults its pages in one by one. T_IN is
#                           0 and the cost shows in T_DEC2 - T_DEC.
#           lros-<page>[-lazy]-keep<pct>
#                           the partial swap out: the window is squeezed until
#                           <pct>% of the KV is left (the policy picks what
#                           stays) and only what left is read back. The
#                           lengths must be multiples of 256: kv-bench then
#                           sizes the context to exactly N cells so the window
#                           holds nothing but this KV. The state file has no
#                           counterpart, it always holds the whole sequence.
#                           `just kvswap_partial` runs KVSWAP_PARTIAL_CONFIGS
#                           at the one length KVSWAP_PARTIAL_LENGTHS.
#
# Both sides use flash attention (-fa): with the KV paged, V is stored
# transposed without it and every token touches every page of V. The guest
# computes on lros workers (--lros); ggml's own threadpool is not placed on
# the cores by the current miniosv.
#
# Output: bench/out/kvswap/<timestamp>/kvswap.csv, one row per (N, rep) of
# each run, and the console logs under logs/<model>/<disk>/<config>/run<n>.
# The figures go to bench/plots/kvswap/ (just plot_kvswap).

KVSWAP_MODELS := "Llama-3.2-1B-Instruct-f16.gguf Llama-3.2-1B-Instruct-Q4_K_M.gguf qwen2.5-1.5b-instruct-fp16.gguf"
KVSWAP_DISKS := "nvme emmc"
KVSWAP_CONFIGS := "linux-file lros-64K lros-4K-lazy"
# The partial swap-out set: how much of the KV stays resident, at one context
# length (kvswap_partial). keep0 is the full swap in the same exact layout;
# linux-file is the state file, which always moves the whole KV and stands in
# every group of the figure, and it gives the Linux recompute reference.
KVSWAP_PARTIAL_CONFIGS := "linux-file lros-64K-keep0 lros-64K-keep10 lros-64K-keep50 lros-64K-keep90"
KVSWAP_PARTIAL_LENGTHS := "1024"
KVSWAP_LENGTHS := "64,128,256,512,1024,2048"
# Repetitions of every length inside one run (the whole run is repeated n_runs
# times on top).
KVSWAP_REPS := "2"
# Where the models are read from (the NVMe copy).
KVSWAP_MODELS_DIR := "/scratch/orangepi/models"
KVSWAP_NVME_DIR := "/scratch/orangepi/kvswap"
KVSWAP_EMMC_DIR := proot + "/bench/kvswap"
# The raw NVMe area the guest swaps into; big enough for the longest length of
# the largest model (2048 tokens of Llama-3.2-1B f16 are 64 MiB).
KVSWAP_AREA := "1G"
# The Linux baseline is the same app tree (miniosv/app/llama.cpp, which has the
# kv-bench example) built natively with the cpu-clang preset, so both sides
# use one compiler:
#   just LLAMACPP_RKNN_PATH=$PWD/miniosv/app/llama.cpp LLAMA_CPP_TARGETS=llama-kv-bench \
#     build linux cpu-clang
KVSWAP_LINUX_BIN := LROS_PATH + "/app/llama.cpp/build-cpu-clang/bin/llama-kv-bench"
KVSWAP_BENCH_ARGS := "-fa -t 4"
# Wall-clock cap per run (seconds). One run prefills the sum of the lengths,
# reps times: about 8k tokens x 2 at 80 t/s is 4 minutes, plus the swaps.
KVSWAP_TIMEOUT := "1800"

kvswap_sweep n_runs="1" size_mem="8G" variant="cpu" models=KVSWAP_MODELS disks=KVSWAP_DISKS configs=KVSWAP_CONFIGS lengths=KVSWAP_LENGTHS:
	#!/usr/bin/env bash
	set -euo pipefail

	ts="$(date -Iseconds)"
	out_dir="{{OUT_DIR}}/kvswap/$ts"
	mkdir -p "$out_dir"
	csv="$out_dir/kvswap.csv"
	img="{{LROS_PATH}}/build/{{variant}}.aarch64/{{LROS_IMAGE_NAME}}"
	bench_args="{{KVSWAP_BENCH_ARGS}} --lengths {{lengths}} --reps {{KVSWAP_REPS}}"

	case " {{configs}} " in *" lros-"*)
		if [ ! -f "$img" ]; then
			echo "Missing $img -- build it first: just build lros {{variant}}" >&2
			exit 1
		fi
	esac
	case " {{configs}} " in *" linux-"*)
		if [ ! -x "{{KVSWAP_LINUX_BIN}}" ]; then
			echo "Missing {{KVSWAP_LINUX_BIN}} -- see the header of scripts/kvswap.just" >&2
			exit 1
		fi
	esac

	# "64K" -> bytes
	to_bytes() {
		local n="${1%[KMGkmg]}" u="${1##*[0-9]}"
		case "$u" in
			K|k) echo $((n << 10)) ;;
			M|m) echo $((n << 20)) ;;
			G|g) echo $((n << 30)) ;;
			"")  echo "$n" ;;
			*)   echo "bad size: $1" >&2; return 1 ;;
		esac
	}

	# The directory the swapped KV goes to on a disk.
	swap_dir() { # disk
		case "$1" in
			nvme) echo "{{KVSWAP_NVME_DIR}}" ;;
			emmc) echo "{{KVSWAP_EMMC_DIR}}" ;;
			*) echo "unknown disk $1" >&2; return 1 ;;
		esac
	}
	# The guest's raw swap area on a disk, made on first use.
	swap_area() { # disk
		local dir p
		dir=$(swap_dir "$1"); p="$dir/area.img"
		if [ ! -f "$p" ]; then
			mkdir -p "$dir"
			fallocate -l {{KVSWAP_AREA}} "$p"
		fi
		echo "$p"
	}

	# Parse the kv-bench markdown table (on stdin) into CSV rows.
	# Columns: | N | REP | KV_MB | T_PP ms | T_DEC ms | T_OUT ms | MB_OUT | T_IN ms | MB_IN | T_DEC2 ms |
	# $1 = the fixed row prefix, $2 = status
	to_csv() {
		awk -F'|' -v prefix="$1" -v status="$2" '
			/^\|[[:space:]]*[0-9]/ {
				for (i = 1; i <= NF; i++) gsub(/[[:space:]]/, "", $i)
				printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", prefix,
					$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, status
				n++
			}
			END { if (!n) printf "%s,,,,,,,,,,,%s\n", prefix, status }'
	}

	status_of() { # exit code
		case "$1" in 0) echo ok ;; 124|137) echo timeout ;; *) echo "failed($1)" ;; esac
	}

	# ---- the runners; each writes stdout/stderr to the last two args ----
	# stdin from /dev/null: under timeout QEMU's -serial mon:stdio would get
	# SIGTTOU from a background process group and the boot would stall.
	run_linux() { # model_path state_file out err
		vmtouch -qe "$1"
		rm -f "$2"
		timeout -k 10 {{KVSWAP_TIMEOUT}} \
			taskset -c {{AFFINITY}} {{KVSWAP_LINUX_BIN}} -m "$1" $bench_args \
			--swap file --swap-file "$2" \
			< /dev/null > "$3" 2> "$4"
	}
	run_lros() { # model_path area page_bytes restore keep_pct out err
		vmtouch -qe "$1"
		vmtouch -qe "$2"
		timeout -k 10 {{KVSWAP_TIMEOUT}} \
			taskset -c {{AFFINITY}} python3 {{LROS_PATH}}/scripts/run.py \
			--arch aarch64 -p kvm -c 4 -m {{size_mem}} --nogdb \
			--image-path "$img" --emulated-nvme "$1" --emulated-nvme "$2" --nvme-cache none \
			--args "--kv-swap nvme:2 --kv-page-size $(($3 >> 10))K kv-bench -m nvme:1 --lros $bench_args --swap page --restore $4 ${5:+--keep $5}" \
			< /dev/null > "$6" 2> "$7"
	}

	{
		echo "timestamp $ts"
		echo "image $img"
		echo "host binary {{KVSWAP_LINUX_BIN}}"
		echo "bench args $bench_args"
		echo "guest memory {{size_mem}}, cpus {{AFFINITY}}"
		echo "area {{KVSWAP_AREA}}, timeout {{KVSWAP_TIMEOUT}}s"
		echo "models {{models}}"; echo "disks {{disks}}"; echo "configs {{configs}}"; echo "lengths {{lengths}}"
		echo "git $(git -C {{proot}} rev-parse --short HEAD) miniosv $(git -C {{LROS_PATH}} rev-parse --short HEAD) app $(git -C {{LROS_PATH}}/app/llama.cpp rev-parse --short HEAD)"
	} > "$out_dir/sweep.info"

	models=""
	for model in {{models}}; do
		if [ ! -f "{{KVSWAP_MODELS_DIR}}/$model" ]; then
			echo "WARNING: {{KVSWAP_MODELS_DIR}}/$model does not exist; skipping this model" >&2
			continue
		fi
		models="$models $model"
	done
	if [ -z "$models" ]; then echo "no model to run" >&2; exit 1; fi

	echo "timestamp,run,model,disk,config,page_bytes,restore,keep_pct,n,rep,kv_mb,t_pp_ms,t_dec_ms,t_out_ms,mb_out,t_in_ms,mb_in,t_dec2_ms,status" > "$csv"

	for run in $(seq 1 {{n_runs}}); do
	for model in $models; do
		path="{{KVSWAP_MODELS_DIR}}/$model"
	for disk in {{disks}}; do
	for config in {{configs}}; do
		log_dir="$out_dir/logs/${model%.gguf}/$disk/$config"
		mkdir -p "$log_dir"
		out="$log_dir/run$run.out.txt"; err="$log_dir/run$run.err.txt"
		echo "=== run $run/{{n_runs}}: $model, KV to $disk, $config ==="
		rc=0
		case "$config" in
			linux-file)
				dir=$(swap_dir "$disk"); mkdir -p "$dir"
				run_linux "$path" "$dir/state.bin" "$out" "$err" || rc=$?
				prefix="$ts,$run,$model,$disk,$config,,file,"
				;;
			lros-*)
				# lros-<page>[-lazy][-keep<pct>]
				spec="${config#lros-}"
				# keep stays empty outside the partial experiment: kv-bench
				# then swaps everything in its ordinary layout.
				restore=bulk; keep=""
				case "$spec" in *-keep*) keep="${spec##*-keep}"; spec="${spec%-keep*}" ;; esac
				case "$spec" in *-lazy) restore=lazy; spec="${spec%-lazy}" ;; esac
				page=$(to_bytes "$spec")
				area=$(swap_area "$disk")
				run_lros "$path" "$area" "$page" "$restore" "$keep" "$out" "$err" || rc=$?
				prefix="$ts,$run,$model,$disk,$config,$page,$restore,$keep"
				;;
			*)
				echo "unknown config $config" >&2; exit 1 ;;
		esac
		status=$(status_of "$rc")
		[ "$status" = ok ] || echo "  -> $status, see $err" >&2
		to_csv "$prefix" "$status" < "$out" >> "$csv"
	done; done; done; done

	echo "Wrote $csv"
	column -t -s, "$csv"

# The partial swap-out experiment: the same sweep over KVSWAP_PARTIAL_CONFIGS
# at one context length, i.e. how the swap costs shrink when 10, 50 or 90% of
# the KV may stay.
kvswap_partial n_runs="1" size_mem="8G" variant="cpu" models=KVSWAP_MODELS disks=KVSWAP_DISKS lengths=KVSWAP_PARTIAL_LENGTHS: (kvswap_sweep n_runs size_mem variant models disks KVSWAP_PARTIAL_CONFIGS lengths)
