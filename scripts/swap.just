# Weight-swapping sweep: how well does inference hold up when the model does
# not fit in memory and has to be paged in from disk?
#
# Manual form of the measurement (what this recipe automates):
#   just MODEL=/scratch/orangepi/models/Llama-3.2-1B-Instruct-f16.gguf \
#     run_lros_stdout miniosv/build/cpu.aarch64/loader.img 8G \
#     "--cache-size 1182M --buffer-size 128K --prefetch-depth 128 \
#      batched-bench -m nvme:1 -npl 1 -ntg 1,16 -npp 128 -t 4"
#
# Dimensions (all swept, innermost last):
#   run     repetition, so drift shows up as spread rather than as a trend
#   model   SWAP_MODELS; a name missing from SWAP_NVME_MODELS_DIR is skipped
#   disk    nvme  = the model in SWAP_NVME_MODELS_DIR (/scratch, NVMe)
#           emmc  = the model in models/ (this checkout, eMMC); copied from
#                   the nvme dir on first use
#   quota   percent of the model file size the paging layer may keep resident
#   config  linux-mmap  llama-batched-bench on the host, model mmap'ed, in a
#                       cgroup whose MemoryMax caps anon + page cache
#           lros-<buf>  miniosv, page cache buffers of <buf> (128K, 2M, ...),
#                       with the default policy: --pin-percent 75 holds the
#                       first 75% of the cache as a resident prefix of the
#                       model and streams the rest through what is left
#           lros-<buf>-nopin
#                       the same with --pin-percent 0, i.e. the S3-FIFO
#                       policy over the whole cache
#
# Quota accounting. On miniosv --cache-size bounds only the model's page
# cache, so the quota is used as is. A cgroup MemoryMax also counts everything
# else the process needs (KV cache, compute buffers, the binary, page tables),
# so the Linux quota is quota + overhead, where overhead is the peak RssAnon of
# an unlimited calibration run of the same command, measured once per model,
# plus SWAP_LINUX_MARGIN for the kernel memory the cgroup charges as well.
#
# Prefetch. --prefetch-depth is chosen so that SWAP_INFLIGHT worth of buffers
# is read ahead of every fault (128 x 128 KiB, 8 x 2 MiB for 16M). Note that
# miniosv clamps the depth to prefetch_max (include/osv/mem/pagecache.hh, 512)
# and parks at most pending_slots (core/mem/pagecache/internal.hh, 64)
# prefetched buffers; beyond 64 the fault handler reads the remaining
# candidates synchronously, so a depth above 64 lengthens every fault rather
# than adding lookahead. The recipe warns when that applies. The NVMe queue
# (15 commands per vCPU, 2 MiB each at --nvme-mdts 9) bounds what is really
# in flight on the device.
#
# Host page cache. The model drive is attached with cache=none (O_DIRECT) so
# QEMU does not serve the second pass over the model from host RAM, and the
# file is evicted with vmtouch before every run, on both sides.
#
# Output: bench/out/swap/<timestamp>/swap.csv, one row per batched-bench
# table line (two per run: ntg=1 and ntg=16), calibration.csv with the
# measured Linux overhead per model, and the raw console logs under
# logs/<model>/<disk>/q<quota>/<config>/run<n>.{out,err}.txt (the calibration
# run's log is logs/<model>/calibrate-linux.log). The figures go to
# bench/plots/swap/ (just plot_swap).

SWAP_NVME_MODELS_DIR := "/scratch/orangepi/models"
SWAP_EMMC_MODELS_DIR := models_dir

# Model file names, resolved in the two model dirs above. Placeholders: add a
# low-bit quantisation of the Llama model (e.g. Llama-3.2-1B-Instruct-Q4_K_M.gguf
# or Q2_K from the same bartowski repo as in models_to_get) and a second
# architecture (qwen2.5-1.5b-instruct-fp16.gguf or gemma-3-1b-it-f16.gguf are
# already in the nvme dir). Names not present in SWAP_NVME_MODELS_DIR are
# skipped with a warning, so they can be listed before they are downloaded.
#SWAP_MODELS := "Llama-3.2-1B-Instruct-f16.gguf"
SWAP_MODELS := "Llama-3.2-1B-Instruct-f16.gguf Llama-3.2-1B-Instruct-Q4_K_M.gguf qwen2.5-1.5b-instruct-fp16.gguf"

SWAP_DISKS := "nvme emmc"
# The -nopin arms are left out of the default: they add nothing over the
# pinned ones (measured 2026-09-17); the recipe still accepts them by name.
SWAP_CONFIGS := "linux-mmap lros-128K lros-2M"
SWAP_QUOTAS := "110 80 50 10"
SWAP_INFLIGHT := "16M"
SWAP_LINUX_MARGIN := "64M"
SWAP_BENCH_ARGS := "-npl 1 -ntg 1,16 -npp 128 -t 4"
# Wall-clock cap per run (seconds); a wedged guest must not stall the sweep.
# The slowest completed run so far (eMMC, 50%, 2M) took about 5 minutes.
SWAP_TIMEOUT := "600"

swap_sweep n_runs="3" size_mem="8G" variant="cpu" models=SWAP_MODELS disks=SWAP_DISKS configs=SWAP_CONFIGS quotas=SWAP_QUOTAS:
	#!/usr/bin/env bash
	set -euo pipefail

	ts="$(date -Iseconds)"
	out_dir="{{OUT_DIR}}/swap/$ts"
	mkdir -p "$out_dir"
	csv="$out_dir/swap.csv"
	calib="$out_dir/calibration.csv"
	img="{{LROS_PATH}}/build/{{variant}}.aarch64/{{LROS_IMAGE_NAME}}"
	bench_args="{{SWAP_BENCH_ARGS}}"

	case " {{configs}} " in *" lros-"*)
		if [ ! -f "$img" ]; then
			echo "Missing $img -- build it first: just build lros {{variant}}" >&2
			exit 1
		fi
	esac
	case " {{configs}} " in *" linux-"*)
		if [ ! -x "{{LLAMACPP_BIN}}" ]; then
			echo "Missing {{LLAMACPP_BIN}} -- build it first: just build linux cpu" >&2
			exit 1
		fi
	esac

	# "16M" -> bytes
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
	inflight=$(to_bytes {{SWAP_INFLIGHT}})
	margin=$(to_bytes {{SWAP_LINUX_MARGIN}})

	# Path of a model on a disk; the eMMC copy is made on first use.
	model_path() { # model disk
		case "$2" in
			nvme) echo "{{SWAP_NVME_MODELS_DIR}}/$1" ;;
			emmc)
				local p="{{SWAP_EMMC_MODELS_DIR}}/$1"
				if [ ! -f "$p" ]; then
					echo "copying $1 to the eMMC model dir ($p)" >&2
					mkdir -p "{{SWAP_EMMC_MODELS_DIR}}"
					cp "{{SWAP_NVME_MODELS_DIR}}/$1" "$p.part" && mv "$p.part" "$p"
					sync
				fi
				echo "$p" ;;
			*) echo "unknown disk $2" >&2; return 1 ;;
		esac
	}

	# Parse a llama-batched-bench markdown table (on stdin) into CSV rows.
	# Columns: | PP | TG | B | N_KV | T_PP s | S_PP | T_TG s | S_TG | T s | S |
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

	# Peak anonymous RSS (bytes) of an unlimited host run: everything the
	# process needs besides the mmap'ed model. Sampled from /proc while it runs.
	calibrate_linux() { # model_path log
		local peak=0 pid rss
		vmtouch -qe "$1"
		taskset -c {{AFFINITY}} {{LLAMACPP_BIN}} $bench_args -m "$1" > "$2" 2>&1 &
		pid=$!
		while kill -0 "$pid" 2>/dev/null; do
			rss=$(awk '/^RssAnon:/ { print $2 }' "/proc/$pid/status" 2>/dev/null || true)
			[ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
			sleep 0.1
		done
		wait "$pid" || { echo "calibration run failed, see $2" >&2; return 1; }
		echo $((peak * 1024))
	}

	# ---- the runners; each writes stdout/stderr to $3/$4 and returns the rc ----
	# stdin comes from /dev/null on purpose: timeout puts the command in its own
	# process group, and QEMU's -serial mon:stdio then tries to put the
	# terminal into raw mode from a background group -- SIGTTOU stops it before
	# the guest even boots, and the run sits there until the timeout.
	run_linux() { # model_path quota_bytes out err
		vmtouch -qe "$1"
		systemd-run --user --scope -p MemoryMax="$2" -p MemorySwapMax=0 --quiet -- \
			timeout -k 10 {{SWAP_TIMEOUT}} \
			taskset -c {{AFFINITY}} {{LLAMACPP_BIN}} $bench_args -m "$1" \
			< /dev/null > "$3" 2> "$4"
	}
	# The guest has no load balancer: ggml's threads are pinned one per vcpu,
	# or they all run on one core.
	run_lros() { # model_path cache_kib buffer_bytes depth extra_opts out err
		vmtouch -qe "$1"
		timeout -k 10 {{SWAP_TIMEOUT}} \
			taskset -c {{AFFINITY}} python3 {{LROS_PATH}}/scripts/run.py \
			--arch aarch64 -p kvm -c 4 -m {{size_mem}} --nogdb \
			--image-path "$img" --emulated-nvme "$1" --nvme-cache none --nvme-aio io_uring --nvme-mdts 9 \
			--args "--cache-size $2K --buffer-size $(($3 >> 10))K --prefetch-depth $4 $5 batched-bench -m nvme:1 $bench_args -C f --cpu-strict 1 -Cb f --cpu-strict-batch 1" \
			< /dev/null > "$6" 2> "$7"
	}

	{
		echo "timestamp $ts"
		echo "image $img"
		echo "host binary {{LLAMACPP_BIN}}"
		echo "bench args $bench_args"
		echo "guest memory {{size_mem}}, cpus {{AFFINITY}}"
		echo "inflight {{SWAP_INFLIGHT}}, linux margin {{SWAP_LINUX_MARGIN}}, timeout {{SWAP_TIMEOUT}}s"
		echo "models {{models}}"; echo "disks {{disks}}"; echo "configs {{configs}}"; echo "quotas {{quotas}}"
		echo "git $(git -C {{proot}} rev-parse --short HEAD) miniosv $(git -C {{LROS_PATH}} rev-parse --short HEAD)"
	} > "$out_dir/sweep.info"

	# Calibration and model sizes, once per model.
	declare -A model_bytes linux_overhead
	echo "model,model_bytes,linux_overhead_bytes" > "$calib"
	models=""
	for model in {{models}}; do
		if [ ! -f "{{SWAP_NVME_MODELS_DIR}}/$model" ]; then
			echo "WARNING: {{SWAP_NVME_MODELS_DIR}}/$model does not exist; skipping this model" >&2
			continue
		fi
		models="$models $model"
		model_bytes[$model]=$(stat -c %s "{{SWAP_NVME_MODELS_DIR}}/$model")
		linux_overhead[$model]=0
		case " {{configs}} " in *" linux-"*)
			echo "=== calibrating Linux overhead for $model ==="
			mkdir -p "$out_dir/logs/${model%.gguf}"
			linux_overhead[$model]=$(calibrate_linux "{{SWAP_NVME_MODELS_DIR}}/$model" "$out_dir/logs/${model%.gguf}/calibrate-linux.log")
			echo "  peak RssAnon $(( ${linux_overhead[$model]} >> 20 )) MiB (+ margin $(( margin >> 20 )) MiB)"
		esac
		echo "$model,${model_bytes[$model]},${linux_overhead[$model]}" >> "$calib"
	done
	if [ -z "$models" ]; then echo "no model to run" >&2; exit 1; fi

	echo "timestamp,run,model,disk,config,quota_pct,quota_bytes,linux_overhead_bytes,buffer_bytes,prefetch_depth,pp,tg,b,n_kv,t_pp_s,s_pp,t_tg_s,s_tg,t_s,s,status" > "$csv"

	for run in $(seq 1 {{n_runs}}); do
	for model in $models; do
	for disk in {{disks}}; do
		path=$(model_path "$model" "$disk")
	for pct in {{quotas}}; do
		quota=$(( ${model_bytes[$model]} * pct / 100 ))
	for config in {{configs}}; do
		# One folder per cell of the sweep; the repetitions sit side by side in it.
		log_dir="$out_dir/logs/${model%.gguf}/$disk/q$pct/$config"
		mkdir -p "$log_dir"
		out="$log_dir/run$run.out.txt"; err="$log_dir/run$run.err.txt"
		echo "=== run $run/{{n_runs}}: $model on $disk, quota $pct% ($(( quota >> 20 )) MiB), $config ==="
		rc=0
		case "$config" in
			linux-mmap)
				limit=$(( quota + ${linux_overhead[$model]} + margin ))
				run_linux "$path" "$limit" "$out" "$err" || rc=$?
				prefix="$ts,$run,$model,$disk,$config,$pct,$limit,${linux_overhead[$model]},,"
				;;
			lros-*)
				# lros-<buf>[-nopin]: the suffix selects --pin-percent 0
				spec="${config#lros-}"
				extra=""
				case "$spec" in *-nopin) extra="--pin-percent 0"; spec="${spec%-nopin}" ;; esac
				buffer=$(to_bytes "$spec")
				depth=$(( inflight / buffer ))
				if [ "$depth" -gt 64 ]; then
					echo "WARNING: depth $depth exceeds miniosv's pending_slots (64); the excess is read synchronously in the fault, see the header of scripts/swap.just" >&2
				fi
				run_lros "$path" $(( (quota + 1023) >> 10 )) "$buffer" "$depth" "$extra" "$out" "$err" || rc=$?
				prefix="$ts,$run,$model,$disk,$config,$pct,$quota,,$buffer,$depth"
				;;
			*)
				echo "unknown config $config" >&2; exit 1 ;;
		esac
		status=$(status_of "$rc")
		[ "$status" = ok ] || echo "  -> $status, see $err" >&2
		to_csv "$prefix" "$status" < "$out" >> "$csv"
	done; done; done; done; done

	echo "Wrote $csv"
	column -t -s, "$csv"
