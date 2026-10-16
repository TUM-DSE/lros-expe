# Weight-swapping sweep: how well does inference hold up when the model does
# not fit in memory and has to be paged in from disk?
#
# Manual form of the measurement (what this recipe automates), for the
# lros-tensor-layer arm at a 50% quota:
#   just MODEL=/scratch/orangepi/models/Llama-3.2-1B-Instruct-f16.gguf \
#     run_lros_stdout miniosv/build/paging.aarch64/loader.img 8G \
#     "--cache-size 1182M --pin-percent 1 --per-tensor-buffers --layer-prefetch \
#      batched-bench -m nvme:1 -npl 1 -ntg 1,16 -npp 128 -t 4 \
#      -C f --cpu-strict 1 -Cb f --cpu-strict-batch 1"
#
# The guest image is the variant's build dir, by default paging.aarch64, built
# from the current miniosv and app/llama.cpp trees:
#   nix develop ./miniosv --command make -C miniosv -j8 app=app/llama.cpp \
#     arch=aarch64 out=build/paging.aarch64 conf_vaccel=0 conf_llama_sgemm=1
# The Linux side runs llama-batched-bench from the same llama.cpp tree, built
# with the same clang major as the guest (in the dev shell of this repo):
#   nix develop --command cmake --build miniosv/app/llama.cpp/build-cpu-clang \
#     --target llama-batched-bench
# (and --target llama-cli for SWAP_FRONTEND=cli).
#
# Frontend (SWAP_FRONTEND):
#   batched-bench  a synthetic prompt: the same token at every position, in
#                  the prompt and in every decode step. Fine for a dense
#                  model, which reads all its weights whatever the text; a
#                  mixture of experts routes such input to the same few
#                  experts, so its working set is far below the file.
#   cli            llama-cli on real text (SWAP_CLI_PROMPT) with sampled
#                  tokens, as the motivation runs: a mixture of experts then
#                  routes as in use. The prompt reaches the guest on an ext4
#                  data disk mounted at /root, the model being the second
#                  drive. One row per run.
#   just SWAP_FRONTEND=cli swap_sweep 3 8G paging olmoe-1b-7b-0924-instruct-q4_k_m.gguf nvme
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
#           lros-tensor miniosv, a page cache buffer per tensor piece (a
#                       tensor's own pages, at most 1/16 of the cache; the
#                       pages two tensors share are pieces of their own, read
#                       together with either neighbour). The first pieces of
#                       the model's use order are pinned, the rest is
#                       streamed through a window of three of the largest
#                       pieces plus 16 MiB.
#           lros-tensor-layer
#                       the same with --layer-prefetch: on entering a layer,
#                       the pieces that follow in the use order are read
#                       ahead within the window
#           lros-<buf>[-nopin]
#                       page cache buffers of a fixed <buf> (128K, 2M, ...)
#                       with SWAP_INFLIGHT of sequential read-ahead. Pinning
#                       needs the use order, which fixed buffers never
#                       complete, so only the -nopin form (--pin-percent 0,
#                       S3-FIFO over the whole cache) is a sensible arm; the
#                       pinned form ends up as a plain FIFO.
#
# Both sides compute on the same four A76 cores (SWAP_AFFINITY). In the guest
# ggml's threads are pinned one per vcpu: it has no load balancer, so
# unpinned threads would all run on one vcpu. On the host they are left to
# the Linux scheduler within those cores.
#
# Quota accounting. On miniosv --cache-size bounds only the model's page
# cache, so the quota is used as is. A cgroup MemoryMax also counts everything
# else the process needs (KV cache, compute buffers, the binary, page tables),
# so the Linux quota is quota + overhead, where overhead is the peak RssAnon of
# an unlimited calibration run of the same command, measured once per model,
# plus SWAP_LINUX_MARGIN for the kernel memory the cgroup charges as well.
#
# Prefetch, for the fixed-buffer arms. --prefetch-depth is chosen so that
# SWAP_INFLIGHT worth of buffers is read ahead of every fault (128 x 128 KiB,
# 8 x 2 MiB for 16M). The per-tensor arms have no such read-ahead. Note that
# miniosv clamps the depth to prefetch_max (include/osv/mem/pagecache.hh, 512)
# and parks at most pending_slots (core/mem/pagecache/internal.hh, 256)
# prefetched buffers; beyond 256 the fault handler reads the remaining
# candidates synchronously, so a depth above 256 lengthens every fault rather
# than adding lookahead. The recipe warns when that applies. The NVMe queue
# (15 commands per vCPU, 2 MiB each at --nvme-mdts 9) bounds what is really
# in flight on the device.
#
# Host page cache. The model drive is attached with cache=none (O_DIRECT) so
# QEMU does not serve the second pass over the model from host RAM, and the
# file is evicted with vmtouch before every run, on both sides.
#
# Output: bench/out/swap/<timestamp>/swap.csv, one row per batched-bench
# table line (two per run: ntg=1 and ntg=16) or one per llama-cli run (pp
# and tg as its timings count them), calibration.csv with the
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
# The per-tensor arms with and without layer prefetch, and fixed 2M buffers as
# the reference for what per-tensor buffers and pinning add.
SWAP_CONFIGS := "linux-mmap lros-tensor lros-tensor-layer lros-2M-nopin"
SWAP_QUOTAS := "110 80 50 10"
SWAP_INFLIGHT := "16M"
SWAP_LINUX_MARGIN := "64M"
SWAP_BENCH_ARGS := "-npl 1 -ntg 1,16 -npp 128 -t 4"
# The Linux side: the guest's llama.cpp tree, built with clang 20.
SWAP_LINUX_BIN := LROS_PATH + "/app/llama.cpp/build-cpu-clang/bin/llama-batched-bench"
SWAP_FRONTEND := "batched-bench"
SWAP_CLI_BIN := LROS_PATH + "/app/llama.cpp/build-cpu-clang/bin/llama-cli"
# The motivation's prompt and generation (scripts/motivation.just paging).
SWAP_CLI_PROMPT := "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a screw press adapted from wine making. Within fifty years presses operated in more than two hundred cities across Europe, and the cost of books fell sharply. Scholars, merchants and reformers used the new medium to spread ideas faster than any authority could control. Explain in detail how the printing press changed science, religion, politics and everyday life, and compare its effects with those of the internet."
SWAP_CLI_ARGS := "-no-cnv -n 128 --ignore-eos --seed 1 -c 1024 -t 4"
# Host cpus for both sides: the four A76 cores.
SWAP_AFFINITY := "4-7"
# Wall-clock cap per run (seconds); a wedged guest must not stall the sweep.
# The slowest completed run so far (eMMC, 50%, 2M) took about 5 minutes.
SWAP_TIMEOUT := "600"

swap_sweep n_runs="3" size_mem="8G" variant="paging" models=SWAP_MODELS disks=SWAP_DISKS configs=SWAP_CONFIGS quotas=SWAP_QUOTAS:
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
			echo "Missing $img -- build it first, see the header of scripts/swap.just" >&2
			exit 1
		fi
	esac
	guest_pin="-C f --cpu-strict 1 -Cb f --cpu-strict-batch 1"
	# What runs on each side: host_bin $host_args -m <model>, and in the guest
	# guest_cmd, with the drives in guest_drives ahead of the model's.
	guest_drives=()
	case "{{SWAP_FRONTEND}}" in
		batched-bench)
			host_bin="{{SWAP_LINUX_BIN}}"
			host_args="$bench_args"
			guest_cmd="batched-bench -m nvme:1 $bench_args $guest_pin"
			;;
		cli)
			host_bin="{{SWAP_CLI_BIN}}"
			stage="$out_dir/stage"; mkdir -p "$stage"
			printf '%s' "{{SWAP_CLI_PROMPT}}" > "$stage/prompt.txt"
			host_args="{{SWAP_CLI_ARGS}} -f $stage/prompt.txt"
			guest_cmd="llama-cli -m nvme:2 {{SWAP_CLI_ARGS}} -f /root/prompt.txt $guest_pin"
			case " {{configs}} " in *" lros-"*)
				data="$out_dir/data.img"
				MKDATA_ASSUME_YES=1 nix develop {{LROS_PATH}} --command bash {{LROS_PATH}}/scripts/mkdata.sh "$data" "$stage" 16M > "$out_dir/mkdata.log" 2>&1
				guest_drives=(--emulated-nvme "$data")
			esac
			;;
		*) echo "unknown SWAP_FRONTEND {{SWAP_FRONTEND}}: batched-bench or cli" >&2; exit 1 ;;
	esac
	case " {{configs}} " in *" linux-"*)
		if [ ! -x "$host_bin" ]; then
			echo "Missing $host_bin -- build it first, see the header of scripts/swap.just" >&2
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

	# A run's console (on stdin) as CSV rows. $1 = the fixed row prefix,
	# $2 = status.
	to_csv() {
		case "{{SWAP_FRONTEND}}" in
			batched-bench) bench_to_csv "$@" ;;
			cli)           cli_to_csv "$@" ;;
		esac
	}

	# llama-cli's timings: "prompt eval time = <ms> ms / <n> tokens" and
	# "eval time = <ms> ms / <n> runs", as one row in batched-bench's columns.
	cli_to_csv() {
		tr -d '\r' | awk -v prefix="$1" -v status="$2" '
			/prompt eval time/ { pt = $6; pp = $9 }
			/ +eval time/ && !/prompt/ { tt = $5; tg = $8 }
			END {
				if (pp && tg) {
					t = (pt + tt) / 1000
					printf "%s,%d,%d,1,%d,%.3f,%.2f,%.3f,%.2f,%.3f,%.2f,%s\n", prefix, pp, tg, pp + tg,
						pt / 1000, pp * 1000 / pt, tt / 1000, tg * 1000 / tt, t, (pp + tg) / t, status
				} else printf "%s,,,,,,,,,,,%s\n", prefix, status
			}'
	}

	# Parse a llama-batched-bench markdown table into CSV rows.
	# Columns: | PP | TG | B | N_KV | T_PP s | S_PP | T_TG s | S_TG | T s | S |
	bench_to_csv() {
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
		taskset -c {{SWAP_AFFINITY}} "$host_bin" $host_args -m "$1" < /dev/null > "$2" 2>&1 &
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
			taskset -c {{SWAP_AFFINITY}} "$host_bin" $host_args -m "$1" \
			< /dev/null > "$3" 2> "$4"
	}
	run_lros() { # model_path cache_kib paging_opts out err
		vmtouch -qe "$1"
		timeout -k 10 {{SWAP_TIMEOUT}} \
			taskset -c {{SWAP_AFFINITY}} python3 {{LROS_PATH}}/scripts/run.py \
			--arch aarch64 -p kvm -c 4 -m {{size_mem}} --nogdb \
			--image-path "$img" "${guest_drives[@]}" --emulated-nvme "$1" \
			--nvme-cache none --nvme-aio io_uring --nvme-mdts 9 \
			--args "--cache-size $2K $3 $guest_cmd" \
			< /dev/null > "$4" 2> "$5"
	}

	{
		echo "timestamp $ts"
		echo "image $img"
		echo "frontend {{SWAP_FRONTEND}}"
		echo "host $host_bin $host_args"
		echo "guest $guest_cmd"
		echo "guest memory {{size_mem}}, cpus {{SWAP_AFFINITY}}"
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
			lros-tensor|lros-tensor-layer)
				# A buffer per tensor piece, pinned by the use order; no read-ahead
				# of its own, the layer prefetch in the -layer arm.
				opts="--pin-percent 1 --per-tensor-buffers"
				[ "$config" = lros-tensor-layer ] && opts="$opts --layer-prefetch"
				run_lros "$path" $(( (quota + 1023) >> 10 )) "$opts" "$out" "$err" || rc=$?
				prefix="$ts,$run,$model,$disk,$config,$pct,$quota,,,0"
				;;
			lros-*)
				# lros-<buf>[-nopin]: the suffix selects --pin-percent 0
				spec="${config#lros-}"
				extra=""
				case "$spec" in *-nopin) extra="--pin-percent 0"; spec="${spec%-nopin}" ;; esac
				buffer=$(to_bytes "$spec")
				depth=$(( inflight / buffer ))
				if [ "$depth" -gt 256 ]; then
					echo "WARNING: depth $depth exceeds miniosv's pending_slots (256); the excess is read synchronously in the fault, see the header of scripts/swap.just" >&2
				fi
				opts="--buffer-size $(( buffer >> 10 ))K --prefetch-depth $depth $extra"
				run_lros "$path" $(( (quota + 1023) >> 10 )) "$opts" "$out" "$err" || rc=$?
				prefix="$ts,$run,$model,$disk,$config,$pct,$quota,,$buffer,$depth"
				;;
			*)
				echo "unknown config $config" >&2; exit 1 ;;
		esac
		status=$(status_of "$rc")
		[ "$status" = ok ] || echo "  -> $status, see $err" >&2
		cat "$out" "$err" | to_csv "$prefix" "$status" >> "$csv"
	done; done; done; done; done

	echo "Wrote $csv"
	column -t -s, "$csv"
