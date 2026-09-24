# Orchestration: three applications with declared roles and QoS on two
# models, one machine. A chat (foreground, the chat model) beside a stream of
# summarisations (background, the base model) and bursts of smart replies
# (background, the base model with a LoRA adapter). The trace has phases:
# background only, the chat's turns, a long pause of the chat, a burst during
# a turn. See scripts/orch_traces.py for the scenario and its knobs.
#
# Manual form (the guest), with the trace, the policies and the adapter in an
# ext4 data image mounted at /root (miniosv/scripts/mkdata.sh):
#   python3 miniosv/scripts/run.py --arch aarch64 -p kvm -c 4 -m 8G --nogdb \
#     --image-path miniosv/build/orch.aarch64/loader.img \
#     --emulated-nvme data.img --emulated-nvme <A.gguf> --emulated-nvme <B.gguf> \
#     --nvme-cache none \
#     --args "--cache-size 2G --buffer-size 2M --prefetch-depth 16 --pin-percent 90 \
#             lros-serve -m nvme:2 --models nvme:2,nvme:2+/root/lora.gguf,nvme:3 \
#             --trace /root/traces/orch.trace --mode lros --policy /root/policies/x.json \
#             -t 4 -c 4608 -b 512 -ub 128 -np 8"
#
# Dimensions (all swept, innermost last):
#   run     repetition of the whole boot / process
#   mem     unbounded   every model's window holds its whole file
#           budget      the windows share ORCH_BUDGET_MB: less than both files,
#                       so the background model streams from disk while the
#                       chat model is whole (the LROS policy), or every window
#                       gets an equal share (the slot engines, which know no
#                       roles)
#   arm     slots-fifo      one pool of slots per model, arrival order: llama.cpp
#           slots-prio      slots with priority, parking and chunked prefill:
#                           a vLLM/SGLang-like engine
#           lros-fixed      the task path with the calibrated width rules,
#                           priorities and in-iteration preemption
#           lros-adaptive   the same, plus the QoS controller: a foreground
#                           app behind its TPOT target widens its task, and
#                           background tasks with deadline slack are held
#
# Every arm streams the same way (2 MiB buffers, 16 read ahead, 90% of a
# window held as a prefix): the default page-at-a-time faulting makes no
# progress at all on a squeezed model.
#
# Output: bench/out/orch/<timestamp>/orch.csv, one row per finished request
# (scripts/parse.py apps), the trace and policies used, and the console logs
# under logs/<mem>/<arm>/run<n>. Figures: just plot_orch <csv>.

# In the order the results are wanted: the LROS arms first, under the budget
# first, so that a partial sweep already compares what matters most.
ORCH_ARMS := "lros-adaptive lros-fixed slots-prio slots-fifo"
ORCH_MEMS := "budget unbounded"
ORCH_MODELS_DIR := "/scratch/orangepi/models"
ORCH_MODEL_A := "Llama-3.2-1B-Instruct-f16.gguf"
ORCH_MODEL_B := "qwen2.5-1.5b-instruct-fp16.gguf"
ORCH_LORA := "/scratch/orangepi/orch/llama1b-lora16.gguf"
ORCH_SHAREGPT := "/scratch/orangepi/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
ORCH_DATA_DIR := "/scratch/orangepi/orch"
# The windows: what each model's page cache may hold at boot, and, under the
# budget, what the LROS policy shares out between the two models. 5120 MiB is
# what the 8 GiB guest leaves for weights beside three 290 MiB contexts; it
# holds the chat model whole (3395 MiB) and 1724 MiB of the base model.
ORCH_CACHE_UNBOUNDED := "4G"
ORCH_BUDGET_MB := "5120"
ORCH_STREAM_ARGS := "--buffer-size 2M --prefetch-depth 16 --pin-percent 90"
ORCH_SERVE_ARGS := "-t 4 -c 4608 -b 512 -ub 128 -np 8"
ORCH_SLOTS := "4"
ORCH_CHUNK := "64"
# Extra arguments for the trace generator (scripts/orch_traces.py --help).
ORCH_TRACE_ARGS := ""
ORCH_TIMEOUT := "900"
# The host cores the guest's four vCPUs run on: the four A76 cores, one each.
# (The shared AFFINITY names one core, which makes the vCPUs time-share it.)
ORCH_AFFINITY := "4-7"

orch_sweep n_runs="1" size_mem="8G" variant="orch" arms=ORCH_ARMS mems=ORCH_MEMS trace_args=ORCH_TRACE_ARGS:
	#!/usr/bin/env bash
	set -euo pipefail

	ts="$(date -Iseconds)"
	out_dir="{{OUT_DIR}}/orch/$ts"
	stage="$out_dir/stage"
	mkdir -p "$stage/traces" "$stage/policies"
	csv="$out_dir/orch.csv"
	img="{{LROS_PATH}}/build/{{variant}}.aarch64/{{LROS_IMAGE_NAME}}"
	model_a="{{ORCH_MODELS_DIR}}/{{ORCH_MODEL_A}}"
	model_b="{{ORCH_MODELS_DIR}}/{{ORCH_MODEL_B}}"

	for f in "$img" "$model_a" "$model_b" "{{ORCH_LORA}}" "{{ORCH_SHAREGPT}}"; do
		[ -e "$f" ] || { echo "Missing $f" >&2; exit 1; }
	done

	# The trace, the adapter and the policies, staged into one data image.
	python3 {{proot}}/scripts/orch_traces.py "$stage/traces/orch.trace" --sharegpt "{{ORCH_SHAREGPT}}" {{trace_args}}
	cp "{{ORCH_LORA}}" "$stage/lora.gguf"
	policy() {   # <name> <budget_mb or 0> <adaptive>
		local budget="" qos=""
		[ "$2" != 0 ] && budget="\"weights_budget_mb\": $2, \"weights_idle_ms\": 5000, \"weights_floor_pct\": 10,"
		cat > "$stage/policies/$1.json" <<-EOF
		{
		  "width": {
		    "prefill": [ { "upto_batch": 1048576, "workers": 0 } ],
		    "decode":  [ { "upto_batch": 1, "workers": 2 }, { "upto_batch": 3, "workers": 3 }, { "upto_batch": 1048576, "workers": 0 } ]
		  },
		  "context": { "n_seq_per_task": 8, "n_ctx_per_seq": 512, "warm": 1 },
		  "preemption": { "in_iteration": true },
		  "memory": { $budget "kv_evict": "swap" },
		  "qos": { "adaptive": $3, "backlog_low": 2.0, "backlog_high": 8.0, "step_tokens": 4, "hold_background": true, "hold_margin": 2.0 },
		  "sessions": { "idle_kv": "keep" }
		}
		EOF
	}
	policy fixed-unbounded 0 false
	policy adaptive-unbounded 0 true
	policy fixed-budget {{ORCH_BUDGET_MB}} false
	policy adaptive-budget {{ORCH_BUDGET_MB}} true
	data="{{ORCH_DATA_DIR}}/data-$ts.img"
	mkdir -p "{{ORCH_DATA_DIR}}"
	MKDATA_ASSUME_YES=1 nix develop {{LROS_PATH}} --command bash {{LROS_PATH}}/scripts/mkdata.sh "$data" "$stage" 128M > "$out_dir/mkdata.log" 2>&1

	{
		echo "timestamp $ts"
		echo "image $img"
		echo "models $model_a $model_b lora {{ORCH_LORA}}"
		echo "guest memory {{size_mem}}, cpus {{ORCH_AFFINITY}}, timeout {{ORCH_TIMEOUT}}s"
		echo "cache unbounded {{ORCH_CACHE_UNBOUNDED}}, budget {{ORCH_BUDGET_MB}} MiB, stream {{ORCH_STREAM_ARGS}}"
		echo "serve {{ORCH_SERVE_ARGS}}, slots {{ORCH_SLOTS}}, chunk {{ORCH_CHUNK}}"
		echo "arms {{arms}}"; echo "mems {{mems}}"; echo "trace args {{trace_args}}"
		echo "git $(git -C {{proot}} rev-parse --short HEAD) miniosv $(git -C {{LROS_PATH}} rev-parse --short HEAD) app $(git -C {{LROS_PATH}}/app/llama.cpp rev-parse --short HEAD)"
	} > "$out_dir/sweep.info"

	# The models the trace's model column indexes: the base, the base with
	# the adapter, the chat model. nvme:1 is the data image.
	models="nvme:2,nvme:2+/root/lora.gguf,nvme:3"
	trace="--trace /root/traces/orch.trace"
	expect=$(grep -vcE '^#|^@|^$' "$stage/traces/orch.trace")

	status_of() { case "$1" in 0) echo ok ;; 124|137) echo timeout ;; *) echo "failed($1)" ;; esac; }

	echo "timestamp,run,mem,arm,$(python3 -c 'import sys; sys.path.insert(0, "{{proot}}/scripts"); import parse; print(parse.APPS_HEADER)'),status" > "$csv"

	# mem outside arm: a partial sweep then holds every arm of one memory
	# setting, which is a complete comparison.
	for run in $(seq 1 {{n_runs}}); do
	for mem in {{mems}}; do
		case "$mem" in
			unbounded) cache="{{ORCH_CACHE_UNBOUNDED}}"; suffix=unbounded ;;
			budget)    cache="$(( {{ORCH_BUDGET_MB}} / 2 ))M"; suffix=budget ;;
			*) echo "unknown mem $mem" >&2; exit 1 ;;
		esac
	for arm in {{arms}}; do
		# The slot engines compute on ggml's own threadpool, which the guest
		# leaves on one core unless it is pinned: one thread per core, strict.
		pin="-C f --cpu-strict 1 -Cb f --cpu-strict-batch 1"
		case "$arm" in
			slots-fifo)    serve="lros-serve -m nvme:2 --models $models $trace --mode slots --fifo -np {{ORCH_SLOTS}} $pin" ;;
			slots-prio)    serve="lros-serve -m nvme:2 --models $models $trace --mode slots --chunk {{ORCH_CHUNK}} -np {{ORCH_SLOTS}} $pin" ;;
			lros-fixed)    serve="lros-serve -m nvme:2 --models $models $trace --mode lros --policy /root/policies/fixed-$suffix.json" ;;
			lros-adaptive) serve="lros-serve -m nvme:2 --models $models $trace --mode lros --policy /root/policies/adaptive-$suffix.json" ;;
			*) echo "unknown arm $arm" >&2; exit 1 ;;
		esac
		# The slot engines take -np as slots per pool; -t, -c and the batch
		# sizes are shared. -np of the lros arms is the sequence count of
		# the loading context alone.
		args="$serve {{ORCH_SERVE_ARGS}}"
		case "$arm" in slots-*) args="${args/-np 8/}" ;; esac

		log_dir="$out_dir/logs/$mem/$arm"; mkdir -p "$log_dir"
		out="$log_dir/run$run.out.txt"; err="$log_dir/run$run.err.txt"
		echo "=== run $run/{{n_runs}}: $mem, $arm ==="
		vmtouch -qe "$model_a" "$model_b" 2>/dev/null || true
		rc=0
		timeout -k 10 {{ORCH_TIMEOUT}} \
			taskset -c {{ORCH_AFFINITY}} python3 {{LROS_PATH}}/scripts/run.py \
			--arch aarch64 -p kvm -c 4 -m {{size_mem}} --nogdb \
			--image-path "$img" --emulated-nvme "$data" --emulated-nvme "$model_a" --emulated-nvme "$model_b" \
			--nvme-cache none \
			--args "--cache-size $cache {{ORCH_STREAM_ARGS}} $args" \
			< /dev/null > "$out" 2> "$err" || rc=$?
		rm -rf /tmp/nix-shell.*/miniosv-run-* 2>/dev/null || true
		status=$(status_of "$rc")
		n=0
		while IFS= read -r row; do
			[ -n "$row" ] && { echo "$ts,$run,$mem,$arm,$row,$status" >> "$csv"; n=$((n + 1)); }
		done < <(python3 {{proot}}/scripts/parse.py apps "$out")
		echo "  -> $status, $n of $expect requests"
	done; done; done

	echo "Wrote $csv"

plot_orch csv="" out_dir="" args="":
	#!/usr/bin/env bash
	set -euo pipefail
	csvs=()
	for c in {{csv}}; do csvs+=("$(cd "{{invocation_directory()}}" && realpath "$c")"); done
	out=(); [ -n "{{out_dir}}" ] && out=(-o "$(cd "{{invocation_directory()}}" && realpath -m "{{out_dir}}")")
	cd {{source_directory()}}/plots
	python3 orch.py "${csvs[@]}" "${out[@]}" {{args}}
