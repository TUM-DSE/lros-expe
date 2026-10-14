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
#   arm     host-fifo       the baselines, on the host: the same binary's slot
#           host-prio       engine on Linux, pinned to the four A76 cores, the
#                           models mmap'd and served by the kernel's page cache.
#                           fifo: one pool of slots per model, arrival order,
#                           the pools stepped in turn, what llama.cpp's
#                           server does; prio: slots with priority, parking
#                           and chunked prefill, and across the models only
#                           the pools holding the best priority present take
#                           a step, what a vLLM/SGLang-like engine with
#                           priorities does. Under the budget the
#                           process runs in a memory cgroup of the budget plus
#                           its own anonymous memory, so the kernel reclaims
#                           the models' pages by its LRU, knowing no roles.
#           lros-default    the task path in the guest with no policy file:
#                           the engine's built-in defaults (every task takes
#                           every free core, priority order, a floor core
#                           past 64 tokens, no in-iteration preemption, no
#                           weight windows -- the models keep the equal
#                           per-file cache the slot arms get)
#           lros-fixed      the task path with a policy: the calibrated width
#                           rules, in-iteration preemption, a chat prefill
#                           exempt from the floor, and under the budget the
#                           weights shared out with the foreground's model
#                           first
#           lros-adaptive   the same policy plus its QoS flags: a foreground
#                           request starts on every core, widens when behind
#                           its rate and gives back when ahead, and background
#                           tasks with deadline slack are held
#           slots-fifo      the slot engines inside the guest (not swept by
#           slots-prio      default): the baselines on LROS's own page cache
#
# Every guest arm streams the same way (2 MiB buffers, 16 read ahead, 90% of
# a window held as a prefix): the default page-at-a-time faulting makes no
# progress at all on a squeezed model. The host arms read the models from
# the same NVMe through the kernel, their page cache dropped before each run.
#
# Output: bench/out/orch/<timestamp>/orch.csv, one row per finished request
# (scripts/parse.py apps), the trace and policies used, and the console logs
# under logs/<mem>/<arm>/run<n>. Figures: just plot_orch <csv>.
#
# With truncate=1 (the default, for quick numbers) the slot arms are stopped
# early: once every background request is done and one more chat turn has
# completed, since their chat crawls under the budget and would run the trace
# out for minutes more. Their rows carry the status `truncated`, and the plots
# take them as they are. The evaluation runs with truncate=0.

# In the order the results are wanted: the LROS arms first, under the budget
# first, so that a partial sweep already compares what matters most.
ORCH_ARMS := "lros-adaptive lros-fixed lros-default host-prio host-fifo"
ORCH_MEMS := "budget unbounded"
# The host arms: the clang build of the same lros-serve (just build linux
# cpu-clang with LLAMA_CPP_TARGETS=llama-lros-serve), its threads pinned one
# per A76 core (mask over the host's cpu numbers), and, under the budget, a
# cgroup of the budget plus what the process holds beside the mapped weights:
# its contexts, samplers and the loaded adapter. That figure is the peak
# RssAnon of an unbounded run (998 MiB on 2026-09-24 with -c 4608 and 4 slots
# per pool, plus some 20 MiB the kernel charges besides); every host run
# records its own in logs/<mem>/<arm>/run<n>.rss.txt for a check.
# The weights are mmap'd, so they sit in the kernel's page cache, which the
# cgroup charges: the kernel then reclaims them by its LRU, with its own
# readahead on the re-faults, nothing pinned. Note that llama.cpp populates
# each mapping in full at load (MAP_POPULATE, WILLNEED), so under the budget
# the first model is partly evicted again while the second one loads.
ORCH_HOST_BIN := LROS_PATH + "/app/llama.cpp/build-cpu-clang/bin/llama-lros-serve"
ORCH_HOST_MASK := "f0"
ORCH_HOST_ANON_MB := "1024"
ORCH_MODELS_DIR := "/scratch/orangepi/models"
# Q8_0 for both: f16 decode is memory-bound past two cores on the A76s (raw
# batched-bench 2026-09-24: Llama f16 160/114/107/109 ms on 1-4 cores against
# Q8 118/69/60/65; Qwen f16 205/149/138/141 against q8 152/92/80/77), so with
# Q8 a core more is a real lever, and the files are half the size.
ORCH_MODEL_A := "Llama-3.2-1B-Instruct-Q8_0.gguf"
# The chat model is the 3B: raw 302/177/156/165 ms per token on 1-4 cores,
# so reading speed (208 ms) is met on two cores alone and missed on one,
# and 12-24 t/s of prefill make the 3 s TTFT bound a constraint too.
ORCH_MODEL_B := "Llama-3.2-3B-Instruct-Q8_0.gguf"
ORCH_LORA := "/scratch/orangepi/orch/llama1b-lora16.gguf"
ORCH_SHAREGPT := "/scratch/orangepi/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
ORCH_DATA_DIR := "/scratch/orangepi/orch"
# The windows: what each model's page cache may hold at boot, and, under the
# budget, what the LROS policy shares out between the two models. 4096 MiB
# holds the 3B chat model whole (3263 MiB) and about 65% of the Q8 base model
# (1260 MiB), which then streams some 0.4 GB per token; the slot engines'
# equal split (2048 MiB each) leaves the chat model streaming instead.
# Unbounded: above both files together (3263 + 1259 MiB); 4G left the chat
# model's tail streaming whenever the 1B model was in use (found 2026-09-25).
# Needs the 10G guest: at 4G of cache the 8G guest had 1.7 GB free with the
# task contexts built.
ORCH_CACHE_UNBOUNDED := "5G"
ORCH_BUDGET_MB := "4096"
ORCH_STREAM_ARGS := "--buffer-size 2M --prefetch-depth 16 --pin-percent 90"
# Every request generates exactly its token count (no early stop), with a
# fixed seed and greedy sampling: the work per turn is then the same in every
# arm and run, whatever the model would have said.
ORCH_SERVE_ARGS := "-t 4 -c 4608 -b 512 -ub 128 -np 8 --fixed-length -s 1 --temp 0"
ORCH_SLOTS := "4"
ORCH_CHUNK := "64"
# Whether a chat turn's prefill leaves the floor core to the background like
# its decode does (true), or takes every core for the second or two it lasts
# (false; the LROS policies' setting). `just ORCH_FLOOR_PREFILL=true ...`
# runs the comparison.
ORCH_FLOOR_PREFILL := "false"
# Extra arguments for the trace generator (scripts/orch_traces.py --help).
ORCH_TRACE_ARGS := ""
ORCH_TIMEOUT := "900"
# The host cores the guest's four vCPUs run on: the four A76 cores, one each.
# (The shared AFFINITY names one core, which makes the vCPUs time-share it.)
ORCH_AFFINITY := "4-7"

# truncate=1 stops a slot arm (host or guest) once its background is done and
# one more chat turn through, for quick numbers; the evaluation runs with 0.
orch_sweep n_runs="1" size_mem="10G" variant="orch" arms=ORCH_ARMS mems=ORCH_MEMS trace_args=ORCH_TRACE_ARGS truncate="1":
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

	need_guest=0; need_host=0
	for arm in {{arms}}; do
		case "$arm" in host-*) need_host=1 ;; *) need_guest=1 ;; esac
	done
	for f in "$model_a" "$model_b" "{{ORCH_LORA}}" "{{ORCH_SHAREGPT}}"; do
		[ -e "$f" ] || { echo "Missing $f" >&2; exit 1; }
	done
	[ "$need_guest" = 0 ] || [ -e "$img" ] || { echo "Missing $img" >&2; exit 1; }
	[ "$need_host" = 0 ] || [ -x "{{ORCH_HOST_BIN}}" ] || { echo "Missing {{ORCH_HOST_BIN}}; see the header of scripts/orch.just" >&2; exit 1; }

	# The trace, the adapter and the policies, staged into one data image.
	python3 {{proot}}/scripts/orch_traces.py "$stage/traces/orch.trace" --sharegpt "{{ORCH_SHAREGPT}}" {{trace_args}}
	cp "{{ORCH_LORA}}" "$stage/lora.gguf"
	policy() {   # <name> <budget_mb or 0> <adaptive>
		local budget="" qos=""
		[ "$2" != 0 ] && budget="\"weights_budget_mb\": $2, \"weights_idle_ms\": 5000, \"weights_floor_pct\": 10, \"weights_prefer\": \"foreground\","
		cat > "$stage/policies/$1.json" <<-EOF
		{
		  "width": {
		    "prefill": [ { "upto_batch": 1048576, "workers": 0 } ],
		    "decode":  [ { "upto_batch": 1, "workers": 2 }, { "upto_batch": 3, "workers": 3 }, { "upto_batch": 1048576, "workers": 0 } ]
		  },
		  "context": { "n_seq_per_task": 8, "n_ctx_per_seq": 512, "warm": 1 },
		  "floor": { "cores": 1, "long_work_tokens": 64, "max_pause_ms": 2000, "foreground_prefill": {{ORCH_FLOOR_PREFILL}} },
		  "preemption": { "in_iteration": true },
		  "memory": { $budget "kv_evict": "swap" },
		  "qos": { "widen": $3, "give_back": $3, "start_wide": $3, "hold_background": $3, "backlog_low": 2, "backlog_high": 8, "step_tokens": 4, "hold_margin": 2.0 },
		  "sessions": { "idle_kv": "keep" }
		}
		EOF
	}
	policy fixed-unbounded 0 false
	policy adaptive-unbounded 0 true
	policy fixed-budget {{ORCH_BUDGET_MB}} false
	policy adaptive-budget {{ORCH_BUDGET_MB}} true
	data="{{ORCH_DATA_DIR}}/data-$ts.img"
	if [ "$need_guest" = 1 ]; then
		mkdir -p "{{ORCH_DATA_DIR}}"
		MKDATA_ASSUME_YES=1 nix develop {{LROS_PATH}} --command bash {{LROS_PATH}}/scripts/mkdata.sh "$data" "$stage" 128M > "$out_dir/mkdata.log" 2>&1
	fi

	{
		echo "timestamp $ts"
		echo "image $img"
		echo "models $model_a $model_b lora {{ORCH_LORA}}"
		echo "guest memory {{size_mem}}, cpus {{ORCH_AFFINITY}}, timeout {{ORCH_TIMEOUT}}s"
		echo "cache unbounded {{ORCH_CACHE_UNBOUNDED}}, budget {{ORCH_BUDGET_MB}} MiB, stream {{ORCH_STREAM_ARGS}}"
		echo "serve {{ORCH_SERVE_ARGS}}, slots {{ORCH_SLOTS}}, chunk {{ORCH_CHUNK}}, floor foreground_prefill {{ORCH_FLOOR_PREFILL}}"
		echo "host {{ORCH_HOST_BIN}}, mask {{ORCH_HOST_MASK}}, anon {{ORCH_HOST_ANON_MB}} MiB"
		echo "arms {{arms}}"; echo "mems {{mems}}"; echo "trace args {{trace_args}}"
		echo "git $(git -C {{proot}} rev-parse --short HEAD) miniosv $(git -C {{LROS_PATH}} rev-parse --short HEAD 2>/dev/null || echo '?') app $(git -C {{LROS_PATH}}/app/llama.cpp rev-parse --short HEAD 2>/dev/null || echo '?')"
	} > "$out_dir/sweep.info"

	# The models the trace's model column indexes: the base, the base with
	# the adapter, the chat model. nvme:1 is the data image.
	models="nvme:2,nvme:2+/root/lora.gguf,nvme:3"
	trace="--trace /root/traces/orch.trace"
	expect=$(grep -vcE '^#|^@|^$' "$stage/traces/orch.trace")
	expect_bg=$(grep -cE '^[+0-9]+ (summarise|reply) ' "$stage/traces/orch.trace")

	status_of() { case "$1" in 0) echo ok ;; 124|137) echo timeout ;; *) echo "failed($1)" ;; esac; }

	echo "timestamp,run,mem,arm,$(python3 -c 'import sys; sys.path.insert(0, "{{proot}}/scripts"); import parse; print(parse.APPS_HEADER)'),status" > "$csv"
	# The cores in use over each run (scripts/parse.py width), for the
	# utilization figure of the plots.
	width_csv="$out_dir/width.csv"
	echo "timestamp,run,mem,arm,$(python3 -c 'import sys; sys.path.insert(0, "{{proot}}/scripts"); import parse; print(parse.WIDTH_HEADER)')" > "$width_csv"

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
		# The host arms: the same slot engine natively, the trace read from
		# the staging directory, the models and the adapter from their files.
		# The mask is over the host's cpu numbers.
		hpin="-C {{ORCH_HOST_MASK}} --cpu-strict 1 -Cb {{ORCH_HOST_MASK}} --cpu-strict-batch 1"
		hmodels="$model_a,$model_a+{{ORCH_LORA}},$model_b"
		htrace="--trace $stage/traces/orch.trace"
		case "$arm" in
			host-fifo)     serve="--models $hmodels $htrace --mode slots --fifo -np {{ORCH_SLOTS}} $hpin -m $model_a" ;;
			host-prio)     serve="--models $hmodels $htrace --mode slots --chunk {{ORCH_CHUNK}} -np {{ORCH_SLOTS}} $hpin -m $model_a" ;;
			slots-fifo)    serve="lros-serve -m nvme:2 --models $models $trace --mode slots --fifo -np {{ORCH_SLOTS}} $pin" ;;
			slots-prio)    serve="lros-serve -m nvme:2 --models $models $trace --mode slots --chunk {{ORCH_CHUNK}} -np {{ORCH_SLOTS}} $pin" ;;
			lros-default)  serve="lros-serve -m nvme:2 --models $models $trace --mode lros" ;;
			lros-fixed)    serve="lros-serve -m nvme:2 --models $models $trace --mode lros --policy /root/policies/fixed-$suffix.json" ;;
			lros-adaptive) serve="lros-serve -m nvme:2 --models $models $trace --mode lros --policy /root/policies/adaptive-$suffix.json" ;;
			*) echo "unknown arm $arm" >&2; exit 1 ;;
		esac
		# The slot engines take -np as slots per pool; -t, -c and the batch
		# sizes are shared. -np of the lros arms is the sequence count of
		# the loading context alone.
		args="$serve {{ORCH_SERVE_ARGS}}"
		case "$arm" in slots-*|host-*) args="${args/-np 8/}" ;; esac

		log_dir="$out_dir/logs/$mem/$arm"; mkdir -p "$log_dir"
		out="$log_dir/run$run.out.txt"; err="$log_dir/run$run.err.txt"
		echo "=== run $run/{{n_runs}}: $mem, $arm ==="
		vmtouch -qe "$model_a" "$model_b" "{{ORCH_LORA}}" 2>/dev/null || true
		rc=0
		case "$arm" in
			host-*)
				# Under the budget, a cgroup of the budget plus the process's
				# own memory: the kernel then reclaims the models' pages.
				limit=()
				[ "$mem" = budget ] && limit=(-p "MemoryMax=$(( {{ORCH_BUDGET_MB}} + {{ORCH_HOST_ANON_MB}} ))M")
				# Everything lros-serve logs goes to stderr; one file, like
				# the guest's console.
				timeout -k 10 {{ORCH_TIMEOUT}} \
					systemd-run --user --scope -q "${limit[@]}" \
					taskset -c {{ORCH_AFFINITY}} {{ORCH_HOST_BIN}} $args \
					< /dev/null > "$out" 2>&1 &
				;;
			*)
				timeout -k 10 {{ORCH_TIMEOUT}} \
					taskset -c {{ORCH_AFFINITY}} python3 {{LROS_PATH}}/scripts/run.py \
					--arch aarch64 -p kvm -c 4 -m {{size_mem}} --nogdb \
					--image-path "$img" --emulated-nvme "$data" --emulated-nvme "$model_a" --emulated-nvme "$model_b" \
					--nvme-cache none --nvme-aio io_uring --nvme-mdts 9 \
					--args "--cache-size $cache {{ORCH_STREAM_ARGS}} $args" \
					< /dev/null > "$out" 2> "$err" &
				;;
		esac
		guest=$!
		truncated=0
		# A host run's own memory beside the mapped weights, at its peak: what
		# the budget's cgroup has to allow on top of the weights.
		rss_peak=0
		rss_of() {   # kB of anonymous memory of the serving process, 0 if gone
			local pid; pid=$(pgrep -n -f "^{{ORCH_HOST_BIN}} " 2>/dev/null || true)
			[ -n "$pid" ] && awk '/^RssAnon/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0
		}
		case "$arm" in slots-*|host-*|lros-default)
			# The slot engines' chat crawls under the budget and would run
			# the trace out for minutes after the background is done. Once
			# every summarise and reply request has finished, one more chat
			# turn is let through and the guest is stopped.
			chat_at_bg_done=-1
			while kill -0 "$guest" 2>/dev/null; do
				sleep 2
				case "$arm" in host-*) r=$(rss_of); [ "${r:-0}" -gt "$rss_peak" ] && rss_peak=$r ;; esac
				bg_done=$(grep -ac "DONE.*app \(summarise\|reply\) " "$out" 2>/dev/null || true)
				chat_done=$(grep -ac "DONE.*app chat " "$out" 2>/dev/null || true)
				if [ "$bg_done" -ge "$expect_bg" ] && [ "$chat_at_bg_done" -lt 0 ]; then
					chat_at_bg_done=$chat_done
				fi
				if [ "$chat_at_bg_done" -ge 0 ] && [ "$chat_done" -gt "$chat_at_bg_done" ] && [ "{{truncate}}" = 1 ]; then
					echo "  background done and one more chat turn: stopping the run"
					# run.py takes SIGTERM and kills its QEMU; the pkills are the fallback.
					kill "$guest" 2>/dev/null || true
					sleep 3
					pkill -KILL -f "qemu-system-aarch64.*orch" 2>/dev/null || true
					pkill -KILL -f "^{{ORCH_HOST_BIN}} " 2>/dev/null || true
					truncated=1
					break
				fi
			done ;;
		esac
		wait "$guest" || rc=$?
		[ "$truncated" = 1 ] && rc=0
		rm -rf /tmp/nix-shell.*/miniosv-run-* 2>/dev/null || true
		status=$(status_of "$rc")
		[ "$truncated" = 1 ] && status=truncated
		case "$arm" in host-*) echo "peak RssAnon $(( rss_peak / 1024 )) MiB" > "$log_dir/run$run.rss.txt"; echo "  peak RssAnon $(( rss_peak / 1024 )) MiB" ;; esac
		n=0
		while IFS= read -r row; do
			[ -n "$row" ] && { echo "$ts,$run,$mem,$arm,$row,$status" >> "$csv"; n=$((n + 1)); }
		done < <(python3 {{proot}}/scripts/parse.py apps "$out")
		python3 {{proot}}/scripts/parse.py width "$out" | sed "s/^/$ts,$run,$mem,$arm,/" >> "$width_csv"
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
