# Scheduling: interactive requests arriving into a stream of background work,
# on llama.cpp, a vLLM/SGLang-like engine and LROS; one recipe per row.

HERE := source_directory()
RESULTS := HERE / "../results"
LIB := HERE / "lib.sh"
PAPER := HERE / "../../lros_paper/plots"
POLICIES := HERE / "../miniosv/app/llama.cpp/examples/lros-serve/policies"
HEADER := "exp,target,platform,arm,fill,rep,req,prio,task,cores_at_start,t_done_s,prompt,gen,secs,tps,waited_s,ttft_s"

# Shared by the recipes: the traces and policy staged for the guest, and one
# arm run, its rows kept only when the run finished every request.
_common := '''
stage() {   # <dir> <policy> <fills>
  rm -rf "$1"; mkdir -p "$1/traces" "$1/policies"
  cp "$2" "$1/policies/" || exp_die "no policy $2"
  python3 "$EXP_ROOT/scripts/traces.py" "$1/traces" "$3" || exp_die "cannot write the traces"
}

run() {   # <arm> <args>
  [ -z "$ARMS" ] || echo ",$ARMS," | grep -q ",$1," || return 0
  EXP_TIMEOUT=${GUEST_TIMEOUT:-900} exp_guest_retry "$1-$DEV-k$fill" "$rep" "$remote" "$2" "$data" "$model"
  if ! exp_parse serve "$EXP_LOG" | grep -q '^ok=1'; then
    exp_warn "$1 k$fill rep $rep: the run did not finish; rows dropped"; return
  fi
  local n=0 r
  while IFS= read -r r; do
    [ -n "$r" ] && { exp_row "$EXP_NAME" "$TARGET" "$T_PLATFORM" "$1" "$fill" "$rep" "$r"; n=$((n + 1)); }
  done < <(exp_parse requests "$EXP_LOG")
  exp_say "$1 k$fill rep $rep: $n request(s)"
  [ "$n" -lt "$EXPECT" ] && exp_warn "$1 k$fill rep $rep: only $n of $EXPECT requests finished"
}
'''

[doc('the CPU row')]
cpu target="joy" model="Llama-3.2-1B-Instruct-f16.gguf" fills="2,4,8" chunk="64" arms="" name="sched-cpu":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	ARMS="{{arms}}" DEV=cpu
	{{_common}}
	exp_setup
	exp_sync_models
	exp_begin "{{name}}" "{{HEADER}}"
	model=$(exp_model "{{model}}")
	# The engine's best policy (the best priority takes every core, prompts in
	# chunks), which LROS applies preempting inside an iteration.
	pol={{POLICIES}}/orin-cpu-llama32-1b-f16-exclusive.json
	stage "$LOCAL_TMP/sched-cpu-$TARGET" "$pol" "{{fills}}"
	data=$(exp_mkdata "$LOCAL_TMP/sched-cpu-$TARGET" "sched-cpu-$TARGET" 256M)
	remote=$(exp_ship_image cpu "$(exp_build_guest cpu 0 1)")
	nt=$(exp_ncores); [ "$T_VCPUS" -lt "$nt" ] && nt=$T_VCPUS
	# The engines run one batch over n slots, their context holding the slots
	# and as many parking sequences.
	mask=$(printf '%x' $(( (1 << nt) - 1 )))
	slots="lros-serve -m nvme:2 -np $nt -c $(( 768 * (2 * nt + 1) )) -t $nt --mode slots \
	  -C $mask --cpu-strict 1 -Cb $mask --cpu-strict-batch 1"
	lros="lros-serve -m nvme:2 -np $nt -c 4608 -t $nt --mode lros --policy /root/policies/$(basename "$pol")"
	for rep in $(exp_reps); do
	  for fill in $(echo {{fills}} | tr ',' ' '); do
	    trace="--trace /root/traces/cpu-k$fill.trace"
	    EXPECT=$(grep -vc '^#' "$LOCAL_TMP/sched-cpu-$TARGET/traces/cpu-k$fill.trace")
	    run fifo "$slots $trace --fifo"
	    run priority-chunked "$slots $trace --chunk {{chunk}}"
	    run lros "$lros $trace"
	  done
	done
	exp_finish

[doc('the GPU row (joy) or the NPU row (orangepi)')]
accel target="joy" model="Llama-3.2-1B-Instruct-f16.gguf" fills="" chunk="64" arms="" name="sched-accel":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	ARMS="{{arms}}"
	{{_common}}
	exp_setup
	exp_sync_models
	exp_has_accel || exp_die "no accelerator stack on $TARGET; run: just boards::accel-setup $TARGET"
	tgt_sh "${TGT_VACCEL_ENV[$TARGET]}
	        grep -qa VIAI_PREEMPT \"\$VACCEL_PLUGINS\"" \
	  || exp_die "the plugin on $TARGET has no device lease; run: just boards::accel-plugin $TARGET"
	exp_begin "{{name}}" "{{HEADER}}"
	model=$(exp_model "{{model}}")
	case "$T_PLATFORM" in
	  orin)   pol={{POLICIES}}/orin-gpu-llama32-1b-f16-preempt.json DEV=gpu fills=8,16,32 ;;
	  rk3588) pol={{POLICIES}}/rk3588-npu-llama32-1b-f16-preempt.json DEV=npu fills=2,4,8 ;;
	  *)      exp_die "no accelerator policy for $T_PLATFORM" ;;
	esac
	[ -n "{{fills}}" ] && fills="{{fills}}"
	stage "$LOCAL_TMP/sched-accel-$TARGET" "$pol" "$fills"
	data=$(exp_mkdata "$LOCAL_TMP/sched-accel-$TARGET" "sched-accel-$TARGET" 64M)
	remote=$(exp_ship_image viai "$(exp_build_guest viai)")
	nt=$(exp_ncores); [ "$T_VCPUS" -lt "$nt" ] && nt=$T_VCPUS
	export EXP_VACCEL=1
	mask=$(printf '%x' $(( (1 << nt) - 1 )))
	slots="lros-serve -m nvme:2 -np $nt -c $(( 2048 * (2 * nt + 1) )) -t $nt --mode slots -ngl 99 \
	  -C $mask --cpu-strict 1 -Cb $mask --cpu-strict-batch 1"
	lros="lros-serve -m nvme:2 -c 8192 -np 8 -t $nt --mode lros -ngl 99 --policy /root/policies/$(basename "$pol")"
	for rep in $(exp_reps); do
	  for fill in $(echo $fills | tr ',' ' '); do
	    trace="--trace /root/traces/$DEV-k$fill.trace"
	    EXPECT=$(grep -vc '^#' "$LOCAL_TMP/sched-accel-$TARGET/traces/$DEV-k$fill.trace")
	    # An engine submits one batch at a time, so the device lease has nothing
	    # to reorder; LROS's lease hands the device over at a layer or operator.
	    EXP_GUEST_ENV="VIAI_PREEMPT=0" run fifo "$slots $trace --fifo"
	    EXP_GUEST_ENV="VIAI_PREEMPT=0" run priority-chunked "$slots $trace --chunk {{chunk}}"
	    EXP_GUEST_ENV="VIAI_PREEMPT=1" run lros "$lros $trace"
	  done
	done
	exp_finish

[doc('the figure; sched-cpu-lros replaces the LROS arm of sched-cpu')]
plot out=PAPER:
	#!/usr/bin/env bash
	set -euo pipefail
	cpu=$(ls {{RESULTS}}/sched-cpu/*/sched-cpu.csv {{RESULTS}}/sched-cpu-lros/*/sched-cpu-lros.csv 2>/dev/null || true)
	accel=$(ls {{RESULTS}}/sched-accel/*/sched-accel.csv 2>/dev/null || true)
	[ -n "$cpu$accel" ] || { echo "no results yet; run just sched::cpu or sched::accel" >&2; exit 1; }
	cd {{HERE}}/plots
	python3 sched.py {{out}}/sched.pdf --cpu $cpu --accel $accel
