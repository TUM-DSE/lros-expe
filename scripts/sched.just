# Scheduling: interactive requests arriving into a stream of background work.
# LROS runs in the guest; llama.cpp and a vLLM/SGLang-like engine run natively
# on the Linux host, on the same pinned cores and device. One recipe per row.

HERE := source_directory()
RESULTS := HERE / "../results"
LIB := HERE / "lib.sh"
PAPER := HERE / "../../lros_paper/plots"
POLICIES := HERE / "../miniosv/app/llama.cpp/examples/lros-serve/policies"
HEADER := "exp,target,platform,arm,fill,rep,req,prio,task,cores_at_start,t_done_s,prompt,gen,secs,tps,waited_s,ttft_s"

# Shared by the recipes: the traces and policy staged for the guest (and, from
# the data image's staging copy, for the host), and one arm run on either side,
# its rows kept only when the run finished and, with $MARKER, used the device.
_common := '''
stage() {   # <dir> <policy> <fills>
  rm -rf "$1"; mkdir -p "$1/traces" "$1/policies"
  cp "$2" "$1/policies/" || exp_die "no policy $2"
  python3 "$EXP_ROOT/scripts/traces.py" "$1/traces" "$3" || exp_die "cannot write the traces"
}

want() { [ -z "$ARMS" ] || echo ",$ARMS," | grep -q ",$1,"; }

guest() {   # <arm> <args>
  want "$1" || return 0
  EXP_TIMEOUT=${RUN_TIMEOUT:-900} exp_guest_retry "$1-$DEV-k$fill" "$rep" "$remote" "$2" "$data" "$model"
  rows "$1"
}

host() {   # <arm> <args>
  want "$1" || return 0
  EXP_HOST_ENV="${TGT_HOST_ACCEL_ENV[$TARGET]}" EXP_TIMEOUT=${RUN_TIMEOUT:-900} \
    exp_host "$1-$DEV-k$fill" "$rep" "'$bin/llama-lros-serve' -m '$model' $2"
  if [ -n "${MARKER:-}" ] && ! grep -qE "$MARKER" "$EXP_LOG"; then
    exp_warn "$1 k$fill rep $rep: no '$MARKER' in $EXP_LOG; rows dropped"; return
  fi
  rows "$1"
}

rows() {   # <arm>
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
	bin=$(exp_build_host cpu llama-lros-serve) || exp_die "cannot build the host baseline"
	nt=$(exp_ncores) mask=$(exp_cpu_mask)
	# The engines run one batch over n slots, their context holding the slots
	# and as many parking sequences; their threads pinned one per core.
	slots="-np $nt -c $(( 768 * (2 * nt + 1) )) -t $nt --mode slots \
	  -C $mask --cpu-strict 1 -Cb $mask --cpu-strict-batch 1"
	lros="lros-serve -m nvme:2 -np $nt -c 4608 -t $nt --mode lros --policy /root/policies/$(basename "$pol")"
	for rep in $(exp_reps); do
	  for fill in $(echo {{fills}} | tr ',' ' '); do
	    htrace="--trace $T_SCRATCH/stage/sched-cpu-$TARGET/traces/cpu-k$fill.trace"
	    EXPECT=$(grep -vc '^#' "$LOCAL_TMP/sched-cpu-$TARGET/traces/cpu-k$fill.trace")
	    host fifo "$slots $htrace --fifo"
	    host priority-chunked "$slots $htrace --chunk {{chunk}}"
	    guest lros "$lros --trace /root/traces/cpu-k$fill.trace"
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
	bin=$(exp_build_host "$T_ACCEL" llama-lros-serve) || exp_die "cannot build the host $T_ACCEL baseline"
	nt=$(exp_ncores) mask=$(exp_cpu_mask)
	# The engines use the device through its native backend.
	case "$T_ACCEL" in
	  cuda) ngl="-ngl 99" MARKER="found [1-9][0-9]* CUDA devices" ;;
	  *)    ngl="" MARKER="using device RKNN" ;;
	esac
	slots="-np $nt -c $(( 2048 * (2 * nt + 1) )) -t $nt --mode slots $ngl \
	  -C $mask --cpu-strict 1 -Cb $mask --cpu-strict-batch 1"
	lros="lros-serve -m nvme:2 -c 8192 -np 8 -t $nt --mode lros -ngl 99 --policy /root/policies/$(basename "$pol")"
	for rep in $(exp_reps); do
	  for fill in $(echo $fills | tr ',' ' '); do
	    htrace="--trace $T_SCRATCH/stage/sched-accel-$TARGET/traces/$DEV-k$fill.trace"
	    EXPECT=$(grep -vc '^#' "$LOCAL_TMP/sched-accel-$TARGET/traces/$DEV-k$fill.trace")
	    host fifo "$slots $htrace --fifo"
	    host priority-chunked "$slots $htrace --chunk {{chunk}}"
	    # LROS's device lease hands the device over at a layer or operator.
	    EXP_VACCEL=1 EXP_GUEST_ENV="VIAI_PREEMPT=1" guest lros "$lros --trace /root/traces/$DEV-k$fill.trace"
	  done
	done
	exp_finish

# An arm in a later CSV replaces the same arm in an earlier one: the *-host
# runs replace the engine arms that were measured in the guest.
[doc('the figure')]
plot out=PAPER:
	#!/usr/bin/env bash
	set -euo pipefail
	csvs() { for e in "$@"; do ls {{RESULTS}}/$e/*/$e.csv 2>/dev/null || true; done; }
	cpu=$(csvs sched-cpu sched-cpu-lros sched-cpu-host)
	accel=$(csvs sched-accel sched-accel-host)
	[ -n "$cpu$accel" ] || { echo "no results yet; run just sched::cpu or sched::accel" >&2; exit 1; }
	cd {{HERE}}/plots
	python3 sched.py {{out}}/sched.pdf --cpu $cpu --accel $accel
