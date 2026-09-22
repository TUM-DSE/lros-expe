# Motivation: inference on a shared device under Linux, with llama.cpp on the
# host; one recipe per panel of the figure.

HERE := source_directory()
RESULTS := HERE / "../results"
LIB := HERE / "lib.sh"
PAPER := HERE / "../../lros_paper/plots"
MODEL := "Llama-3.2-1B-Instruct-f16.gguf"

[doc('(a) Linux reclaiming weights or swapping the KV cache under a quota')]
kvswap target="joy" quota="3399":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_sync_models
	# The arms need swap, so the lock is taken without exp_begin turning it off.
	exp_lock
	bin=$(exp_build_host cpu llama-batched-bench) || exp_die "no host build"
	tgt_push "$EXP_ROOT/scripts/kvswap.sh" bin/kvswap.sh
	mkdir -p "$RESULTS_DIR/mot-kvswap/$TARGET"
	tgt_sh "bash \$SCRATCH/bin/kvswap.sh '$bin/llama-batched-bench' '$(exp_model {{MODEL}})' {{quota}}" \
	  | tee "$RESULTS_DIR/mot-kvswap/$TARGET/kvswap.log"

[doc('(b) decode batching: aggregate against per-request rate as the batch grows')]
batching target="joy" slots="1,2,3,4,5,6,7,8,9,10,11,12" npp="64" ntg="128" ub="64":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_sync_models
	exp_begin mot-batching "exp,target,platform,slots,npp,ntg,rep,t_pp,s_pp,t_tg,s_tg"
	model=$(exp_model {{MODEL}}) nt=$(exp_ncores)
	export EXP_NO_DROP=1 EXP_TIMEOUT=1800
	bin=$(exp_build_host cpu llama-batched-bench) || exp_die "no host build"
	exp_host warmup 1 "'$bin/llama-batched-bench' -m '$model' -npp 128 -ntg 4 -npl 1 -t $nt --no-perf"
	for rep in $(exp_reps); do
	  for pl in $(echo {{slots}} | tr ',' ' '); do
	    exp_host "slots$pl" "$rep" "'$bin/llama-batched-bench' -m '$model' -npp {{npp}} -ntg {{ntg}} \
	      -npl $pl -c $((pl * ({{npp}} + {{ntg}}) + 256)) -b 2048 -ub {{ub}} -t $nt --no-perf"
	    while IFS=, read -r _ _ _ _ tpp spp ttg stg _; do
	      exp_row mot-batching "$TARGET" "$T_PLATFORM" "$pl" {{npp}} {{ntg}} "$rep" "$tpp" "$spp" "$ttg" "$stg"
	    done < <(exp_parse bench "$EXP_LOG")
	  done
	done
	exp_finish

[doc('(c) an interactive request arriving into seven decoding background requests')]
arrival target="joy" nbg="7" deadline="208" intprompt="512" inttok="256" trials="3":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_sync_models
	exp_begin mot-arrival \
	  "exp,target,platform,nbg,arm,trial,ttft_ms,tpot_mean_ms,tpot_p95_ms,late_frac,int_tokens,bg_before,bg_during"
	model=$(exp_model {{MODEL}}) nt=$(exp_ncores)
	export EXP_NO_DROP=1 EXP_TIMEOUT=1800
	srv=$(exp_build_host cpu llama-server) || exp_die "no llama-server"
	tgt_push "$EXP_ROOT/scripts/timelineprobe.py" bin/timelineprobe.py
	# alone: nothing else running; base: in the batch; deadline: prioritized,
	# the background kept to what the reading-speed deadline leaves.
	for arm in alone base deadline; do
	  nb={{nbg}} env=""
	  case $arm in
	    alone)    nb=0 ;;
	    deadline) env="export LLAMA_PRIORITY_SLOT=0 LLAMA_TPOT_TARGET_MS={{deadline}}" ;;
	  esac
	  pid=$(tgt_sh "$env
	    nohup taskset -c $T_CORES '$srv/llama-server' -m '$model' --host 127.0.0.1 --port 8195 \
	      -np $((nb + 1)) -c $(((nb + 1) * 2048)) -b 2048 -ub 512 -t $nt \
	      > \$SCRATCH/run/arrival-$arm.log 2>&1 & echo \$!")
	  tgt_sh "for i in \$(seq 90); do curl -sf -m 2 http://127.0.0.1:8195/health >/dev/null && exit 0; sleep 1; done; exit 1" \
	    || { exp_warn "server for $arm never became ready"; tgt_sh "kill -9 $pid"; continue; }
	  log="$EXP_RAW/$arm.log"
	  tgt_sh "timeout $EXP_TIMEOUT python3 \$SCRATCH/bin/timelineprobe.py --port 8195 --arm $arm \
	    --n-bg $nb --int-prompt {{intprompt}} --int-tokens {{inttok}} --deadline-ms {{deadline}} \
	    --trials {{trials}}" > "$log" 2> "$log.err"
	  while IFS=, read -r a trial rest; do
	    [ "$a" = "$arm" ] && exp_row mot-arrival "$TARGET" "$T_PLATFORM" "$nb" "$a" "$trial" "$rest"
	  done < "$log"
	  tgt_sh "kill -9 $pid 2>/dev/null; true"
	done
	exp_finish

[doc('the three motivation figures')]
plot target="joy" out=PAPER:
	#!/usr/bin/env bash
	set -euo pipefail
	cd {{HERE}}/plots
	python3 mot-kvswap.py {{RESULTS}}/mot-kvswap/{{target}}/kvswap.log {{out}}/mot-kvswap.pdf
	python3 mot-batching.py {{RESULTS}}/mot-batching/{{target}}/mot-batching.csv {{out}}/mot-batching.pdf
	python3 mot-arrival.py {{RESULTS}}/mot-arrival/{{target}}/mot-arrival.csv {{out}}/mot-arrival.pdf
