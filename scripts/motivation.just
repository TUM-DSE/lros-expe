# Motivation: inference on a shared device, llama.cpp on Linux against LROS;
# one recipe per panel of the figure, plus the paging sweeps behind (a).

HERE := source_directory()
RESULTS := HERE / "../results"
LIB := HERE / "lib.sh"
PAPER := HERE / "../../lros_paper/plots"
# A dense model and a mixture of experts.
DENSE := "Llama-3.2-1B-Instruct-f16.gguf"
MOE := "olmoe-1b-7b-0924-instruct-q4_k_m.gguf"
PAGING_MODELS := DENSE + "," + MOE

[doc('(b) the next token after a KV cache left memory: the state file of llama.cpp on Linux against LROS')]
kvswap target="joy" model="Llama-3.2-1B-Instruct-f16.gguf" lengths="512,1024,2048,4096,8192" keeps="0,90" reps="3" page="64K" recompute="1":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_begin mot-kvswap "exp,target,platform,arm,keep,n,rep,kv_mb,t_pp,t_dec,t_out,mb_out,t_in,mb_in,t_dec2"
	export EXP_NO_DROP=1 EXP_TIMEOUT=${EXP_TIMEOUT:-3600} EXP_NVME_CACHE=none
	model=$(exp_model {{model}}) nt=$(exp_ncores)
	bench="-fa -t $nt --lengths {{lengths}} --reps {{reps}}"
	cpu=$(exp_build_host cpu-clang llama-kv-bench) || exp_die "no host build"
	if [ {{recompute}} = 1 ]; then gpu=$(exp_build_host "$T_ACCEL" llama-kv-bench) || exp_die "no host $T_ACCEL build"; fi
	remote=$(exp_ship_image cpu "$(exp_build_guest cpu 0 1)")
	area="$T_SCRATCH/kvswap/area.img"
	tgt_sh "mkdir -p '$T_SCRATCH/kvswap'; [ -s '$area' ] || fallocate -l 1G '$area'"
	rows() {   # <arm> <keep>
	  local n=0 r
	  while IFS= read -r r; do
	    exp_row mot-kvswap "$TARGET" "$T_PLATFORM" "$1" "$2" "$r"; n=$((n + 1))
	  done < <(awk -F'|' '/^\| *[0-9]/ {o = ""; for (i = 2; i < NF; i++) {gsub(/ /, "", $i); o = o (i > 2 ? "," : "") $i} print o}' "$EXP_LOG")
	  [ "$n" -gt 0 ] || exp_warn "$1 keep $2: no rows in $EXP_LOG"
	}
	# Linux: the engine saves the sequence to a file on the NVMe and loads it
	# back, the whole of it; its prefills are the CPU recompute.
	exp_host linux-file 1 "'$cpu/llama-kv-bench' -m '$model' $bench -C $(exp_cpu_mask) --cpu-strict 1 \
	  --swap file --swap-file '$T_SCRATCH/kvswap/state.bin'"
	rows linux ""
	if [ {{recompute}} = 1 ]; then
	  EXP_HOST_ENV="${TGT_HOST_ACCEL_ENV[$TARGET]}" exp_host recompute-gpu 1 "'$gpu/llama-kv-bench' -m '$model' $bench -ngl 99 --swap none"
	  rows recompute-gpu ""
	fi
	# LROS: the KV in a page-cache window over a raw NVMe area, squeezed to
	# the share that stays and read back in bulk before the decode.
	for k in $(echo {{keeps}} | tr ',' ' '); do
	  exp_guest "lros-keep$k" 1 "$remote" "--kv-swap nvme:2 --kv-page-size {{page}} kv-bench -m nvme:1 --lros $bench \
	    --swap page --keep $k --restore bulk" "$model" "$area"
	  rows lros "$k"
	done
	exp_finish

[doc('(c) the first token of an interactive request arriving during a background prefill: llama.cpp, LROS, and alone, on one device')]
preempt device="cpu" target="joy":
	#!/usr/bin/env bash
	set -euo pipefail
	# The scheduling runs on the preemption trace (fill 0): llama.cpp's slots
	# engine (fifo) on the host against LROS in the guest.
	case "{{device}}" in
	  cpu) just -f {{HERE}}/../justfile sched::cpu {{target}} Llama-3.2-1B-Instruct-f16.gguf 0 64 fifo,lros,alone mot-preempt-cpu cpu-clang ;;
	  *)   just -f {{HERE}}/../justfile sched::accel {{target}} Llama-3.2-1B-Instruct-f16.gguf 0 64 fifo,lros,alone mot-preempt-{{device}} ;;
	esac

[doc('weights paged under a memory cap, per model: dense against mixture-of-experts')]
paging target="joy" models=PAGING_MODELS pcts="10,25,50,75" ntg="128" mlock="" name="mot-paging" preset="cpu-clang":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_begin {{name}} "exp,target,platform,model,quota_pct,quota_mb,anon_mb,model_mb,rep,pp,tg,t_pp,s_pp,t_tg,s_tg,majflt,read_mb,rss_mb,lck_mb"
	export EXP_NO_DROP=1 EXP_TIMEOUT=3600
	bin=$(exp_build_host {{preset}} llama-cli) || exp_die "no host build"
	tgt_push "$EXP_ROOT/scripts/paging.sh" bin/paging.sh
	# Real text and sampled tokens, so that a mixture of experts routes as in use.
	prompt="The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a screw press adapted from wine making. Within fifty years presses operated in more than two hundred cities across Europe, and the cost of books fell sharply. Scholars, merchants and reformers used the new medium to spread ideas faster than any authority could control. Explain in detail how the printing press changed science, religion, politics and everyday life, and compare its effects with those of the internet."
	cli="-no-cnv -p \"$prompt\" -n {{ntg}} --ignore-eos --seed 1 -c 1024 -t $(exp_ncores) -C $(exp_cpu_mask) --cpu-strict 1"
	# run <model> <pct> <cap MiB>: one row, the cap being the uncapped run's own
	# memory plus pct of the model file, so that only the weights are squeezed.
	# With mlock, all but that many MiB of the weights' share is pinned.
	run() {
	  local pin=""
	  if [ -n "{{mlock}}" ] && [ "$3" -gt 0 ]; then
	    pin="LLAMA_MLOCK_LIMIT_MB=$(( $3 - anon - {{mlock}} ))"
	  fi
	  exp_host "${1%.gguf}-q$2" "$rep" "env $pin bash \$SCRATCH/bin/paging.sh '$bin/llama-cli' '$(exp_model $1)' $3 $cli${pin:+ --mlock}"
	  local stats r; stats=$(grep -a '^paging ' "$EXP_LOG" | awk '{print $3","$5","$9","$11}')
	  r=$(awk '/prompt eval time/ {pt=$6; pp=$9} / +eval time/ && !/prompt/ {tt=$5; tg=$8}
	    END {if (pp && tg) printf "%d,%d,%.3f,%.2f,%.3f,%.2f", pp, tg, pt/1000, pp*1000/pt, tt/1000, tg*1000/tt}' "$EXP_LOG")
	  [ "$2" = none ] && anon=$(grep -a '^paging ' "$EXP_LOG" | awk '{print $7}')
	  [ -n "$r" ] && [ -n "$stats" ] || { exp_warn "$1 at $2%: no result in $EXP_LOG"; return; }
	  exp_row {{name}} "$TARGET" "$T_PLATFORM" "$1" "$2" "$3" "$anon" "$size" "$rep" "$r" "$stats"
	}
	for m in $(echo {{models}} | tr ',' ' '); do
	  size=$(( $(tgt_sh "stat -c %s '$(exp_model $m)'") / 1048576 ))
	  rep=0 anon=0
	  run "$m" none 0
	  [ "$anon" -gt 0 ] || { exp_warn "$m: the uncapped run failed; skipped"; continue; }
	  for rep in $(exp_reps); do
	    for pct in $(echo {{pcts}} | tr ',' ' '); do
	      run "$m" "$pct" $(( anon + size * pct / 100 ))
	    done
	  done
	done
	exp_finish

[doc('the paging runs in the LROS guest, with the page cache counters compiled in')]
paging-lros target="joy" models=PAGING_MODELS pcts="10,25,50,75" ntg="128" policy="pin" buffer="tensor" name="mot-paging-lros" prefetch="0" layer="0" aio="io_uring" mdts="9":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_begin {{name}} "exp,target,platform,model,quota_pct,cache_mb,pin_mb,buffer,rep,pp,tg,t_pp,s_pp,t_tg,s_tg,faults,hits,read_mb,evicted,fault_ms,claim_ms,room_ms,wait_ms,adopt_ms,spin_ms"
	export EXP_TIMEOUT=${EXP_TIMEOUT:-3600} EXP_NVME_CACHE=none EXP_NVME_AIO="{{aio}}" EXP_NVME_MDTS="{{mdts}}"
	remote=$(exp_ship_image cpu-stats "$(EXP_GUEST_CONF=conf_pagecache_stats=1 exp_build_guest cpu-stats 0 1)")
	prompt="The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a screw press adapted from wine making. Within fifty years presses operated in more than two hundred cities across Europe, and the cost of books fell sharply. Scholars, merchants and reformers used the new medium to spread ideas faster than any authority could control. Explain in detail how the printing press changed science, religion, politics and everyday life, and compare its effects with those of the internet."
	# The prompt goes on a data disk, the boot arguments being too short for it.
	mkdir -p "$LOCAL_TMP/paging-lros"; printf '%s' "$prompt" > "$LOCAL_TMP/paging-lros/prompt.txt"
	data=$(exp_mkdata "$LOCAL_TMP/paging-lros" paging-lros 16M)
	cli="llama-cli -m nvme:2 -no-cnv -f /root/prompt.txt -n {{ntg}} --ignore-eos --seed 1 -c 1024 -t $(exp_ncores) -C $(printf %x $(( (1 << $(exp_ncores)) - 1 ))) --cpu-strict 1"
	# run <model> <pct> <cache MiB>: the cache holds pct of the model, as the
	# Linux cap leaves it. policy=pin lets the guest pin the first units of
	# the model's use order and keep a window for streaming the rest;
	# buffer=tensor makes a fault read a tensor.
	run() {
	  local pin=0 pf="" bs="--buffer-size {{buffer}}"
	  [ "{{buffer}}" = tensor ] && bs="--per-tensor-buffers"
	  [ "{{policy}}" = pin ] && pin=1
	  [ "{{prefetch}}" != 0 ] && pf="--prefetch-depth {{prefetch}}"
	  [ "{{layer}}" != 0 ] && pf="$pf --layer-prefetch"
	  exp_guest "${1%.gguf}-q$2" "$rep" "$remote" \
	    "--cache-size $3M --pin-percent $pin $bs $pf $cli" "$data" "$(exp_model $1)"
	  local r st pinned
	  pinned=$(tr -d '\r' < "$EXP_LOG" | awk '/^page cache: / {print $7; exit}')
	  r=$(awk '/prompt eval time/ {pt=$6; pp=$9} / +eval time/ && !/prompt/ {tt=$5; tg=$8}
	    END {if (pp && tg) printf "%d,%d,%.3f,%.2f,%.3f,%.2f", pp, tg, pt/1000, pp*1000/pt, tt/1000, tg*1000/tt}' "$EXP_LOG")
	  # The last dump of the counters, which covers the decode only.
	  st=$(tr -d '\r' < "$EXP_LOG" | awk '/######## page cache/ {split("", v)}
	    / faults, .* found resident/ {v[1]=$1; v[2]=$3; v[3]=$6}
	    / buffers evicted/ {v[4]=$1}
	    /^  whole fault/ {v[5]=$3} /^  +claim and submit/ {v[6]=$4} /^  +make room/ {v[7]=$3}
	    /^  +wait for the device/ {v[8]=$5} /^  +install a prefetch/ {v[9]=$4} /^  +spin for another cpu/ {v[10]=$5}
	    END {for (i = 1; i <= 10; i++) printf "%s%s", (i > 1 ? "," : ""), (i in v ? v[i] : "")}')
	  [ -n "$r" ] || { exp_warn "$1 at $2%: no result in $EXP_LOG"; return; }
	  exp_row {{name}} "$TARGET" "$T_PLATFORM" "$1" "$2" "$3" "${pinned:-0}" "{{buffer}}" "$rep" "$r" "$st"
	}
	for m in $(echo {{models}} | tr ',' ' '); do
	  size=$(( $(tgt_sh "stat -c %s '$(exp_model $m)'") / 1048576 ))
	  rep=0
	  run "$m" none $(( size + 64 ))
	  for rep in $(exp_reps); do
	    for pct in $(echo {{pcts}} | tr ',' ' '); do
	      run "$m" "$pct" $(( size * pct / 100 ))
	    done
	  done
	done
	exp_finish

[doc('the paging runs in the LROS guest with VIAI, ngl layers offloaded to the GPU, the weight budget covering both copies')]
paging-lros-gpu target="joy" model=DENSE pcts="75,25,10" ngls="0,4,8,12,16" ntg="64" name="mot-paging-lros-gpu" aio="io_uring" mdts="9" stream="0":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_begin {{name}} "exp,target,platform,model,ngl,quota_pct,budget_mb,dev_mb,cache_mb,rep,status,pp,tg,t_pp,s_pp,t_tg,s_tg,faults,hits,read_mb,evicted"
	export EXP_TIMEOUT=${EXP_TIMEOUT:-3600} EXP_NVME_CACHE=none EXP_NVME_AIO="{{aio}}" EXP_NVME_MDTS="{{mdts}}"
	remote=$(exp_ship_image viai-stats "$(EXP_GUEST_CONF='conf_viai=1 conf_pagecache_stats=1' exp_build_guest viai-stats 0 1)")
	prompt="The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a screw press adapted from wine making. Within fifty years presses operated in more than two hundred cities across Europe, and the cost of books fell sharply. Scholars, merchants and reformers used the new medium to spread ideas faster than any authority could control. Explain in detail how the printing press changed science, religion, politics and everyday life, and compare its effects with those of the internet."
	mkdir -p "$LOCAL_TMP/paging-lros-gpu"; printf '%s' "$prompt" > "$LOCAL_TMP/paging-lros-gpu/prompt.txt"
	data=$(exp_mkdata "$LOCAL_TMP/paging-lros-gpu" paging-lros-gpu 16M)
	cli="llama-cli -m nvme:2 -no-cnv -f /root/prompt.txt -n {{ntg}} --ignore-eos --seed 1 -c 1024 -t $(exp_ncores) -C $(printf %x $(( (1 << $(exp_ncores)) - 1 ))) --cpu-strict 1"
	# The offloaded layers live in device memory that QEMU holds on the host,
	# outside the guest. The budget for the weights is pct of the model file,
	# as under the Linux cap, and covers both copies: the guest's cache gets
	# what the device copy leaves.
	# With stream=1, VIAI also runs the matmuls whose weights stay in the guest,
	# and the scheduler copies those weights to the device at every evaluation.
	# run <tag> <ngl> <pct> <cache MiB>
	run() {
	  local env=""
	  [ "{{stream}}" != 0 ] && env="--env VIAI_STREAM=1"
	  EXP_VACCEL=1 exp_guest "${model%.gguf}-ngl$2-q$3" "$rep" "$remote" \
	    "$env --cache-size $4M --pin-percent 1 --per-tensor-buffers $cli -ngl $2" "$data" "$(exp_model $model)"
	  dev=$(tr -d '\r' < "$EXP_LOG" | awk '/model buffer size/ && !/CPU/ {s += $(NF-1)} END {printf "%d", s}')
	  local r st status=ok
	  r=$(awk '/prompt eval time/ {pt=$6; pp=$9} / +eval time/ && !/prompt/ {tt=$5; tg=$8}
	    END {if (pp && tg) printf "%d,%d,%.3f,%.2f,%.3f,%.2f", pp, tg, pt/1000, pp*1000/pt, tt/1000, tg*1000/tt}' "$EXP_LOG")
	  st=$(tr -d '\r' < "$EXP_LOG" | awk '/######## page cache/ {split("", v)}
	    / faults, .* found resident/ {v[1]=$1; v[2]=$3; v[3]=$6} / buffers evicted/ {v[4]=$1}
	    END {for (i = 1; i <= 4; i++) printf "%s%s", (i > 1 ? "," : ""), (i in v ? v[i] : "")}')
	  [ -n "$r" ] || { status=failed; r=",,,,,"; exp_warn "ngl $2 at $3%: no result in $EXP_LOG"; }
	  exp_row {{name}} "$TARGET" "$T_PLATFORM" "$model" "$2" "$3" "$budget" "$dev" "$4" "$rep" "$status" "$r" "$st"
	}
	model={{model}}
	size=$(( $(tgt_sh "stat -c %s '$(exp_model $model)'") / 1048576 ))
	for ngl in $(echo {{ngls}} | tr ',' ' '); do
	  rep=0 budget=$(( size + 64 ))
	  run uncapped "$ngl" none $(( size + 64 ))
	  devngl=$dev
	  for rep in $(exp_reps); do
	    for pct in $(echo {{pcts}} | tr ',' ' '); do
	      budget=$(( size * pct / 100 )) cache=$(( size * pct / 100 - devngl ))
	      if [ "$cache" -lt 64 ]; then
	        dev=$devngl
	        exp_row {{name}} "$TARGET" "$T_PLATFORM" "$model" "$ngl" "$pct" "$budget" "$devngl" "$cache" "$rep" "over-budget" ",,,,," ",,,"
	        continue
	      fi
	      run capped "$ngl" "$pct" "$cache"
	    done
	  done
	done
	exp_finish

[doc('the paging results of the paper on one board, under the names the figures read')]
paging-all target="joy":
	#!/usr/bin/env bash
	set -euo pipefail
	j="just -f {{HERE}}/../justfile"
	# llama.cpp as shipped, then with all but 32 MiB of the weights' share locked.
	$j motivation::paging {{target}} {{DENSE}} 10,25,50,75 128 "" mot-paging-f16-default
	$j motivation::paging {{target}} {{DENSE}} 10,25,50,75 128 32 mot-paging-f16-final
	$j motivation::paging {{target}} {{MOE}} 10,25,50,75 128 "" mot-paging-moe-final
	$j motivation::paging-lros {{target}} {{DENSE}} 10,25,50,75 128 pin tensor mot-paging-lros-f16-final
	$j motivation::paging-lros {{target}} {{MOE}} 10,25,50,75 128 pin tensor mot-paging-lros-moe-final

[doc('the three motivation figures')]
plot out=PAPER:
	#!/usr/bin/env bash
	set -euo pipefail
	R={{RESULTS}}
	cd {{HERE}}/plots
	export PLOT_USETEX=1   # the paper's figures are typeset by LaTeX
	python3 mot-legend.py {{out}}/mot-legend.pdf
	python3 mot-paging.py {{out}}/mot-paging.pdf \
	  --default $R/mot-paging-f16-default/joy/mot-paging-f16-default.csv \
	  --mlock $R/mot-paging-f16-final/joy/mot-paging-f16-final.csv \
	  --lros $R/mot-paging-lros-f16-final/joy/mot-paging-lros-f16-final.csv
	python3 mot-kvswap.py $R/mot-kvswap/joy/mot-kvswap.csv {{out}}/mot-kvswap.pdf
	args=()
	for d in cpu:joy gpu:joy npu:orangepi; do
	  f=$R/mot-preempt-${d%%:*}/${d#*:}/mot-preempt-${d%%:*}.csv
	  [ -f "$f" ] && args+=("$(echo ${d%%:*} | tr a-z A-Z)" "$f")
	done
	python3 mot-preempt.py {{out}}/mot-preempt.pdf "${args[@]}"
