# Accelerators: what the VIAI path costs against the host's own accelerator
# library, and how much code a device takes.

HERE := source_directory()
RESULTS := HERE / "../results"
LIB := HERE / "lib.sh"
PAPER := HERE / "../../lros_paper/plots"

[doc('VIAI in the guest against the native accelerator library on the host, with llama-cli on real text')]
perf target="joy" models="Llama-3.2-1B-Instruct-f16.gguf,olmoe-1b-7b-0924-instruct-q8_0.gguf" ntg="64" mem="":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_sync_models
	[ -n "$T_ACCEL" ] || exp_die "no native accelerator library for $TARGET"
	exp_has_accel || exp_die "no accelerator stack on $TARGET; run: just boards::accel-setup $TARGET"
	tgt_sh "${TGT_VACCEL_ENV[$TARGET]}
	        grep -qa viai_ \"\$VACCEL_PLUGINS\"" \
	  || exp_die "the plugin on $TARGET has no VIAI server; run: just boards::accel-plugin $TARGET"
	exp_begin accel-perf "exp,target,platform,model,arm,rep,pass,pp,tg,t_pp,s_pp,t_tg,s_tg"
	nt=$(exp_ncores); [ "$T_VCPUS" -lt "$nt" ] && nt=$T_VCPUS
	# Real text and sampled tokens, so that a mixture of experts routes as in use.
	prompt="The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable metal type, oil-based ink and a screw press adapted from wine making. Within fifty years presses operated in more than two hundred cities across Europe, and the cost of books fell sharply. Scholars, merchants and reformers used the new medium to spread ideas faster than any authority could control. Explain in detail how the printing press changed science, religion, politics and everyday life, and compare its effects with those of the internet."
	gen="-no-cnv -n {{ntg}} --ignore-eos --seed 1 -c 1024 -t $nt"
	# The guest has no load balancer, so ggml's threads are pinned one per core there.
	pin="-C $(printf '%x' $(( (1 << nt) - 1 ))) --cpu-strict 1"
	case "$T_ACCEL" in cuda) native="-ngl 99" ;; *) native="" ;; esac
	# The NPU takes the matmuls the host rknnoh config offloads, ffn.* and no others.
	case "$T_ACCEL" in rknnoh) policy="--env VIAI_POLICY={nodes:ffn.*} " ;; *) policy="" ;; esac
	# The prompt goes on a data disk, the boot arguments being too short for it.
	mkdir -p "$LOCAL_TMP/accel-perf"; printf '%s' "$prompt" > "$LOCAL_TMP/accel-perf/prompt.txt"
	data=$(exp_mkdata "$LOCAL_TMP/accel-perf" accel-perf 16M)

	# An accelerator arm counts only when its log names the device, so a silent
	# fallback to the CPU is dropped rather than reported.
	row() {   # <model> <arm> <marker regex>
	  if ! grep -qaE "$3" "$EXP_LOG"; then
	    exp_warn "$2 $1 rep $rep: no '$3' in $EXP_LOG; row dropped"; return
	  fi
	  local r
	  r=$(tr -d '\r' < "$EXP_LOG" | awk '/prompt eval time/ {pt=$6; pp=$9} / +eval time/ && !/prompt/ {tt=$5; tg=$8}
	    END {if (pp && tg) printf "%d,%d,%.3f,%.2f,%.3f,%.2f", pp, tg, pt/1000, pp*1000/pt, tt/1000, tg*1000/tt}')
	  [ -n "$r" ] || { exp_warn "$2 $1 rep $rep: no timings in $EXP_LOG"; return; }
	  exp_row accel-perf "$TARGET" "$T_PLATFORM" "$1" "$2" "$rep" 1 "$r"
	}

	remote_viai=$(exp_ship_image viai "$(exp_build_guest viai)")
	bin_acc=$(exp_build_host "$T_ACCEL" llama-cli) || exp_die "cannot build the host $T_ACCEL baseline"
	for rep in $(exp_reps); do
	  for m in $(echo {{models}} | tr ',' ' '); do
	    model=$(exp_model "$m") tag=${m%.gguf}
	    EXP_HOST_ENV="${TGT_HOST_ACCEL_ENV[$TARGET]}" \
	      exp_host "host-native.$tag" "$rep" "'$bin_acc/llama-cli' -m '$model' -p '$prompt' $gen $native"
	    case "$T_ACCEL" in
	      cuda) row "$m" host-native "found [1-9][0-9]* CUDA devices" ;;
	      *)    row "$m" host-native "using device RKNN" ;;
	    esac
	    EXP_VACCEL=1 EXP_MEM="{{mem}}" EXP_TIMEOUT=${GUEST_TIMEOUT:-900} exp_guest_retry "guest-viai.$tag" "$rep" "$remote_viai" \
	      "${policy}llama-cli -m nvme:2 -f /root/prompt.txt $gen $pin -ngl 99" "$data" "$model"
	    row "$m" guest-viai "using device viai"
	  done
	done
	exp_finish

[doc('lines of code of the VIAI path per device, against the native backends')]
loc:
	#!/usr/bin/env python3
	import csv, os, re
	root = os.path.realpath("{{HERE}}/..")
	ggml = os.path.join(root, "miniosv/app/llama.cpp/ggml")
	plug = os.path.join(root, "vaccel_plugins")

	def count(path):
	    """Lines that are neither blank nor only comment."""
	    text = open(path, errors="replace").read()
	    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
	    return sum(1 for l in text.splitlines() if l.strip() and not l.strip().startswith("//"))

	def tree(path):
	    ext = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")
	    return sum(count(os.path.join(d, f)) for d, _, fs in os.walk(path)
	               for f in fs if f.endswith(ext))

	guest = sum(count(f"{ggml}/{p}") for p in
	            ("src/ggml-viai/ggml-viai.cpp", "src/ggml-viai/viai-proto.h", "include/ggml-viai.h"))
	server = count(f"{plug}/viai_server.cpp") + count(f"{plug}/ggml_probe.cpp")
	rows = [
	    ("orin", "guest backend", guest), ("orin", "host server", server),
	    ("orin", "device adapter", 0), ("orin", "native backend", tree(f"{ggml}/src/ggml-cuda")),
	    ("rk3588", "guest backend", guest), ("rk3588", "host server", server),
	    ("rk3588", "device adapter", count(f"{plug}/viai_rknn.cpp")),
	    ("rk3588", "native backend", tree(f"{ggml}/src/ggml-rknnoh")),
	]
	out = os.path.join("{{RESULTS}}", "accel-loc", "local")
	os.makedirs(out, exist_ok=True)
	with open(os.path.join(out, "accel-loc.csv"), "w", newline="") as f:
	    csv.writer(f).writerows([("platform", "part", "loc")] + rows)
	for r in rows:
	    print(f"{r[0]:7s} {r[1]:15s} {r[2]:6d}")

[doc('both accelerator figures')]
plot out=PAPER:
	#!/usr/bin/env bash
	set -euo pipefail
	csvs=$(ls {{RESULTS}}/accel-perf/*/accel-perf.csv 2>/dev/null || true)
	[ -n "$csvs" ] || { echo "no accel-perf results yet; run: just accel::perf <target>" >&2; exit 1; }
	cd {{HERE}}/plots
	export PLOT_USETEX=1   # the paper's figures are typeset by LaTeX
	python3 accel-perf.py {{out}}/accel-perf.pdf $csvs
	[ -f {{RESULTS}}/accel-loc/local/accel-loc.csv ] || just -f {{HERE}}/accel.just loc
	python3 accel-loc.py {{RESULTS}}/accel-loc/local/accel-loc.csv {{out}}/accel-loc.pdf
