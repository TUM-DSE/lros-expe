# Accelerators: what the VIAI path costs against the host's own accelerator
# library, and how much code a device takes.

HERE := source_directory()
RESULTS := HERE / "../results"
LIB := HERE / "lib.sh"
PAPER := HERE / "../../lros_paper/plots"

[doc('CPU, VIAI and the native accelerator library, in the guest and on the host')]
perf target="joy" models="Llama-3.2-1B-Instruct-f16.gguf,gemma-3-1b-it-Q4_1.gguf" npp="512" ntg="64":
	#!/usr/bin/env bash
	source {{LIB}}; exp_target {{target}}
	exp_setup
	exp_sync_models
	[ -n "$T_ACCEL" ] || exp_die "no native accelerator library for $TARGET"
	exp_has_accel || exp_die "no accelerator stack on $TARGET; run: just boards::accel-setup $TARGET"
	tgt_sh "${TGT_VACCEL_ENV[$TARGET]}
	        grep -qa viai_ \"\$VACCEL_PLUGINS\"" \
	  || exp_die "the plugin on $TARGET has no VIAI server; run: just boards::accel-plugin $TARGET"
	exp_begin accel-perf "exp,target,platform,model,arm,rep,pass,pp,tg,b,n_kv,t_pp,s_pp,t_tg,s_tg,t,s"
	nt=$(exp_ncores); [ "$T_VCPUS" -lt "$nt" ] && nt=$T_VCPUS
	# The prompt runs twice, the second pass measured warm. The guest has no load
	# balancer, so ggml's threads are pinned one per core there.
	bench="-npp 1,{{npp}},{{npp}} -ntg {{ntg}} -npl 1 -t $nt --no-perf"
	pin="-C $(printf '%x' $(( (1 << nt) - 1 ))) --cpu-strict 1"
	case "$T_ACCEL" in cuda) native="-ngl 99" ;; *) native="" ;; esac

	# An accelerator arm counts only when its log names the device, so a silent
	# fallback to the CPU is dropped rather than reported.
	rows() {   # <model> <arm> <marker regex>
	  local pass=0 r
	  if [ -n "$3" ] && ! grep -qE "$3" "$EXP_LOG"; then
	    exp_warn "$2 $1 rep $rep: no '$3' in $EXP_LOG; rows dropped"; return
	  fi
	  while IFS= read -r r; do
	    [ "${r%%,*}" = "{{npp}}" ] || continue
	    pass=$((pass + 1))
	    exp_row accel-perf "$TARGET" "$T_PLATFORM" "$1" "$2" "$rep" "$pass" "$r"
	  done < <(exp_parse bench "$EXP_LOG")
	}

	remote_cpu=$(exp_ship_image cpu "$(exp_build_guest cpu 0 1)")
	remote_viai=$(exp_ship_image viai "$(exp_build_guest viai)")
	bin_cpu=$(exp_build_host cpu llama-batched-bench) || exp_die "cannot build the host cpu baseline"
	bin_acc=$(exp_build_host "$T_ACCEL" llama-batched-bench) || exp_die "cannot build the host $T_ACCEL baseline"
	for rep in $(exp_reps); do
	  for m in $(echo {{models}} | tr ',' ' '); do
	    model=$(exp_model "$m") tag=${m%.gguf}
	    exp_host "host-cpu.$tag" "$rep" "'$bin_cpu/llama-batched-bench' -m '$model' $bench"
	    rows "$m" host-cpu ""
	    EXP_HOST_ENV="${TGT_HOST_ACCEL_ENV[$TARGET]}" \
	      exp_host "host-native.$tag" "$rep" "'$bin_acc/llama-batched-bench' -m '$model' $bench $native"
	    case "$T_ACCEL" in
	      cuda) rows "$m" host-native "found [1-9][0-9]* CUDA devices" ;;
	      *)    rows "$m" host-native "using device RKNN" ;;
	    esac
	    EXP_TIMEOUT=${GUEST_TIMEOUT:-420} exp_guest_retry "guest-cpu.$tag" "$rep" "$remote_cpu" \
	      "batched-bench -m nvme:1 $bench $pin" "$model"
	    rows "$m" guest-cpu ""
	    EXP_VACCEL=1 EXP_TIMEOUT=${GUEST_TIMEOUT:-420} exp_guest_retry "guest-viai.$tag" "$rep" "$remote_viai" \
	      "batched-bench -m nvme:1 $bench $pin -ngl 99" "$model"
	    rows "$m" guest-viai "using device viai"
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
	python3 accel-perf.py {{out}}/accel-perf.pdf $csvs
	[ -f {{RESULTS}}/accel-loc/local/accel-loc.csv ] || just -f {{HERE}}/accel.just loc
	python3 accel-loc.py {{RESULTS}}/accel-loc/local/accel-loc.csv {{out}}/accel-loc.pdf
