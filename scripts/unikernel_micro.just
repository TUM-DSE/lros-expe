# Unikernel microbenchmarks: boot time and boot memory footprint.
#
# Non-interactive version of demo.just's boottime_demo: each config boots
# n_runs times under the bpftrace probes from benchmarks/boottime, and every
# boot becomes one CSV row instead of the terminal tables. Console output and
# the raw bpftrace logs are kept next to the CSV.
#
# Not standalone: relies on proot, OUT_DIR, LROS_PATH, LROS_IMAGE_NAME,
# AFFINITY and cpu_type from the importing justfile, and on the vm/ssh
# recipes from vm.just.
#
# Configs:
#   linux  the NixOS guest from vm.just (direct -kernel boot, no firmware)
#   lros   the unikernel image (miniOSv is its current backend, as Unikraft
#          was before), booted through AAVMF by miniosv/scripts/run.py
#
# Guest-side events (hvc #0x42, see boot_time_eval.bt): the patched Linux
# kernel/systemd emit 231/240; the miniOSv backend mirrors its boot time
# chart events (core/chart.cc), of which "premain" (kernel entry) and "Total
# time" (boot done) bound the kernel phase. Linux has no firmware column
# (-kernel boot skips UEFI); lros has no init column (the app is entered
# directly, there is no init system).
#
# Memory: libstop (LD_PRELOAD into qemu) parks qemu right before it exits,
# and the RSS of the guest-RAM mapping (the largest mapping in smaps) is
# sampled then -- the boot-plus-app footprint, since both configs run
# llama-batched-bench without a model and exit.
#
# Output: bench/out/boottime/<timestamp>/boottime.csv with columns
#   config,run,qemu_ms,firmware_ms,kernel_ms,init_ms,total_ms,rss_kib
# (empty field = phase does not exist for that config, or incomplete trace)

boottime n_runs="5" size_mem="2G" variant="cpu" configs="linux lros":
	#!/usr/bin/env bash

	set -euo pipefail

	bt_script={{proot}}/benchmarks/boottime/boot_time_eval.bt
	out_dir={{OUT_DIR}}/boottime/$(date -Iseconds)
	mkdir -p "$out_dir"
	csv=$out_dir/boottime.csv

	img={{LROS_PATH}}/build/{{variant}}.aarch64/{{LROS_IMAGE_NAME}}
	case " {{configs}} " in *" lros "*)
		if [ ! -f "$img" ]; then
			echo "Missing $img -- build it first: just build lros {{variant}}" >&2
			exit 1
		fi
	esac

	cleanup() {
		kill "${vm_pid:-}" 2>/dev/null || true
		if pgrep -f "$bt_script" > /dev/null; then
			setsid sudo pkill -INT -f "$bt_script" < /dev/null 2>/dev/null || true
			sleep 1
			setsid sudo pkill -KILL -f "$bt_script" < /dev/null 2>/dev/null || true
		fi
	}
	trap cleanup EXIT

	# bpftrace attaches its uprobes by binary path, so resolve the actual qemu
	# executable: qemu-kvm is a symlink to the qemu-system-aarch64 wrapper,
	# which execs the dot-prefixed real binary.
	resolve_qemu() {
		local p wrapped
		p=$(readlink -f "$(command -v "$1")")
		wrapped=$(dirname "$p")/.$(basename "$p")-wrapped
		if [ -e "$wrapped" ]; then echo "$wrapped"; else echo "$p"; fi
	}

	# Attach the boot probes, then boot the VM with its stdin on a pipe: with
	# libstop preloaded (the caller's job -- see measure_linux/measure_lros)
	# qemu stops before dying so we can read its memory stats. Only qemu may
	# get the preload: libstop parks whichever preloaded process exits first,
	# so a helper process exiting before qemu would steal the exit stop.
	start_trace() { # name qemu-binary launch-cmd...
		name=$1 qemu_real=$2; shift 2
		# setsid: op5 configures sudo with use_pty, which puts our controlling
		# terminal into raw mode for as long as bpftrace runs (staircased
		# output). In its own session sudo has no terminal to touch.
		setsid sudo "$(command -v bpftrace)" "$bt_script" "$qemu_real" "$LIBSTOP" \
			> "$out_dir/$name.bt" 2> "$out_dir/$name.bt.err" &
		bt_pid=$!
		until grep -q "Attaching" "$out_dir/$name.bt" 2>/dev/null; do
			if ! pgrep -f "$bt_script" > /dev/null; then
				echo "bpftrace failed to start:" >&2
				cat "$out_dir/$name.bt.err" >&2
				exit 1
			fi
			sleep 0.2
		done

		mkfifo "$out_dir/$name.stdin"
		"$@" < "$out_dir/$name.stdin" \
			> "$out_dir/$name.console" 2>&1 &
		vm_pid=$!
		exec 3> "$out_dir/$name.stdin"
	}

	# Wait for the exit stop, sample the RSS of the guest RAM mapping, release
	# qemu and append the run's CSV row. Empty kstart/iend skip that column.
	finish_trace() { # config run kernel-start-event kernel-end-event init-end-event
		local config=$1 run=$2 kstart=$3 kend=$4 iend=$5 i pid rss=""
		for ((i = 0; i < 240; i++)); do
			grep -q "QEMU: exit" "$out_dir/$name.bt" && break
			sleep 0.5
		done
		pid=$(pidof "$qemu_real") || echo "warning: qemu already exited" >&2
		if [ -n "${pid:-}" ] && [ -e "/proc/$pid/smaps" ]; then
			# The guest RAM is by far the largest mapping of the qemu process.
			rss=$(awk '$1 == "Size:" { sz = $2 } $1 == "Rss:" && sz > max { max = sz; rss = $2 }
				END { print rss }' "/proc/$pid/smaps")
		fi
		exec 3>&-	# EOF on the stdin pipe releases the exit stop
		# Bounded wait: a wedged qemu must not hang the whole sweep.
		for ((i = 0; i < 60; i++)); do
			kill -0 "$vm_pid" 2>/dev/null || break
			sleep 0.5
		done
		kill "$vm_pid" 2>/dev/null || true
		wait "$vm_pid" 2>/dev/null || true
		setsid sudo pkill -INT -f "$bt_script" 2>/dev/null || true
		for ((i = 0; i < 10; i++)); do
			pgrep -f "$bt_script" > /dev/null || break
			sleep 0.5
		done
		if pgrep -f "$bt_script" > /dev/null; then
			echo "bpftrace did not exit, killing it" >&2
			setsid sudo pkill -KILL -f "$bt_script" 2>/dev/null || true
		fi
		wait "$bt_pid" || true

		awk -F': ' -v config="$config" -v run="$run" -v rss="$rss" \
			-v kstart="$kstart" -v kend="$kend" -v iend="$iend" '
			/: QEMU: main$/                          { boot = $1 }
			/: QEMU: kvm_cpu_exec$/                  { cpu = $1 }
			kstart != "" && !ks && index($0, kstart) { ks = $1 }
			!k && index($0, kend)                    { k = $1 }
			iend != "" && !i && index($0, iend)      { i = $1 }
			END {
				if (!boot || !cpu || !k || (kstart != "" && !ks) || (iend != "" && !i)) {
					printf "%s run %s: incomplete trace\n", config, run > "/dev/stderr"
					printf "%s,%s,,,,,,%s\n", config, run, rss
					exit
				}
				fw   = (kstart != "") ? sprintf("%.1f", (ks - cpu) / 1e6) : ""
				kern = (kstart != "") ? sprintf("%.1f", (k - ks) / 1e6) \
				                      : sprintf("%.1f", (k - cpu) / 1e6)
				init = (iend != "") ? sprintf("%.1f", (i - k) / 1e6) : ""
				printf "%s,%s,%.1f,%s,%s,%s,%.1f,%s\n", config, run,
					(cpu - boot) / 1e6, fw, kern, init,
					((iend != "" ? i : k) - boot) / 1e6, rss
			}
		' "$out_dir/$name.bt" >> "$csv"
	}

	measure_linux() { # run
		prog_vm="$({{proot}}/scripts/run_in_vm.sh llama.cpp-rknn/build-cpu/bin/llama-batched-bench)"
		# Preloading the whole just-vm chain is fine here: qemu is the first
		# process in it to exit, so the exit stop is guaranteed to catch it.
		start_trace linux_run$1 "$(resolve_qemu qemu-kvm)" \
			env LD_PRELOAD=$LIBSTOP just vm 1 {{size_mem}} {{cpu_type}}
		# sshd accepts connections before systemd is done: wait for the
		# boot-end event itself before running the benchmark
		for ((i = 0; i < 200; i++)); do
			grep -q "systemd init end" "$out_dir/$name.bt" && break
			sleep 0.2
		done
		# mirror the unikernel image: run llama.cpp without any configuration
		# (prints its usage and exits). stdbuf: without a pty the guest side
		# would block-buffer stdout and only show it in one piece at the end
		# (env runs the LD_LIBRARY_PATH= prefix without a shell)
		just ssh_local_clean "stdbuf -oL env $prog_vm" \
			> "$out_dir/$name.bench" 2>&1 || true
		just ssh_local_clean poweroff 2>/dev/null || true
		finish_trace linux $1 "" "231 Linux: init_start" "240 Linux: systemd init end"
	}

	measure_lros() { # run
		# batched-bench --version exits right after boot; run.py boots the
		# image through AAVMF, so the trace includes a firmware phase
		# --preload keeps libstop out of run.py's helper subprocesses: stty
		# exits before qemu even starts and would otherwise take the stop.
		start_trace lros_run$1 "$(resolve_qemu qemu-system-aarch64)" \
			taskset -c {{AFFINITY}} python3 {{LROS_PATH}}/scripts/run.py \
			--arch aarch64 -p kvm -c 1 -m {{size_mem}} --nogdb \
			--preload "$LIBSTOP" \
			--image-path "$img" --args "batched-bench --version"
		finish_trace lros $1 "260 miniOSv: premain" "269 miniOSv: Total time" ""
	}

	echo "config,run,qemu_ms,firmware_ms,kernel_ms,init_ms,total_ms,rss_kib" > "$csv"

	for run in $(seq 1 {{n_runs}}); do
		for config in {{configs}}; do
			echo "=== run $run/{{n_runs}}: $config ==="
			measure_$config $run
		done
	done

	echo "Wrote $csv"
	column -t -s, "$csv"