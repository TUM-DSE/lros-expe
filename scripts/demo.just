cli:
	@just -f {{justfile()}} run_lros_stdout \
	{{LROS_PATH}}/.unikraft/build-cli/cli_qemu-arm64 \
	8G "llama-cli -m /root/models/Llama-3.2-1B-Instruct-f16.gguf --no-mmap --no-perf --simple-io"
	
paging_demo:
	#!/usr/bin/env bash

		set -euo pipefail

		# Shared arguments (16 prompt token, 8 text gen token, 1 batch, 1 thread, no detailed integrated timings)
		args="-npp 16 -ntg 1 -t 1 --no-perf"


		echo "=========================="
		echo "========Linux:read========"
		echo "=========================="
		# Evict the model from the page cache before each run so no config
		# benefits from a previous run's warm cache
		vmtouch -qe {{MODEL}}
		# Linux process (host) — cgroup MemoryMax=$m matches the VM memory size
		# read is OOM-killed below 16G (model does not fit): expected failure
		systemd-run --user --scope -p MemoryMax="512M" -p MemorySwapMax=0 --quiet -- \
			taskset -c {{AFFINITY}} \
			{{LLAMACPP_BIN}} $args -npl 1 --no-mmap \
			-m {{MODEL}} || echo "Cannot run. Model does not fit into memory."

		echo "=========================="
		echo "========Linux:mmap========"
		echo "=========================="
		vmtouch -qe {{MODEL}}
		systemd-run --user --scope -p MemoryMax="512M" -p MemorySwapMax=0 --quiet -- \
			taskset -c {{AFFINITY}} \
			{{LLAMACPP_BIN}} $args -npl 1 \
			-m {{MODEL}}

		# LROS
		echo "=========================="
		echo "========LROS:read========="
		echo "=========================="
		vmtouch -qe {{MODEL}}
		just -f {{justfile()}} EVAL_ARGS="$args --no-mmap" run_lros \
			{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 \
			/dev/null /dev/null \
			512M

		echo "=========================="
		echo "======LROS:prefetch======="
		echo "=========================="
		vmtouch -qe {{MODEL}}
		just -f {{justfile()}} EVAL_ARGS="libukvmem.prefetch_size=64 -- $args" run_lros \
			{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 \
			/dev/null /dev/null \
			512M

boottime_demo:
	#!/usr/bin/env bash

	set -euo pipefail

	bt_script={{proot}}/benchmarks/boottime/boot_time_eval.bt
	work=$(mktemp -d)
	cleanup() {
		kill "${vm_pid:-}" 2>/dev/null || true
		if pgrep -f "$bt_script" > /dev/null; then
			setsid sudo pkill -INT -f "$bt_script" < /dev/null 2>/dev/null || true
			sleep 1
			setsid sudo pkill -KILL -f "$bt_script" < /dev/null 2>/dev/null || true
		fi
		rm -rf "$work"
		# an interrupted child may leave the terminal in a raw state; restore
		# it if we have one. Ignore SIGTTOU: changing tty settings from a
		# background process group would otherwise stop us here.
		trap '' TTOU
		stty sane 2>/dev/null || true
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

	# Attach the boot probes, then boot the VM with libstop preloaded and its
	# stdin on a pipe: qemu stops before dying so we can read its memory stats.
	start_trace() { # name qemu-binary launch-cmd...
		name=$1 qemu_real=$2; shift 2
		# setsid: op5 configures sudo with use_pty, which puts our controlling
		# terminal into raw mode for as long as bpftrace runs (staircased
		# output). In its own session sudo has no terminal to touch.
		setsid sudo "$(command -v bpftrace)" "$bt_script" "$qemu_real" \
			> "$work/$name.bt" 2> "$work/$name.bt.err" &
		bt_pid=$!
		until grep -q "Attaching" "$work/$name.bt" 2>/dev/null; do
			if ! pgrep -f "$bt_script" > /dev/null; then
				echo "bpftrace failed to start:" >&2
				cat "$work/$name.bt.err" >&2
				exit 1
			fi
			sleep 0.2
		done

		mkfifo "$work/$name.stdin"
		# Strip the cursor position query from the console output: the guest's
		# resize login script probes the terminal for its size, and the answer
		# our terminal sends would end up as stray input.
		LD_PRELOAD=$LIBSTOP "$@" < "$work/$name.stdin" \
			> >(sed -u $'s/\x1b\\[6n//g') 2>&1 &
		vm_pid=$!
		exec 3> "$work/$name.stdin"
	}

	# Wait for the exit stop, sample the RSS of the guest RAM mapping, release
	# qemu and report boot time and memory use.
	finish_trace() { # label kernel-end-event init-end-event
		local label=$1 kend=$2 iend=$3 i pid rss="?"
		for ((i = 0; i < 240; i++)); do
			grep -q "QEMU: exit" "$work/$name.bt" && break
			sleep 0.5
		done
		pid=$(pidof "$qemu_real") || echo "warning: qemu already exited" >&2
		if [ -n "${pid:-}" ]; then
			# The guest RAM is by far the largest mapping of the qemu process.
			rss=$(awk '$1 == "Size:" { sz = $2 } $1 == "Rss:" && sz > max { max = sz; rss = $2 }
				END { print rss " KiB" }' "/proc/$pid/smaps")
		fi
		exec 3>&-	# EOF on the stdin pipe releases the exit stop
		wait "$vm_pid" || true
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

		awk -F': ' -v label="$label" -v kend="$kend" -v iend="$iend" '
			/: QEMU: main$/               { boot = $1 }
			/: QEMU: kvm_cpu_exec$/       { cpu = $1 }
			!k && index($0, kend)         { k = $1 }
			iend && !i && index($0, iend) { i = $1 }
			END {
				if (!boot || !cpu || !k || (iend && !i)) {
					printf "%-8s incomplete trace\n", label
					exit
				}
				init = iend ? sprintf("%10.1f", (i - k) / 1e6) : sprintf("%10s", "-")
				printf "%-8s %10.1f %10.1f %s %10.1f\n",
					label, (cpu - boot) / 1e6, (k - cpu) / 1e6,
					init, ((iend ? i : k) - boot) / 1e6
			}
		' "$work/$name.bt" >> "$work/report"
		printf '%-8s %14s\n' "$label" "$rss" >> "$work/rss_report"
	}

	prog_vm="$({{proot}}/scripts/run_in_vm.sh llama.cpp-rknn/build-cpu/bin/llama-batched-bench)"

	echo "=========================="
	echo "==========Linux==========="
	echo "=========================="
	start_trace linux "$(resolve_qemu qemu-kvm)" just vm 1 2G big
	#	for ((i = 0; i < 60; i++)); do
	#		just ssh_local_clean true 2>/dev/null && break
	#		sleep 2
	#	done
	# sshd accepts connections before systemd is done: wait for the boot-end
	# event itself before shutting the VM down again
	for ((i = 0; i < 200; i++)); do
		grep -q "systemd init end" "$work/linux.bt" && break
		sleep 0.2
	done
	# mirror the LROS image: run llama.cpp without any configuration.
	# stdbuf: without a pty the guest side would block-buffer stdout and only
	# show it in one piece at the end (env runs the LD_LIBRARY_PATH= prefix
	# without a shell)
	just ssh_local_clean "stdbuf -oL env $prog_vm" || true
	just ssh_local_clean poweroff 2>/dev/null || true
	finish_trace "Linux" "231 Linux: init_start" "240 Linux: systemd init end"

	echo "=========================="
	echo "==========LROS============"
	echo "=========================="
	# batched-bench without arguments exits right after boot
	start_trace lros "$(resolve_qemu qemu-system-aarch64)" \
		just -f {{justfile()}} run_lros_stdout \
		{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 2G "llama-batched-bench --version"
	finish_trace "LROS" "250 Unikraft: init end" ""

	echo "=========================="
	echo "=========================="
	echo "Boot time breakdown (ms):"
	printf '%-8s %10s %10s %10s %10s\n' "" "QEMU" "Kernel" "Init" "Total"
	cat "$work/report"

	echo "=========================="
	echo "VM RAM RSS:"
	cat "$work/rss_report"

