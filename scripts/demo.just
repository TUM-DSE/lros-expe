cli:
	@just -f {{justfile()}} run_lros_stdout \
	{{LROS_PATH}}/.unikraft/build-cli/cli_qemu-arm64 \
	8G "llama-cli -m /root/models/Llama-3.2-1B-Instruct-f16.gguf --no-mmap --no-perf --simple-io"
	
paging_demo:
	#!/usr/bin/env bash

		set -euo pipefail

		# Shared arguments (16 prompt token, 8 text gen token, 1 batch, 1 thread, no detailed integrated timings)
		args="-npp 16 -ntg 1 -t 1 --no-perf"

		# Evict the model from the page cache before each run so no config
		# benefits from a previous run's warm cache
		vmtouch -qe {{MODEL}}
		# Linux process (host) — cgroup MemoryMax=$m matches the VM memory size
		# read is OOM-killed below 16G (model does not fit): expected failure
		systemd-run --user --scope -p MemoryMax="512M" -p MemorySwapMax=0 --quiet -- \
			taskset -c {{AFFINITY}} \
			{{LLAMACPP_BIN}} $args -npl 1 --no-mmap \
			-m {{MODEL}} || echo "Ca"
		vmtouch -qe {{MODEL}}
		systemd-run --user --scope -p MemoryMax="512M" -p MemorySwapMax=0 --quiet -- \
			taskset -c {{AFFINITY}} \
			{{LLAMACPP_BIN}} $args -npl 1 \
			-m {{MODEL}}

		# LROS
		vmtouch -qe {{MODEL}}
		just -f {{justfile()}} EVAL_ARGS="$args --no-mmap" run_lros \
			{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 \
			/dev/null /dev/null \
			512M

		vmtouch -qe {{MODEL}}
		just -f {{justfile()}} EVAL_ARGS="libukvmem.prefetch_size=64 -- $args" run_lros \
			{{LROS_PATH}}/.unikraft/build-cpu/bench-cpu_qemu-arm64 \
			/dev/null /dev/null \
			512M