run_vm out out_err variant="build" size_mem="16G" additional_vm_args="":
	#!/usr/bin/env bash

	set -e

	just vm 1 {{size_mem}} {{cpu_type}} '{{additional_vm_args}}' > tmp_out.txt 2>&1 &

	sleep 40

	just ssh_local "$({{proot}}/scripts/run_in_vm.sh llama.cpp-rknn/{{variant}}/bin/llama-batched-bench {{EVAL_ARGS}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf)" \
		> {{out}} 2> {{out_err}} || echo "Command exited with error code $?"

	sleep 5

	just ssh_local "poweroff" || ! jobs -p %% >/dev/null 2>/dev/null

	sleep 10

# Shared QEMU launcher for the unikraft images (bench + parallel).
# Silenced so callers that capture stdout (e.g. run_lros_stdout) stay clean.
# Pass vaccel="true" to attach the vAccel host backend + virtio-accel device.
_qemu img append size="16G" vaccel="false":
	@taskset -c {{AFFINITY}} qemu-system-aarch64 \
		-kernel {{img}} \
		-append "{{append}}" \
		-cpu host \
		-machine virt \
		-m size={{size}} \
		-smp cpus=1,threads=1,sockets=1 \
		-parallel none \
		-device virtio-9p-pci,fsdev=rootfs,mount_tag=rootfs \
		-fsdev local,id=rootfs,path={{proot}},security_model=mapped-xattr \
		-nic none \
		-display none \
		-nographic \
		-vga none \
		-no-reboot \
		-rtc base=utc \
		{{ if vaccel == "true" { "-object acceldev-backend-vaccel,id=gen0 -device virtio-accel-pci,id=accl0,runtime=gen0,disable-legacy=off,disable-modern=on,event_idx=off" } else { "" } }} \
		-enable-kvm

# Pass tee="true" to also stream stdout/stderr to the console while still
# capturing them to the out/out_err files. Shebang recipe (bash) both for the
# process substitution and so just doesn't echo the long qemu invocation.
run_lros img out out_err size_mem="16G" batch_size="1" vaccel="false" model="Llama-3.2-1B-Instruct-f16.gguf" tee="true":
	#!/usr/bin/env bash
	set -euo pipefail
	cmd="llama-batched-bench {{EVAL_ARGS}} -npl {{batch_size}} -m /root/models/{{model}}"
	if [ "{{tee}}" = "true" ]; then
		just -f {{justfile()}} _qemu {{img}} "$cmd" {{size_mem}} {{vaccel}} \
			2> >(tee {{out_err}} >&2) \
			| tee {{out}}
	else
		just -f {{justfile()}} _qemu {{img}} "$cmd" {{size_mem}} {{vaccel}} \
			> {{out}} 2> {{out_err}}
	fi

run_lros_stdout img size_mem="16G" args="":
	@just -f {{justfile()}} _qemu {{img}} "{{args}}" {{size_mem}}

run_preemption:
	just -f {{justfile()}} _qemu {{parallel_img}} \
		"llama-parallel -np 2 -ns 6 --arrival-interval-ms 10000 -n 20 -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" 8G

run_batch:
	just -f {{justfile()}} _qemu {{parallel_img}} \
		"llama-parallel -np 4 -ns 4 --arrival-interval-ms 10000 -n 20 -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" 8G