import? "../justfile"

LROS_IMAGE_NAME := "llama.cpp_qemu-arm64"
LROS_PATH := proot + "/../lros/"

LLAMA_CPP_TARGETS := "llama-batched-bench"

OUT_DIR := proot + "/../bench/out"

EVAL_ARGS := "-npp 16,32 -ntg 128,256 -npl 1,2 -t 1 -ngl 20 --no-mmap --no-perf"



build spec:
	#!/usr/bin/env bash

	set -e

	case {{lowercase(spec)}} in
		lros)
			cd {{LROS_PATH}}
			kraft build -c {{LROS_PATH}}/bench.vaccel.config.lama.cpp_qemu-arm64 --no-configure && mv .unikraft/build/llama.cpp_qemu-arm64 bench-vaccel_qemu-arm64
			kraft build -c {{LROS_PATH}}/bench.vaccel-novec.config.lama.cpp_qemu-arm64 --no-configure && mv .unikraft/build/llama.cpp_qemu-arm64 bench-vaccel-novec_qemu-arm64
			kraft build -c {{LROS_PATH}}/bench.cpu.config.lama.cpp_qemu-arm64 --no-configure && mv .unikraft/build/llama.cpp_qemu-arm64 bench-cpu_qemu-arm64
			kraft build -c {{LROS_PATH}}/bench.cpu-novec.config.lama.cpp_qemu-arm64 --no-configure && mv .unikraft/build/llama.cpp_qemu-arm64 bench-cpu-nopvec_qemu-arm64
		;;

		linux-vm | vm | linux | process)
			if [ ! -x "$(command -v llama-batched-bench)" ]; then  # Check for external build
				[ -x "$(command -v cmake)" ] && cd {{LROS_PATH}}/llama.cpp-rknn
				# Only run setup if build files don't exist already

				# CPU-only
				[ -f {{LROS_PATH}}/llama.cpp-rknn/build/CMakeCache.txt ] || cmake -B build . -DGGML_CUDA=OFF -DGGML_RKNN=OFF -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release
				cmake --build build -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)

				# CUDA
				if [ -x "$(command -v nvcc)" ]; then
					[ -f {{LROS_PATH}}/llama.cpp-rknn/build-cuda/CMakeCache.txt ] || cmake -B build-cuda . -DGGML_CUDA=ON -DGGML_RKNN=OFF -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release
					cmake --build build-cuda -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
				fi

				# RKNN
				if [ -d /sys/devices/platform/fdab0000.npu ]; then
					[ -f {{LROS_PATH}}/llama.cpp-rknn/build-rknn/CMakeCache.txt ] || cmake -B build-rknn . -DGGML_CUDA=OFF -DGGML_RKNN=ON -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release
					cmake --build build-rknn -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
				fi
			fi
		;;
	esac

bench spec size_mem="16G": (build spec)
	#!/usr/bin/env bash

	set -ex

	mkdir -p {{OUT_DIR}}

	unset VACCEL_PLUGINS
	unset VACCEL_BACKENDS

	# Detect if we are on NPU or GPU system
	if [ -d /sys/devices/platform/fdab0000.npu ]; then
		export VACCEL_PLUGINS=$VACCEL_PLUGINS_RKNN
	elif [ -d /proc/driver/nvidia/ ]; then
		export VACCEL_PLUGINS=$VACCEL_PLUGINS_CUDA
	fi

	if [ ! -f {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf ]; then
		echo "Please download the Llama-3.2-1B-Instruct-f16.gguf model." >&2
		exit 1
	fi

	case {{lowercase(spec)}} in
		lros)
			qemu-system-aarch64 \
				-kernel {{LROS_PATH}}/.unikraft/build/{{LROS_IMAGE_NAME}} \
				-append "llama-batched-bench {{EVAL_ARGS}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" \
				-cpu host \
				-machine virt \
				-m size={{size_mem}} \
				-smp cpus=1,threads=1,sockets=1 \
				-parallel none \
				-device virtio-9p-pci,fsdev=rootfs,mount_tag=rootfs \
				-fsdev local,id=rootfs,path={{proot}}/..,security_model=mapped-xattr \
				-object acceldev-backend-vaccel,id=gen0 \
				-device virtio-accel-pci,id=accl0,runtime=gen0,disable-legacy=off,disable-modern=on,event_idx=off \
				-display none \
				-nographic \
				-vga none \
				-no-reboot \
				-rtc base=utc \
				-enable-kvm \
			> {{OUT_DIR}}/bench_lros_$(date -Iseconds).out.txt \
			2> {{OUT_DIR}}/bench_lros_$(date -Iseconds).err.txt

		;;

		linux-vm | vm)
			just vm 1 {{size_mem}} > /dev/null 2> /dev/null &

			sleep 40

			just ssh "/root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" \
				> {{OUT_DIR}}/bench_vm_$(date -Iseconds).out.txt 2> {{OUT_DIR}}/bench_vm_$(date -Iseconds).err.txt

			sleep 40

			just ssh "poweroff"
		;;

		linux | process)
			{{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
				2> >(tee {{OUT_DIR}}/bench_process_$(date -Iseconds).err.txt >&2) \
				| tee {{OUT_DIR}}/bench_process_$(date -Iseconds).out.txt
		;;
	esac
