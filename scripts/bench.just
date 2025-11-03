import? "../justfile"

LROS_IMAGE_NAME := "llama.cpp_qemu-arm64"
LROS_PATH := proot + "/../lros/"

LLAMA_CPP_TARGETS := "llama-batched-bench"

OUT_DIR := proot + "/../bench/out"

EVAL_ARGS := "-npp 16,32 -ntg 128,256 -npl 1,2 -t 1 -ngl 20 --no-mmap --no-perf"


run_lros img out out_err cpu_type="big" size_mem="16G" :
	taskset -c {{ if cpu_type == "big" { "4-4" } else { "0-0" } }} qemu-system-aarch64 \
		-kernel {{img}} \
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
	> {{out}} \
	2> {{out_err}}


build spec variant:
	#!/usr/bin/env bash

	set -e

	case {{lowercase(spec)}} in
		lros)
			cd {{LROS_PATH}}
			case {{lowercase(variant)}} in
				all | vaccel )
					kraft build -t bench-vaccel --no-configure
				;;&
				all | vaccel-novec )
					kraft build -t bench-vaccel-novec --no-configure
				;;&
				all | cpu )
					kraft build -t bench-cpu --no-configure
				;;&
				all | cpu-novec )
					kraft build -t bench-cpu-novec --no-configure
				;;&
				all | vaccel | vaccel-novec | cpu | cpu-novec)
				;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >2
				;;
			esac
		;;

		linux-vm | vm | linux | process)
			if [ ! -x "$(command -v llama-batched-bench)" ]; then  # Check for external build
				[ -x "$(command -v cmake)" ] && cd {{LROS_PATH}}/llama.cpp-rknn
				# Only run setup if build files don't exist already
				case {{lowercase(variant)}} in
					all | cpu )
						# CPU-only
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build/CMakeCache.txt ] || cmake -B build . -DGGML_CUDA=OFF -DGGML_RKNN=OFF -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
						cmake --build build -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
					;;&
					cpu-thp )
						# CPU with huge page
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build-thp/CMakeCache.txt ] || cmake -B build-thp . -DGGML_CUDA=OFF -DGGML_RKNN=OFF -DGGML_VACCEL=OFF -DGGML_THP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
						cmake --build build-thp -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
					;;
					all | cuda )
						# CUDA
						if [ -x "$(command -v nvcc)" ]; then
							[ -f {{LROS_PATH}}/llama.cpp-rknn/build-cuda/CMakeCache.txt ] || cmake -B build-cuda . -DGGML_CUDA=ON -DGGML_RKNN=OFF -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
							cmake --build build-cuda -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
						elif [ {{lowercase(variant)}} -eq "cuda" ]; then
							echo "Cannot build CUDA. Could not find nvcc." >2
							exit 1
						fi
					;;&
					all | rknn )
						# RKNN
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build-rknn/CMakeCache.txt ] || cmake -B build-rknn . -DGGML_CUDA=OFF -DGGML_RKNN=ON -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
						cmake --build build-rknn -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
					;;&
					all | cpu | cuda | rknn)
					;;
					*)
						echo "Unknown variant {{variant}} for spec {{spec}}" >2
					;;
				esac
			fi
		;;

		* )
			echo "Unknown spec {{spec}}" >2
		;;
	esac

bench spec variant="all" size_mem="16G": (build spec variant)
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
			case {{lowercase(variant)}} in
				all | vaccel )
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-vaccel_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_vaccel_$(date -Iseconds).out.txt \
						{{OUT_DIR}}/bench_lros_vaccel_$(date -Iseconds).err.txt
				;;&
				all | vaccel-novec )
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-vaccel-novec_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_vaccel-novec_$(date -Iseconds).out.txt \
						{{OUT_DIR}}/bench_lros_vaccel-novec_$(date -Iseconds).err.txt
				;;&
				all | cpu )
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-cpu_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_cpul_$(date -Iseconds).out.txt \
						{{OUT_DIR}}/bench_lros_cpu_$(date -Iseconds).err.txt
				;;&
				all | cpu-novec )
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-cpu-novec_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_cpu-novec_$(date -Iseconds).out.txt \
						{{OUT_DIR}}/bench_lros_cpu-novec_$(date -Iseconds).err.txt
				;;&
				all | vaccel | vaccel-novec | cpu | cpu-novec)
				;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >2
				;;
			esac
		;;

		linux-vm | vm)
			just vm 1 {{size_mem}} > /dev/null 2> /dev/null &

			sleep 40

			just ssh_local "/root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" \
				> {{OUT_DIR}}/bench_vm_$(date -Iseconds).out.txt 2> {{OUT_DIR}}/bench_vm_$(date -Iseconds).err.txt

			sleep 40

			just ssh_local "poweroff"
		;;

		linux | process)
			case {{lowercase(variant)}} in
				all | cpu )
					{{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
						2> >(tee {{OUT_DIR}}/bench_process_cpu_$(date -Iseconds).err.txt >&2) \
						| tee {{OUT_DIR}}/bench_process_cpu_$(date -Iseconds).out.txt
				;;&
				all | cuda )
					# CUDA
					if [ -d /proc/driver/nvidia/ ]; then
						{{proot}}/../lros/llama.cpp-rknn/build-cuda/bin/llama-batched-bench {{EVAL_ARGS}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
							2> >(tee {{OUT_DIR}}/bench_process_cuda_$(date -Iseconds).err.txt >&2) \
							| tee {{OUT_DIR}}/bench_process_cuda_$(date -Iseconds).out.txt
					elif [ {{lowercase(variant)}} -eq "cuda" ]; then
						echo "Cannot execute CUDA on this device. Could not find GPU driver." >2
						exit 1
					fi
				;;&
				all | rknn )
					# RKNN
					if [ -d /sys/devices/platform/fdab0000.npu ]; then
						{{proot}}/../lros/llama.cpp-rknn/build-rknn/bin/llama-batched-bench {{EVAL_ARGS}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
							2> >(tee {{OUT_DIR}}/bench_process_rknn_$(date -Iseconds).err.txt >&2) \
							| tee {{OUT_DIR}}/bench_process_rknn_$(date -Iseconds).out.txt
					elif [ {{lowercase(variant)}} -eq "rknn" ]; then
						echo "Cannot execute RKNN on this device. Could not find NPU." >2
						exit 1
					fi
				;;&
				all | cpu | cuda | rknn)
				;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >2
				;;
			esac
		;;
	esac

hugepage: (build "linux" "cpu") (build "linux" "cpu-thp")
	#!/usr/bin/env bash

	set -ex

	mkdir -p {{OUT_DIR}}/thp

	perf stat -e l1d_tlb,l1d_tlb_refill,l2d_tlb,l2d_tlb_refill,dtlb_walk,l1i_tlb,l1i_tlb_refill,itlb_walk -j \
		-o {{OUT_DIR}}/thp/perf_base_$(date -Iseconds).json \
		{{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
		2> >(tee {{OUT_DIR}}/thp/bench_base_$(date -Iseconds).err.txt >&2) \
		| tee {{OUT_DIR}}/thp/bench_base_$(date -Iseconds).out.txt

	perf stat -e l1d_tlb,l1d_tlb_refill,l2d_tlb,l2d_tlb_refill,dtlb_walk,l1i_tlb,l1i_tlb_refill,itlb_walk -j \
		-o {{OUT_DIR}}/thp/perf_thp_$(date -Iseconds).json \
		{{proot}}/../lros/llama.cpp-rknn/build-thp/bin/llama-batched-bench {{EVAL_ARGS}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
		2> >(tee {{OUT_DIR}}/thp/bench_thp_$(date -Iseconds).err.txt >&2) \
		| tee {{OUT_DIR}}/thp/bench_thp_$(date -Iseconds).out.txt
