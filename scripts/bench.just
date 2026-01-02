import? "../justfile"

LROS_IMAGE_NAME := "llama.cpp_qemu-arm64"
LROS_PATH := proot + "/../lros/"

LLAMA_CPP_TARGETS := "llama-batched-bench"

OUT_DIR := proot + "/../bench/out"

EVAL_ARGS := "-npp 16,32 -ntg 128,256 -t 1 -ngl 20 --no-mmap --no-perf"
EVAL_ARGS_SINGLE := "-npp 16 -ntg 128 -npl 1 -t 1 -ngl 20 --no-mmap --no-perf"


run_vm out out_err size_mem="16G" cpu_type="big" additional_vm_args="":
	#!/usr/bin/env bash

	set -e

	just vm 1 {{size_mem}} {{cpu_type}} '{{additional_vm_args}}' > /dev/null 2> /dev/null &

	sleep 40

	just ssh_local "/root/scripts/run_in_vm.sh /root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" \
		> {{out}} 2> {{out_err}} || echo "Command exited with error code $?"

	sleep 5

	just ssh_local "poweroff" || ! jobs -p %% >/dev/null 2>/dev/null

	sleep 10

run_lros img out out_err cpu_type="big" size_mem="16G" batch_size="1":
	taskset -c {{ if cpu_type == "big" { "4-4" } else { "0-0" } }} qemu-system-aarch64 \
		-kernel {{img}} \
		-append "llama-batched-bench {{EVAL_ARGS}} -npl {{batch_size}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" \
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

bench spec variant="all" cpu_type="big" size_mem="16G" $log_power="false" $batch_size="1": (build spec variant)
	#!/usr/bin/env bash

	set -ex

	mkdir -p {{OUT_DIR}}

	unset VACCEL_PLUGINS
	unset VACCEL_BACKENDS

	# Detect if we are on NPU or GPU system
	if [ -d /sys/devices/platform/fdab0000.npu ]; then
		export VACCEL_PLUGINS=$VACCEL_PLUGINS_RKNN
		powerlogger="measure_orangepi"
	elif [ -d /proc/driver/nvidia/ ]; then
		export VACCEL_PLUGINS=$VACCEL_PLUGINS_CUDA
		powerlogger="measure_jetson 0.1"
	fi

	if [ ! -f {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf ]; then
		echo "Please download the Llama-3.2-1B-Instruct-f16.gguf model." >&2
		exit 1
	fi

	d="$batch_size-$(date -Iseconds)"

	case {{lowercase(spec)}} in
		lros)
			if [ ! -f {{proot}}/../config/mat_kernel_size_$batch_size.json ]; then
				{{proot}}/mat_kernel_size_generator.py {{proot}}/../config/mat_kernel_size_$batch_size.json -b $batch_size
			fi
			cp -fl {{proot}}/../config/mat_kernel_size_$batch_size.json {{proot}}/../config/mat_kernel_size.json
			case {{lowercase(variant)}} in
				all | vaccel )
					if [ $log_power = true ]; then just -f {{source_directory()}}/power.just $powerlogger > {{OUT_DIR}}/bench_lros_vaccel_$d.power.txt & fi
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-vaccel_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_vaccel_$d.out.txt \
						{{OUT_DIR}}/bench_lros_vaccel_$d.err.txt \
						{{ cpu_type }} {{ size_mem }} {{ batch_size }}
					if [ $log_power = true ]; then pkill -P $!; fi
				;;&
				all | vaccel-novec )
					if [ $log_power = true ]; then just -f {{source_directory()}}/power.just $powerlogger > {{OUT_DIR}}/bench_lros_vaccel-novec_$d.power.txt & fi
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-vaccel-novec_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_vaccel-novec_$d.out.txt \
						{{OUT_DIR}}/bench_lros_vaccel-novec_$d.err.txt \
						{{ cpu_type }} {{ size_mem }} {{ batch_size }}
					if [ $log_power = true ]; then pkill -P $!; fi
				;;&
				all | cpu )
					if [ $log_power = true ]; then just -f {{source_directory()}}/power.just $powerlogger > {{OUT_DIR}}/bench_lros_cpu_$d.power.txt & fi
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-cpu_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_cpu_$d.out.txt \
						{{OUT_DIR}}/bench_lros_cpu_$d.err.txt \
						{{ cpu_type }} {{ size_mem }} {{ batch_size }}
					if [ $log_power = true ]; then pkill -P $!; fi
				;;&
				all | cpu-novec )
					if [ $log_power = true ]; then just -f {{source_directory()}}/power.just $powerlogger > {{OUT_DIR}}/bench_lros_cpu-novec_$d.power.txt & fi
					just -f {{source_file()}} run_lros \
						{{LROS_PATH}}/.unikraft/build/bench-cpu-novec_qemu-arm64 \
						{{OUT_DIR}}/bench_lros_cpu-novec_$d.out.txt \
						{{OUT_DIR}}/bench_lros_cpu-novec_$d.err.txt \
						{{ cpu_type }} {{ size_mem }} {{ batch_size }}
					if [ $log_power = true ]; then pkill -P $!; fi
				;;&
				all | vaccel | vaccel-novec | cpu | cpu-novec)
				;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >2
				;;
			esac
		;;

		linux-vm | vm)
			just -f {{source_file()}} run_vm {{OUT_DIR}}/bench_vm_$d.out.txt {{OUT_DIR}}/bench_vm_$d.err.txt {{size_mem}} {{cpu_type}}
		;;

		linux | process)
			case {{lowercase(variant)}} in
				all | cpu )
					taskset -c {{ if cpu_type == "big" { "4-4" } else { "0-0" } }} \
					{{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS}} {{batch_size}} \
						-m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
						2> >(tee {{OUT_DIR}}/bench_process_cpu_$d.err.txt >&2) \
						| tee {{OUT_DIR}}/bench_process_cpu_$d.out.txt
				;;&
				all | cuda )
					# CUDA
					if [ -d /proc/driver/nvidia/ ]; then
						taskset -c {{ if cpu_type == "big" { "4-4" } else { "0-0" } }} \
						{{proot}}/../lros/llama.cpp-rknn/build-cuda/bin/llama-batched-bench {{EVAL_ARGS}} {{batch_size}} \
							-m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
							2> >(tee {{OUT_DIR}}/bench_process_cuda_$d.err.txt >&2) \
							| tee {{OUT_DIR}}/bench_process_cuda_$d.out.txt
					elif [ {{lowercase(variant)}} -eq "cuda" ]; then
						echo "Cannot execute CUDA on this device. Could not find GPU driver." >2
						exit 1
					fi
				;;&
				all | rknn )
					# RKNN
					if [ -d /sys/devices/platform/fdab0000.npu ]; then
						taskset -c {{ if cpu_type == "big" { "4-4" } else { "0-0" } }} \
						{{proot}}/../lros/llama.cpp-rknn/build-rknn/bin/llama-batched-bench {{EVAL_ARGS}}  {{batch_size}} \
							-m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf \
							2> >(tee {{OUT_DIR}}/bench_process_rknn_$d.err.txt >&2) \
							| tee {{OUT_DIR}}/bench_process_rknn_$d.out.txt
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

paging cpu_type="big": (build "linux" "cpu") (build "lros" "cpu")
	#!/usr/bin/env bash

	set -ex

	mkdir -p {{OUT_DIR}}/paging

	d=$(date -Iseconds)

	# Shared arguments (16 prompt token, 8 text gen token, 1 batch, 1 thread, no detailed integrated timings)
	args="-npp 16 -ntg 8 -npl 1 -t 1 --no-perf"

	for m in "16G" "2G"; do
		# Linux VM
		just -f {{source_file()}} EVAL_ARGS="$args --no-mmap" run_vm \
			 {{OUT_DIR}}/paging/bench_paging_"$m"_read_"$d".out.txt \
			 {{OUT_DIR}}/paging/bench_paging_"$m"_read_"$d".err.txt \
			 $m {{cpu_type}}
		just -f {{source_file()}} EVAL_ARGS="$args --no-prefetch" run_vm \
			 {{OUT_DIR}}/paging/bench_paging_"$m"_mmap_"$d".out.txt \
			 {{OUT_DIR}}/paging/bench_paging_"$m"_mmap_"$d".err.txt \
			 $m {{cpu_type}}
		just -f {{source_file()}} EVAL_ARGS="$args --no-perf" run_vm \
			 {{OUT_DIR}}/paging/bench_paging_"$m"_prefetch_"$d".out.txt \
			 {{OUT_DIR}}/paging/bench_paging_"$m"_prefetch_"$d".err.txt \
			 $m {{cpu_type}}

		# LROS
		just -f {{source_file()}} EVAL_ARGS="$args --no-mmap" run_lros \
			{{LROS_PATH}}/.unikraft/build/bench-cpu_qemu-arm64 \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosread_"$d".out.txt \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosread_"$d".errout.txt \
			{{cpu_type}} $m
		# Fix merged output
		grep "llama_perf_context_print" {{OUT_DIR}}/paging/bench_paging_"$m"_lrosread_"$d".out.txt \
			> {{OUT_DIR}}/paging/bench_paging_"$m"_lrosread_"$d".err.txt || test $? = 1;

		just -f {{source_file()}} EVAL_ARGS="$args" run_lros \
			{{LROS_PATH}}/.unikraft/build/bench-cpu_qemu-arm64 \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch_"$d".out.txt \
			{{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch_"$d".errout.txt \
			{{cpu_type}} $m
		# Fix merged output
		grep "llama_perf_context_print" {{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch_"$d".out.txt \
			> {{OUT_DIR}}/paging/bench_paging_"$m"_lrosprefetch_"$d".err.txt || test $? = 1;
	done

perf_isolation: (build "linux" "cpu")
  #!/usr/bin/env bash
  set -ex
  mkdir -p {{OUT_DIR}}/perf_isolation
  echo "setup,type_interference,nb_thread_interference,time" > {{OUT_DIR}}/perf_isolation/perf_isolation.csv
  type_interference="cpu vm io"


  just vm 1 4G big > /dev/null 2> /dev/null &
  sleep 40
  
  echo -n "vm,none,0," >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
  sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
  just ssh_local "/root/scripts/run_in_vm.sh /root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" 2>&1 | grep "total time" | awk '{print $5}' >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
  
  for ti in $type_interference
  do 
    echo -n "vm,$ti,7," >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
    taskset -c 0-3,5-7 stress --$ti 7 > /dev/null 2> /dev/null &
    just ssh_local "/root/scripts/run_in_vm.sh /root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" 2>&1 | grep "total time" | awk '{print $5}' >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo kill -9 $(pidof stress)
    
    echo -n "vm,$ti,8," >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
    taskset -c 0-7 stress --$ti 8 > /dev/null 2> /dev/null &
    just ssh_local "/root/scripts/run_in_vm.sh /root/lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m /root/models/Llama-3.2-1B-Instruct-f16.gguf" 2>&1 | grep "total time" | awk '{print $5}' >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo kill -9 $(pidof stress)
  done

  just ssh_local "poweroff"
  sleep 40

  echo -n "process,none,0," >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
  sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
  taskset -c 4 {{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf 2>&1 | grep "total time" | awk '{print $5}' >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv

  for ti in $type_interference
  do    
    echo -n "process,$ti,7," >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
    taskset -c 0-3,5-7 stress --$ti 7 > /dev/null 2> /dev/null &
    taskset -c 4 {{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf 2>&1 | grep "total time" | awk '{print $5}' >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo kill -9 $(pidof stress)
    
    echo -n "process,$ti,8," >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
    taskset -c 0-7 stress --$ti 8 > /dev/null 2> /dev/null &
    taskset -c 4 {{proot}}/../lros/llama.cpp-rknn/build/bin/llama-batched-bench {{EVAL_ARGS_SINGLE}} -m {{proot}}/../models/Llama-3.2-1B-Instruct-f16.gguf 2>&1 | grep "total time" | awk '{print $5}' >> {{OUT_DIR}}/perf_isolation/perf_isolation.csv
    sudo kill -9 $(pidof stress)
  done
