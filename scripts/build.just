build spec variant:
	#!/usr/bin/env bash

	set -e

	case {{lowercase(spec)}} in
		lros)
			cd {{LROS_PATH}}
			case {{lowercase(variant)}} in
				all | vaccel )
					kraft build -t bench-vaccel --no-configure -o .unikraft/build-vaccel
				;;&
				all | vaccel-novec )
					kraft build -t bench-vaccel-novec --no-configure -o .unikraft/build-vaccel-novec
				;;&
				all | vacceloh )
					kraft build -t bench-vacceloh --no-configure -o .unikraft/build-vacceloh
				;;&
				all | vacceloh-novec )
					kraft build -t bench-vacceloh-novec --no-configure -o .unikraft/build-vacceloh-novec
				;;&
				all | cpu )
					kraft build -t bench-cpu --no-configure -o .unikraft/build-cpu
				;;&
				all | cpu-novec )
					kraft build -t bench-cpu-novec --no-configure -o .unikraft/build-cpu-novec
				;;&
				all | parallel )
					kraft build -t parallel --no-configure -o .unikraft/build-parallel
				;;&
				all | vaccel | vaccel-novec | vacceloh | vacceloh-novec | cpu | cpu-novec | parallel )
				;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
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
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build/CMakeCache.txt ] || cmake -B build . -DGGML_CUDA=OFF -DGGML_RKNN=OFF -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
						cmake --build build -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
					;;&
					all | cpu-novec )
						# CPU-only
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build-novec/CMakeCache.txt ] || cmake -B build-novec . -DGGML_CUDA=OFF -DGGML_RKNN=OFF -DGGML_LLAMAFILE=OFF -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
						cmake --build build-novec -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
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
						elif [ {{lowercase(variant)}} == "cuda" ]; then
							echo "Cannot build CUDA. Could not find nvcc." >&2
							exit 1
						fi
					;;&
					all | rknn )
						# RKNN
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build-rknn/CMakeCache.txt ] || cmake -B build-rknn . -DGGML_CUDA=OFF -DGGML_RKNN=ON -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
						cmake --build build-rknn -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
					;;&
					all | rknnoh )
						# RKNN-OH
						[ -f {{LROS_PATH}}/llama.cpp-rknn/build-rknnoh/CMakeCache.txt ] || cmake -B build-rknnoh . -DGGML_CUDA=OFF -DGGML_RKNN=OFF -DGGML_RKNNOH=ON -DGGML_VACCEL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
						cmake --build build-rknnoh -t {{LLAMA_CPP_TARGETS}} -- -j $(nproc)
					;;&
					all | cpu | cpu-novec | cuda | rknn | rknnoh)
					;;
					*)
						echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					;;
				esac
			fi
		;;

		* )
			echo "Unknown spec {{spec}}" >&2
		;;
	esac
