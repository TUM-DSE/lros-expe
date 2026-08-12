# Builds the three configurations. `lros` is the unikernel (miniOSv with
# llama.cpp linked in); `linux`/`process`/`vm` are the native llama.cpp used as
# the baseline, built from the llama.cpp-rknn checkout with its own presets.
#
# Not standalone: relies on LROS_PATH, LLAMACPP_RKNN_PATH and LLAMA_CPP_TARGETS
# from the importing justfile.

build spec variant:
	#!/usr/bin/env bash
	set -e

	case {{lowercase(spec)}} in
		lros)
			# miniOSv has no separate frontend builds: llama-cli and
			# batched-bench both live in every image and are chosen by the
			# first word of the boot arguments. So the variant axis is only
			# the two config knobs below.
			case {{lowercase(variant)}} in
				cpu | cpu-novec | vaccel | vaccel-novec )
					variants={{lowercase(variant)}} ;;
				all )
					variants="cpu cpu-novec vaccel vaccel-novec" ;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					variants="" ;;
			esac

			for v in $variants; do
				# conf_vaccel: link the vAccel backend (the accelerator path).
				# conf_llama_sgemm: llamafile's vectorised sgemm kernels, which
				# is what -novec turns off.
				case $v in
					cpu*) conf_vaccel=0 ;;
					*)    conf_vaccel=1 ;;
				esac
				case $v in
					*-novec) conf_sgemm=0 ;;
					*)       conf_sgemm=1 ;;
				esac

				# Each variant gets its own $(out), which the miniOSv Makefile
				# documents as the way to build variants side by side. It is not
				# just convenience: changing a -D flag in a .mk file does not
				# invalidate the objects already built with the old one, so
				# sharing one output directory across variants would silently
				# link, say, the CPU image against vAccel-compiled objects.
				#
				# `nix develop` on the submodule, not this repo's shell: miniOSv
				# builds with a self-contained LLVM toolchain (clang/lld/
				# llvm-libc/libc++) that must not be mixed with the gcc13 stdenv
				# the native baselines use.
				nix develop {{LROS_PATH}} --command \
					make -C {{LROS_PATH}} -j"$(nproc)" \
						app=llama arch=aarch64 \
						out=build/$v.aarch64 \
						conf_vaccel=$conf_vaccel \
						conf_llama_sgemm=$conf_sgemm
			done
		;;

		linux-vm | vm | linux | process)
			[ -x "$(command -v cmake)" ] && cd {{LLAMACPP_RKNN_PATH}}
			case {{lowercase(variant)}} in
				cpu | cpu-novec | cuda | rknn | rknnoh | cpu-thp )
					variants={{lowercase(variant)}} ;;
				all )
					variants="cpu cpu-novec cpu-thp cuda rknn rknnoh" ;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					variants="" ;;
			esac

			for v in $variants; do
				# For `all`, silently skip CUDA when nvcc is absent;
				# for an explicit request, report the missing toolchain.
				if [ "$v" = cuda ] && ! [ -x "$(command -v nvcc)" ]; then
					[ {{lowercase(variant)}} = all ] && continue
					echo "Cannot build CUDA. Could not find nvcc." >&2
					exit 1
				fi
				# The rknn/rknnoh backends read their kernel-shape list from
				# CONFIG_DIR, which model_related_config.h.in fixes to
				# <source>/scripts. Generate it if it is not there yet.
				case $v in
					rknn | rknnoh )
						[ -f {{LLAMACPP_RKNN_PATH}}/scripts/mat_kernel_size.json ] || \
							{{proot}}/scripts/mat_kernel_size_generator.py \
								{{LLAMACPP_RKNN_PATH}}/scripts/mat_kernel_size.json -b 1
					;;
				esac
				# Only run setup if build files don't exist already
				[ -f {{LLAMACPP_RKNN_PATH}}/build-$v/CMakeCache.txt ] || cmake --preset "$v"
				cmake --build --preset "$v" -t {{LLAMA_CPP_TARGETS}} -j"$(nproc)"
			done
		;;

		* )
			echo "Unknown spec {{spec}}" >&2
		;;
	esac
