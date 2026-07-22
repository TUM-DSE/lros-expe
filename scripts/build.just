build spec variant:
	#!/usr/bin/env bash
	set -e

	case {{lowercase(spec)}} in
		lros)
			cd {{LROS_PATH}}
			case {{lowercase(variant)}} in
				vaccel | vaccel-novec | vacceloh | vacceloh-novec | cpu | cpu-novec | parallel )
					variants={{lowercase(variant)}} ;;
				all )
					variants="vaccel vaccel-novec vacceloh vacceloh-novec cpu cpu-novec parallel" ;;
				*)
					echo "Unknown variant {{variant}} for spec {{spec}}" >&2
					variants="" ;;
			esac

			for v in $variants; do
				kraft build -t bench-$v --no-configure -o .unikraft/build-$v
			done
		;;

		linux-vm | vm | linux | process)
			[ -x "$(command -v cmake)" ] && cd {{LROS_PATH}}/llama.cpp-rknn
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
				# Only run setup if build files don't exist already
				[ -f {{LROS_PATH}}/llama.cpp-rknn/build-$v/CMakeCache.txt ] || cmake --preset "$v"
				cmake --build --preset "$v" -t {{LLAMA_CPP_TARGETS}} -j"$(nproc)"
			done
		;;

		* )
			echo "Unknown spec {{spec}}" >&2
		;;
	esac