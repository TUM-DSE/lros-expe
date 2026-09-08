#!/usr/bin/env bash
# Builds the plugin QEMU's acceldev backend dlopens, on the board it will run
# on: it links against that board's vAccel, its accelerator runtime (librknnrt,
# CUDA) and a ggml built here from the guest's own llama.cpp tree.
#
# That tree and not llama.cpp-rknn: the VIAI wire format embeds raw ggml_type
# and ggml_op values, so the two sides have to come from the same source.
#
# Runs on the target, out of the checkout there. lib.sh's exp_accel_plugin
# ships this directory and calls it; it is equally runnable by hand on a board,
# which is the point of it being a script rather than a quoted string:
#
#   PLATFORM=rk3588 REPO=$PWD SCRATCH=/var/tmp/lros \
#   VACCEL_PREFIX=/nix/store/...-vaccel-0.7.1 RKNN=/nix/store/...-librknnrt \
#       vaccel_plugins/build-plugin.sh
#
# Inputs, all from the environment:
#   PLATFORM        rk3588 | orin
#   REPO            this checkout on the target
#   SCRATCH         the plugin is written to $SCRATCH/plugin
#   VACCEL_PREFIX   vAccel install, or VA as targets.sh names it
#   RKNN            librknnrt install                          (rk3588)
#   CUDART CUBLAS CUDA_DEV_INCLUDE CUDA_ARCH                   (orin)
#   NIX_CACHE_OPT   extra nix options for the nvcc shell, may be empty
set -euo pipefail

: "${PLATFORM:?set PLATFORM to rk3588 or orin}"
: "${REPO:?set REPO to the checkout on this machine}"
: "${SCRATCH:?set SCRATCH to the scratch directory on this machine}"

VACCEL_PREFIX="${VACCEL_PREFIX:-${VA:-}}"
[ -n "$VACCEL_PREFIX" ] || { echo "no vAccel install: set VACCEL_PREFIX or VA" >&2; exit 1; }

GG="$REPO/miniosv/app/llama.cpp"     # the guest's tree, deliberately
GGB="$GG/build-plugin"               # ggml for the plugin, kept apart from the guest build
PLUGINS="$REPO/vaccel_plugins"
OUT="$SCRATCH/plugin"

say() { printf '\033[1m-->\033[0m %s\n' "$*" >&2; }

# The plugin links -lggml, so this is a dependency and not a convenience. It is
# incremental: a no-op rebuild costs seconds, which is worth paying to keep the
# .so and the ggml it was compiled against from drifting apart.
build_ggml() {
    local flags target
    case "$PLATFORM" in
        orin)
            flags="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native}"
            target="ggml-cuda" ;;
        rk3588)
            # No OpenMP: the plugin drives the accelerator and the guest keeps
            # the CPU work, so libgomp would be a runtime dependency that buys
            # nothing -- and QEMU is a nix binary that will not find the board's
            # system libgomp, so the plugin failed to dlopen with it linked in.
            #
            # No accelerator backend either: ggml-rknnoh is host-only and does
            # not run under QEMU. The plugin needs ggml for the graph and its
            # buffers; the NPU is driven by viai_rknn.cpp through librknnrt.
            flags="-DGGML_OPENMP=OFF"
            target="ggml-cpu" ;;
        *)
            echo "no plugin ggml backend for platform '$PLATFORM'" >&2; exit 1 ;;
    esac

    # A cache left by an interrupted configure has no build system under it.
    if [ -f "$GGB/CMakeCache.txt" ] && [ ! -f "$GGB/Makefile" ]; then
        rm -rf "$GGB"
    fi
    if [ ! -f "$GGB/CMakeCache.txt" ]; then
        say "configure ggml for the plugin ($target)"
        # shellcheck disable=SC2086
        cmake -S "$GG" -B "$GGB" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
            $flags \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
    fi
    say "build ggml ($target)"
    cmake --build "$GGB" -j"$(nproc)" -t ggml ggml-base "$target"
}

# Compiled here rather than taken from the .#vaccel-plugin-rknn derivation,
# which builds rknn.c alone: the plugin now hosts the VIAI server too, and that
# links the ggml built above.
link_rk3588() {
    : "${RKNN:?set RKNN to the librknnrt install}"
    cc -c -fPIC -O2 -o "$GGB/va-rknn.o" "$PLUGINS/rknn.c" \
        -I "$PLUGINS" -I "$VACCEL_PREFIX/include"
    c++ -shared -fPIC -O2 -DVIAI_ADAPTER_RKNN \
        -o "$OUT/libvaccel-rknn.so" \
        "$GGB/va-rknn.o" \
        "$PLUGINS/viai_server.cpp" "$PLUGINS/viai_rknn.cpp" "$PLUGINS/ggml_probe.cpp" \
        -I "$PLUGINS" -I "$VACCEL_PREFIX/include" \
        -I "$GG/ggml/include" -I "$GG/ggml/src" -I "$GG/ggml/src/ggml-viai" \
        -L "$VACCEL_PREFIX/lib" -L "$RKNN/lib" -L "$GGB/bin" \
        -Wl,-rpath,"$GGB/bin" -Wl,-rpath,"$RKNN/lib" \
        -lvaccel -lrknnrt -lggml -lggml-base -lggml-cpu
}

# No CUDA plugin derivation exists, and one could not: it links against the
# board's own CUDA driver, so it cannot come from a builder.
link_orin() {
    : "${CUDART:?set CUDART to the cuda_cudart lib directory}"
    : "${CUBLAS:?set CUBLAS to the libcublas lib directory}"
    # shellcheck disable=SC2086
    nix shell nixpkgs#cudaPackages.cuda_nvcc nixpkgs#cudaPackages.cuda_cudart \
              nixpkgs#cudaPackages.libcublas ${NIX_CACHE_OPT:-} --command \
      nvcc -shared -Xcompiler -fPIC -O2 --expt-relaxed-constexpr \
           --cudart shared --cudadevrt none \
           -o "$OUT/libvaccel-cuda.so" \
           "$PLUGINS/cuda.cu" "$PLUGINS/ggml_probe.cpp" "$PLUGINS/viai_server.cpp" \
           -I "$PLUGINS" -I "$VACCEL_PREFIX/include" ${CUDA_DEV_INCLUDE:-} \
           -I "$GG/ggml/include" -I "$GG/ggml/src" -I "$GG/ggml/src/ggml-viai" \
           -L "$VACCEL_PREFIX/lib" -L "$CUDART" -L "$CUBLAS" -L "$GGB/bin" \
           -Xlinker -rpath -Xlinker "$GGB/bin" \
           -lvaccel -lcublas -lcudart -lggml -lggml-base
}

mkdir -p "$OUT"
build_ggml
say "link the plugin for $PLATFORM"
case "$PLATFORM" in
    rk3588) link_rk3588 ;;
    orin)   link_orin ;;
    *)      echo "no accelerator plugin for platform '$PLATFORM'" >&2; exit 1 ;;
esac
say "plugin in $OUT"
