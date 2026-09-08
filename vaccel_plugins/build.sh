#!/usr/bin/env bash
# Builds the plugin on the board you are logged into, without the experiment
# harness: the same build-plugin.sh that `just exp::accel-plugin` drives over
# ssh, but with the paths and toolchain carried here rather than read from
# scripts/exp/targets.sh.
#
#   vaccel_plugins/build.sh [rk3588|orin]     platform, else autodetected
#
# Writes $SCRATCH/plugin; point VACCEL_PLUGINS and VACCEL_BACKENDS at the .so to
# use it. Every path below can be overridden from the environment.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD="$REPO/vaccel_plugins/build-plugin.sh"

PLATFORM="${1:-}"
if [ -z "$PLATFORM" ]; then
    compat=$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null || true)
    case "$compat" in
        *rk3588*)  PLATFORM=rk3588 ;;
        *tegra*)   PLATFORM=orin ;;
        *) echo "cannot tell what board this is; pass rk3588 or orin" >&2; exit 1 ;;
    esac
fi

# The store paths the boards happen to have. When one moves it moves in
# scripts/exp/targets.sh too, which is what the harness reads.
case "$PLATFORM" in
    rk3588)
        SCRATCH="${SCRATCH:-/var/tmp/lros}"
        export VACCEL_PREFIX="${VACCEL_PREFIX:-/nix/store/np95wg8afv5xsa6acd1nr6ar6c8gwyk5-vaccel-0.7.1}"
        export RKNN="${RKNN:-/nix/store/b1c98zp85m8chaw748xp68fiwsg867zc-librknnrt-unknown}"
        ;;
    orin)
        SCRATCH="${SCRATCH:-/scratch/$USER/lros}"
        export CUDART="${CUDART:-/nix/store/hihpfsyjh9h8c7sxp75n8qlqbs4w517n-cuda_cudart-12.6.77-lib/lib}"
        export CUBLAS="${CUBLAS:-/nix/store/ypbj1is9jy373kflw6gqa9cik5zrfp2y-libcublas-12.6.4.1-lib/lib}"
        # 12.6 headers to match those libs, and cccl because cuda_fp16.h
        # includes <nv/target>, which ships there and not in cudart.
        export CUDA_DEV_INCLUDE="${CUDA_DEV_INCLUDE:-\
-I/nix/store/q3544ab35knrqvfkhqs2ffnnkf9g3dwh-cuda_cudart-12.6.77-dev/include \
-I/nix/store/ga1mp9dg4s9fc7ryfb5ii2lqk2b27i81-libcublas-12.6.4.1-dev/include \
-I/nix/store/zqksvizqb6cvda6wl20kc58vjhafpp9w-cuda_cccl-12.6.77-dev/include}"
        export VACCEL_PREFIX="${VACCEL_PREFIX:-$(readlink -f "$SCRATCH/vaccel")}"
        ;;
    *) echo "no plugin for platform '$PLATFORM'" >&2; exit 1 ;;
esac
export PLATFORM REPO SCRATCH

# Each stage in a subshell, so the build toolchain does not reach the link: on
# the Orin it puts a gcc newer than nvcc accepts on PATH, and the link then
# fails a host_config.h version check.
(
    if [ "$PLATFORM" = orin ]; then
        # joy is NixOS and has no cmake; 12.6 and not the nixpkgs default,
        # because the board driver is the JetPack one and a binary built
        # against a newer toolkit fails at load.
        PATH="$(nix build --no-link --print-out-paths \
                nixpkgs#cmake nixpkgs#gnumake nixpkgs#gcc \
                | sed 's|$|/bin|' | paste -sd:):$PATH"
        CUDAToolkit_ROOT=$(nix build --no-link --print-out-paths \
                           nixpkgs#cudaPackages_12_6.cudatoolkit)
        # nvcc resolves its own symlink and looks for headers beside the real
        # binary, which in the joined toolkit has no cuda_runtime.h.
        export PATH="$CUDAToolkit_ROOT/bin:$PATH" CUDAToolkit_ROOT
        export NVCC_PREPEND_FLAGS="-I$CUDAToolkit_ROOT/include"
        export CUDA_ARCH="${CUDA_ARCH:-87}"
    fi
    "$BUILD" ggml
)
( "$BUILD" link )
