# The machines an experiment runs on. Each keeps everything in a local scratch;
# nothing is shared with the orchestrator, which builds and collects.

# target -> ssh destination, empty for here.
declare -A TGT_HOST=(
    [local]=""
    [orangepi]="orangepi5ultra.dos.cit.tum.de"
    [joy]="joy"
)

# target -> a directory on local storage, not NFS.
declare -A TGT_SCRATCH=(
    [local]="${TMPDIR:-/tmp}/lros-exp"
    [orangepi]="/scratch/$USER/lros"
    [joy]="/scratch/$USER/lros"
)

declare -A TGT_PLATFORM=(
    [local]="unknown"
    [orangepi]="rk3588"
    [joy]="orin"
)

# Cores a measurement is pinned to: the big cores only on the RK3588.
declare -A TGT_CORES=(
    [local]="0-3"
    [orangepi]="4-7"
    [joy]="0-7"
)

declare -A TGT_MEM=(
    [local]="4G"
    [orangepi]="8G"
    [joy]="8G"
)

# One vcpu per pinned core.
declare -A TGT_VCPUS=(
    [local]="4"
    [orangepi]="4"
    [joy]="8"
)

# The host llama.cpp preset that uses the board's accelerator.
declare -A TGT_HOST_ACCEL=(
    [local]=""
    [orangepi]="rknnoh"
    [joy]="cuda"
)

# What a vAccel guest needs: the patched QEMU, the plugin and its libraries.
declare -A TGT_VACCEL_ENV=(
    [local]=""
    [orangepi]='
Q=$SCRATCH/qemu-vaccel
VA=/nix/store/np95wg8afv5xsa6acd1nr6ar6c8gwyk5-vaccel-0.7.1
RKNN=/nix/store/b1c98zp85m8chaw748xp68fiwsg867zc-librknnrt-unknown
[ -x "$Q/bin/qemu-system-aarch64" ] || Q=/nix/store/drpsk13l2ximn4j2qx5f5vyl9r6x745p-qemu-vaccel-10.1.50-vaccel
export QEMU_VACCEL=$Q/bin/qemu-system-aarch64
export VACCEL_PLUGINS=$SCRATCH/plugin/libvaccel-rknn.so
[ -e "$VACCEL_PLUGINS" ] || export VACCEL_PLUGINS=/nix/store/j1n1j87y0qf42hah0i81rhj37cldd8rg-vaccel-plugin-rknn-0.7.1/lib/libvaccel-rknn.so
export VACCEL_BACKENDS=$VACCEL_PLUGINS
export LD_LIBRARY_PATH=$VA/lib:$RKNN/lib
export AAVMF_CODE=$Q/share/qemu/edk2-aarch64-code.fd
'
    [joy]='
VA=/nix/store/np95wg8afv5xsa6acd1nr6ar6c8gwyk5-vaccel-0.7.1
CUDART=/nix/store/hihpfsyjh9h8c7sxp75n8qlqbs4w517n-cuda_cudart-12.6.77-lib/lib
CUBLAS=/nix/store/ypbj1is9jy373kflw6gqa9cik5zrfp2y-libcublas-12.6.4.1-lib/lib
export QEMU_VACCEL=$SCRATCH/qemu-vaccel/bin/qemu-system-aarch64
export VACCEL_PLUGINS=$SCRATCH/plugin/libvaccel-cuda.so
export VACCEL_BACKENDS=$VACCEL_PLUGINS
export LD_LIBRARY_PATH=/run/opengl-driver/lib:$CUBLAS:$CUDART:$VA/lib:$SCRATCH/plugin
export AAVMF_CODE=$SCRATCH/qemu-vaccel/share/qemu/edk2-aarch64-code.fd
'
)

# Models each target holds, by file name.
declare -A TGT_MODELS=(
    [local]="Llama-3.2-1B-Instruct-f16.gguf"
    [orangepi]="Llama-3.2-1B-Instruct-f16.gguf gemma-3-1b-it-Q4_1.gguf"
    [joy]="Llama-3.2-1B-Instruct-f16.gguf gemma-3-1b-it-Q4_1.gguf"
)

# The llama.cpp checkout the native host baselines are built from.
declare -A TGT_HOST_SRC=(
    [local]="$EXP_ROOT/llama.cpp-rknn"
    [orangepi]="/home/ilya/llama.cpp-rknn"
    [joy]="/home/ilya/miniosv_dev/lros-expe/llama.cpp-rknn"
)

# Where a target already keeps models; taken into the scratch on first setup,
# copied rather than moved from a network filesystem.
declare -A TGT_MODEL_SRC=(
    [local]="/home/ilya/lros/models"
    [orangepi]="/home/ilya/models"
    [joy]="/home/ilya/lros/models"
)

# A checkout of this repo on the target, for what only aarch64 can build.
declare -A TGT_REPO=(
    [local]="$EXP_ROOT"
    [orangepi]="/home/ilya/lros-expe"
    [joy]="/home/ilya/miniosv_dev/lros-expe"
)

# The aarch64 machine that builds nix artefacts for a target, empty for itself.
declare -A TGT_BUILDER=(
    [local]=""
    [orangepi]="eliza"
    [joy]="eliza"
)

NIX_CACHE="https://cache.dos.cit.tum.de"

# What a native build needs that the target lacks: joy has no cmake, and its
# CUDA must be 12.6 to match the JetPack driver.
declare -A TGT_BUILD_ENV=(
    [local]=""
    [orangepi]=""
    [joy]='export PATH=$(nix build --no-link --print-out-paths nixpkgs#cmake nixpkgs#gnumake nixpkgs#gcc | sed "s|\$|/bin|" | paste -sd:):$PATH
export CUDAToolkit_ROOT=$(nix build --no-link --print-out-paths nixpkgs#cudaPackages_12_6.cudatoolkit)
export PATH=$CUDAToolkit_ROOT/bin:$PATH
export NVCC_PREPEND_FLAGS="-I$CUDAToolkit_ROOT/include"
export CUDA_ARCH=87
'
)

# What a native accelerator run needs to find the real driver, not the stub.
declare -A TGT_HOST_ACCEL_ENV=(
    [local]=""
    [orangepi]=""
    [joy]='export LD_LIBRARY_PATH=/run/opengl-driver/lib:/nix/store/ypbj1is9jy373kflw6gqa9cik5zrfp2y-libcublas-12.6.4.1-lib/lib:/nix/store/hihpfsyjh9h8c7sxp75n8qlqbs4w517n-cuda_cudart-12.6.77-lib/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'
)

exp_targets() { echo "${!TGT_HOST[@]}"; }
