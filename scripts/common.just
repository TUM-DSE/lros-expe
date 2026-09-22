# Paths and defaults shared by the build, the demo and the VM recipes.
import "run.just"

LROS_PATH := proot + "/miniosv"
LROS_IMAGE_NAME := "loader.img"
LLAMACPP_RKNN_PATH := proot + "/llama.cpp-rknn"
LLAMA_CPP_TARGETS := "llama-batched-bench"
OUT_DIR := proot + "/bench/out"
parallel_img := proot + "/lros/.unikraft/build-parallel/parallel_qemu-arm64"

MODEL := proot + "/models/Llama-3.2-1B-Instruct-f16.gguf"
LLAMACPP_BIN := LLAMACPP_RKNN_PATH + "/build-cpu/bin/llama-batched-bench"
EVAL_ARGS := "-npp 16,32 -ntg 1,128,256 -t 1 -ngl 0 --no-mmap --no-perf"
EVAL_ARGS_SINGLE := "-npp 16 -ntg 128 -npl 1 -t 1 -ngl 0 --no-mmap --no-perf"

cpu_type := "big"
AFFINITY := if cpu_type == "big" { "4-4" } else { "0-0" }

_plugins := if path_exists("/sys/devices/platform/fdab0000.npu") == "true" { env("VACCEL_PLUGINS_RKNN", "") } else if path_exists("/proc/driver/nvidia/") == "true" { env("VACCEL_PLUGINS_CUDA", "") } else {""}
export VACCEL_PLUGINS := _plugins
