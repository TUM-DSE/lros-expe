# Builds the matmul-only plugins, which is what the flake packages. The viai
# graph server (viai_server.cpp, viai_rknn.cpp, ggml_probe.cpp) is not built
# here: it links a ggml built on the board itself, so it is compiled by
# scripts/exp (exp_accel_plugin) with -DVIAI_ADAPTER_RKNN instead.
cmake_minimum_required(VERSION 3.25)
project(plugins LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED COMPONENTS cudart cublas)

find_library(VACCEL vaccel)

# Add librknnrt here
add_library(matmulrknn SHARED rknn.c)
target_link_libraries(matmulrknn PUBLIC m ${VACCEL} ${CMAKE_SOURCE_DIR}/librknnrt.so)
target_compile_features(matmulrknn PRIVATE cxx_std_14)

add_library(matmulcuda SHARED cuda.cu)
target_link_libraries(matmulcuda PUBLIC CUDA::cublas CUDA::cudart m ${VACCEL})
target_compile_features(matmulcuda PRIVATE cxx_std_14)
target_compile_options(matmulcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                     --expt-relaxed-constexpr>)

install(TARGETS matmulcuda matmulrknn)
