#include "vaccel.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

#define debug(fmt, ...) vaccel_debug("[cuda] " fmt, ##__VA_ARGS__)
#define error(fmt, ...) vaccel_error("[cuda] " fmt, ##__VA_ARGS__)

static inline bool check(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        error("CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        return true;
    }
    return false;
}

#define CHECK(x, y) if (check(x, #x)) return y;

static inline bool check_blas(cublasStatus_t stat, const char *context) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        error("CUBlas error at %s: %s\n", context, cublasGetStatusString(stat));
        return true;
    }
    return false;
}

#define CHECK_BLAS(x, y) if (check_blas(x, #x)) return y;

struct cuda_matmul_info {
    vaccel_matmul_info v_info;
    vaccel_tensor_mem_handle *A;
    vaccel_tensor_mem_handle *B;
    vaccel_tensor_mem_handle *C;
};

//FIXME: Not thread-safe
static cuda_matmul_info infos[256];
static uint32_t next_ctx = 0;

static cublasHandle_t cublas_handle;


static int va_cuda_matmul_create(struct vaccel_session *sess, vaccel_matmul_ctx *ctx,
                                 vaccel_matmul_info *info,
                                 vaccel_matmul_io_attr *io_attr) {
    uint32_t c = next_ctx++;
    if (c >= 256) {
        return VACCEL_ENOSPC; //Appropriate error code ?? ENOMEM?
    }
    infos[c].v_info = *info;
    *ctx = c;
    auto M = (uint32_t) info->M;
    auto N = (uint32_t) info->N;
    auto K = (uint32_t) info->K;
    *io_attr = {
        .A = {"A", 2, {K, M}, K * M * 2, VACCEL_TENSOR_FLOAT16},
        .B = {"B", 2, {N, K}, N * K * 2, VACCEL_TENSOR_FLOAT16},
        .C = {"C", 2, {N, M}, N * M * 4, VACCEL_TENSOR_FLOAT32},
    };

    if (info->AC_layout != 0 || info->B_layout != 0) {
        error("cuda backend does not support native matrix layouts.\n");
        return VACCEL_EINVAL;
    }

    return VACCEL_OK;
}

static int va_cuda_create_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx,
                              uint32_t size, vaccel_tensor_mem *result) {
    vaccel_tensor_mem_handle *handle;

    cudaError_t err = cudaMalloc(&handle, size);


    if (err != cudaSuccess) {
        return VACCEL_ENOMEM;
    }

    result->virt_addr = handle;
    result->handle = handle;
    return VACCEL_OK;
}

static int va_cuda_destroy_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx,
                               vaccel_tensor_mem *mem) {
    CHECK(cudaFree(mem->handle), VACCEL_EINVAL);
    return VACCEL_OK;
}

static int va_cuda_matmul_destroy(struct vaccel_session *sess, vaccel_matmul_ctx ctx) {
    //TODO Actually free allowing reuse potentially
    //infos[ctx].v_info.type = VACCEL_MATMUL_NOTYPE;
    return VACCEL_OK;
}

static int va_cuda_matmul_set_io_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx,
                                     vaccel_tensor_mem_handle *mem,
                                     vaccel_matmul_tensor_attr *attr) {
    if (attr->name[1] != 0) {
        return VACCEL_EINVAL;
    }
    switch (attr->name[0]) {
        case 'A':
            infos[ctx].A = mem;
            break;
        case 'B':
            infos[ctx].B = mem;
            break;
        case 'C':
            infos[ctx].C = mem;
            break;
        default:
            return VACCEL_EINVAL;
    }
    return VACCEL_OK;
}

static int va_cuda_matmul_set_core_mask(struct vaccel_session *sess,
                                        vaccel_matmul_ctx ctx,
                                        vaccel_core_mask core_mask) {
    debug("set_core_mask not implemented/available for CUDA.");
    return VACCEL_OK;
}

static int va_cuda_matmul_run(struct vaccel_session *sess, vaccel_matmul_ctx ctx) {
    auto &info = infos[ctx];

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_BLAS(cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   info.v_info.N, info.v_info.M, info.v_info.K,
                   &alpha, info.B, CUDA_R_16F, info.v_info.K,
                   info.A, CUDA_R_16F, info.v_info.K,
                   &beta, info.C, CUDA_R_32F, info.v_info.N,
                   CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP), VACCEL_EINVAL);

    return VACCEL_OK;
}

static int va_cuda_matmul_set_matrix(struct vaccel_session *sess,
                                     vaccel_tensor_mem_handle *dst, void *src,
                                     size_t nbytes) {
    CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice), VACCEL_ENOMEM);
    return VACCEL_OK;
}

static int va_cuda_matmul_get_matrix(struct vaccel_session *sess, void *dst,
                                     vaccel_tensor_mem_handle *src, size_t nbytes) {
    CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost), VACCEL_ENOMEM);
    return VACCEL_OK;
}

static int va_cuda_matmul_get_props(struct vaccel_session *sess, char *props, size_t nbytes) {
    if (nbytes == 0) {
        return VACCEL_EINVAL;
    }

    const int maxProps = 1 + 1;
    int nprops = nbytes > maxProps ? maxProps : nbytes;

    switch (nprops) {
        case 2:
            props[1] = 0; // Prefer matrix transforms: 0:no !0:yes
        case 1:
            props[0] = nprops - maxProps;
    }

    return VACCEL_OK;
}


struct vaccel_op ops[] = {
    VACCEL_OP_INIT(ops[0], VACCEL_OP_MATMUL_CREATE, (void*) va_cuda_matmul_create),
    VACCEL_OP_INIT(ops[1], VACCEL_OP_CREATE_MEM, (void*) va_cuda_create_mem),
    VACCEL_OP_INIT(ops[2], VACCEL_OP_DESTROY_MEM, (void*) va_cuda_destroy_mem),
    VACCEL_OP_INIT(ops[3], VACCEL_OP_MATMUL_DESTROY, (void*) va_cuda_matmul_destroy),
    VACCEL_OP_INIT(ops[4], VACCEL_OP_MATMUL_SET_IO, (void*) va_cuda_matmul_set_io_mem),
    VACCEL_OP_INIT(ops[5], VACCEL_OP_MATMUL_SET_CORE_MASK, (void*) va_cuda_matmul_set_core_mask),
    VACCEL_OP_INIT(ops[6], VACCEL_OP_MATMUL_RUN, (void*) va_cuda_matmul_run),
    VACCEL_OP_INIT(ops[7], VACCEL_OP_MATMUL_SET_MATRIX, (void*) va_cuda_matmul_set_matrix),
    VACCEL_OP_INIT(ops[8], VACCEL_OP_MATMUL_GET_MATRIX, (void*) va_cuda_matmul_get_matrix),
    VACCEL_OP_INIT(ops[9], VACCEL_OP_MATMUL_GET_PROPS, (void*) va_cuda_matmul_get_props),
};

static int init(void) {
    CHECK_BLAS(cublasCreate(&cublas_handle), VACCEL_ENOTSUP);
    CHECK_BLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH), VACCEL_ENOTSUP);
    return vaccel_plugin_register_ops(ops, sizeof(ops) / sizeof(ops[0]));
}

static int fini(void) {
    return VACCEL_OK;
}

VACCEL_PLUGIN(.name = "cuda", .version = VACCEL_VERSION,
              .vaccel_version = VACCEL_VERSION,
              .init = init, .fini = fini, .type = VACCEL_PLUGIN_GPU)

#ifdef __cplusplus
}
#endif
