// Risk check for PLAN-viai section 8: the plugin is dlopened by QEMU's
// acceldev backend, so a hosted ggml-cuda would initialise inside the QEMU
// process. Raw CUDA and cuBLAS already work there, but ggml-cuda brings its own
// backend registry, device discovery and stream management.
//
// Builds a small matmul on the CUDA backend and checks the result, reporting
// what it found through a caller-supplied buffer.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int va_ggml_probe(char *out, unsigned int out_size)
{
    if (!out || out_size == 0) {
        return 1;
    }
#define report(...) snprintf(out, out_size, __VA_ARGS__)

    // Whatever backend this plugin drives: CUDA on the Orin, RKNN on the
    // RK3588. The same check that cleared ggml-cuda for use inside the QEMU
    // process has to be run for every backend hosted the same way.
    const char *name = getenv("VIAI_BACKEND");
    if (!name) {
        name = "CUDA";
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(name);
    if (!reg) {
        report("no '%s' backend registered", name);
        return 1;
    }
    // A backend that takes a thread count expects to be told; unset, RKNN
    // reads an uninitialised one.
    auto set_n_threads = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    const size_t ndev = ggml_backend_reg_dev_count(reg);
    if (ndev == 0) {
        report("'%s' registered but reports 0 devices", name);
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        report("ggml_backend_dev_init failed for %s", ggml_backend_dev_name(dev));
        return 1;
    }
    if (set_n_threads) {
        set_n_threads(backend, 4);
    }

    const int K = 4, M = 4, N = 4;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
                            nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor *c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        report("ggml_backend_alloc_ctx_tensors failed on %s",
               ggml_backend_dev_name(dev));
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> av(K * M, 1.0f), bv(K * N, 2.0f), cv(M * N, 0.0f);
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        report("graph_compute returned %d on %s", (int) st,
               ggml_backend_dev_name(dev));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_tensor_get(c, cv.data(), 0, cv.size() * sizeof(float));

    // Every entry is a dot product of K ones against K twos.
    const float want = 2.0f * K;
    int bad = 0;
    for (float v : cv) {
        if (v != want) { bad++; }
    }

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    report("ok dev=%s devs=%zu c[0]=%.1f want=%.1f bad=%d/%zu mem=%zu/%zu MiB",
           ggml_backend_dev_name(dev), ndev, cv[0], want, bad, cv.size(),
           free_mem >> 20, total_mem >> 20);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return bad ? 1 : 0;
}
