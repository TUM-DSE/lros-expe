// Checks the RKNN adapter's matmul against ggml's CPU backend on random data.
//
//   rknn_matmul_check     prints the largest relative error per case
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

extern "C" bool viai_rknn_supports(const ggml_tensor *op);
extern "C" bool viai_rknn_node(ggml_tensor *node);

static bool check(ggml_type wtype, int K, int N, int M) {
    ggml_init_params p = {(size_t) 512 << 20, nullptr, false};
    ggml_context *ctx = ggml_init(p);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> wf((size_t) K * N);
    for (float &v : wf) v = dist(rng) * 0.05f;
    ggml_tensor *w = ggml_new_tensor_2d(ctx, wtype, K, N);
    if (wtype == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t *) w->data, (int64_t) K * N);
    } else {
        ggml_quantize_chunk(wtype, wf.data(), w->data, 0, N, K, nullptr);
    }
    ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    for (int i = 0; i < K * M; i++) ((float *) a->data)[i] = dist(rng);

    ggml_tensor *ref = ggml_mul_mat(ctx, w, a);
    ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, ref);
    ggml_graph_compute_with_ctx(ctx, g, 4);

    ggml_tensor *npu = ggml_mul_mat(ctx, w, a);
    if (!viai_rknn_supports(npu)) {
        printf("%-5s K=%d N=%d M=%d: not supported\n", ggml_type_name(wtype), K, N, M);
        ggml_free(ctx);
        return false;
    }
    if (!viai_rknn_node(npu)) {
        printf("%-5s K=%d N=%d M=%d: node failed\n", ggml_type_name(wtype), K, N, M);
        ggml_free(ctx);
        return false;
    }
    double err = 0, norm = 0;
    for (int i = 0; i < N * M; i++) {
        const double r = ((float *) ref->data)[i], x = ((float *) npu->data)[i];
        err = std::max(err, std::fabs(x - r));
        norm = std::max(norm, std::fabs(r));
    }
    printf("%-5s K=%d N=%d M=%d: max error %.4f of max %.4f (%.2f%%)\n",
           ggml_type_name(wtype), K, N, M, err, norm, 100 * err / norm);
    ggml_free(ctx);
    return err / norm < 0.05;
}

int main() {
    bool ok = true;
    for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (int M : {1, 8}) {
            ok &= check(t, 2048, 2048, M);
        }
    }
    ok &= check(GGML_TYPE_F16, 4096, 2048, 4);
    for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (int M : {1, 7}) {
            ok &= check(t, 2048, 8192, M);
            ok &= check(t, 8192, 2048, M);
        }
    }
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
