// The RKNN adapter: executes a ggml graph's matmul nodes on the RK3588 NPU
// through librknnrt directly.
//
// The CUDA adapter hosts a real ggml backend and hands it the whole graph.
// That is not available here: ggml-rknnoh is a host-side backend and does not
// run inside the QEMU process. So this adapter speaks ggml on one side and the
// librknnrt matmul API on the other, which is the same thing rknnoh does, minus
// its scheduler and its global caches.
//
// The layout conversions are taken from ggml-rknnoh.cpp: the NPU wants operands
// blocked rather than row-major, and reports the block sizes in the io_attr
// that rknn_matmul_create fills in.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

int adapter_threads() {
    const char *e = getenv("VIAI_BACKEND_THREADS");
    const int n = e ? atoi(e) : 4;
    return n > 0 ? n : 4;
}

// F32 row-major -> F16 blocked by subK. From ggml-rknnoh.
void layout_A(const float *src, uint16_t *dst, int32_t M, int32_t K, int32_t subK) {
    const int outer = (K + subK - 1) / subK;
    for (int k1 = 0; k1 < outer; k1++) {
        const int base = k1 * M * subK;
        for (int m = 0; m < M; m++) {
            const int row = base + m * subK;
            const int srow = m * K;
            for (int j = 0; j < subK; j++) {
                const int ki = k1 * subK + j;
                dst[row + j] = ki < K ? ggml_fp32_to_fp16(src[srow + ki]) : 0;
            }
        }
    }
}

// Blocked F32 -> row-major F32. From ggml-rknnoh, single-threaded: the split is
// already one node here, and spawning threads per node cost more than it saved.
void layout_C(const float *src, float *dst, int32_t M, int32_t N, int32_t subN) {
    for (int32_t m = 0; m < M; m++) {
        float *drow = dst + (size_t) m * N;
        for (int32_t i = 0; i < N; i += subN) {
            const int32_t block = i / subN;
            const int32_t off = block * M * subN + m * subN;
            const int32_t n = std::min<int32_t>(subN, N - i);
            memcpy(drow + i, src + off, n * sizeof(float));
        }
    }
}

// Split the way ggml-rknnoh splits it, and for the same reason: a context and
// its A/C operands depend only on the shape, while B depends on the weight.
// Keying everything by (weight, shape) instead created one context per weight
// and the NPU ran out at 55 of them, mid-graph.
struct shape_slot {
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io = {};
    rknn_tensor_mem *A = nullptr;
    rknn_tensor_mem *C = nullptr;
};

struct weight_mem {
    rknn_tensor_mem *B = nullptr;
    bool loaded = false;
};

std::mutex slots_mutex;
std::unordered_map<uint64_t, shape_slot *> shapes;      // (M,K,N) -> ctx + A + C
std::unordered_map<const void *, weight_mem> weights;   // weight data -> B

uint64_t shape_key(int M, int K, int N) {
    return ((uint64_t) (uint32_t) M << 42) ^ ((uint64_t) (uint32_t) K << 21) ^ (uint32_t) N;
}

} // namespace

extern "C" bool viai_rknn_supports(const ggml_tensor *op) {
    if (op->op != GGML_OP_MUL_MAT) {
        // The no-ops ggml expects any backend to accept.
        return op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE ||
               op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE ||
               op->op == GGML_OP_TRANSPOSE;
    }
    const ggml_tensor *w = op->src[0];
    const ggml_tensor *a = op->src[1];
    if (!w || !a) {
        return false;
    }
    // F16 weights against F32 activations only, for now: the int8 path needs
    // per-row scales that this adapter does not carry yet.
    if (w->type != GGML_TYPE_F16 || a->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // 2D only. A batched matmul computed as 2D silently returns garbage.
    if (op->ne[2] > 1 || op->ne[3] > 1 || w->ne[2] > 1 || w->ne[3] > 1 ||
        a->ne[2] > 1 || a->ne[3] > 1) {
        return false;
    }
    // The documented RK3588 matmul limit.
    if (w->ne[0] > 4096 || w->ne[1] > 4096) {
        return false;
    }
    // And its alignment: ggml-rknnoh asserts K % 32 for F16 and pads N to 16.
    // Handing librknnrt an unaligned shape takes the process down without a
    // diagnostic, so refuse it here where the guest can still schedule the node
    // on the CPU instead.
    if (w->ne[0] % 32 != 0 || w->ne[1] % 16 != 0) {
        return false;
    }
    return ggml_is_contiguous(w) && ggml_is_contiguous(a);
}

extern "C" bool viai_rknn_node(ggml_tensor *node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return true;   // nothing to do for views and reshapes
    }
    const ggml_tensor *w = node->src[0];
    const ggml_tensor *a = node->src[1];

    const int K = (int) w->ne[0];
    const int N = (int) w->ne[1];
    const int M = (int) a->ne[1];


    shape_slot *slot = nullptr;
    weight_mem *wm = nullptr;
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        auto it = shapes.find(shape_key(M, K, N));
        if (it != shapes.end()) {
            slot = it->second;
        } else {
            slot = new shape_slot();
            rknn_matmul_info info = {};
            info.M = M;
            info.K = K;
            info.N = N;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
            // As ggml-rknnoh sets them: A and C in the performance layout, B in
            // the native one. B_layout = 0 asks for a different size and layout
            // for the weights than a straight copy provides.
            info.AC_layout = 1;
            info.B_layout = 1;
            const int rc = rknn_matmul_create(&slot->ctx, &info, &slot->io);
            if (rc != 0) {
                delete slot;
                return false;
            }
            slot->A = rknn_create_mem(slot->ctx, slot->io.A.size);
            slot->C = rknn_create_mem(slot->ctx, slot->io.C.size);
            if (!slot->A || !slot->C) {
                delete slot;
                return false;
            }
            shapes[shape_key(M, K, N)] = slot;
        }
        wm = &weights[w->data];
    }

    // The weights, once per tensor, staying in NPU memory for the session.
    if (!wm->loaded) {
        if (ggml_nbytes(w) != slot->io.B.size) {
            GGML_LOG_ERROR("viai_rknn: %s weights are %zu bytes, device wants %u\n",
                           node->name, ggml_nbytes(w), slot->io.B.size);
            return false;
        }
        wm->B = rknn_create_mem(slot->ctx, slot->io.B.size);
        if (!wm->B) {
            GGML_LOG_ERROR("viai_rknn: no device memory for %s weights\n", node->name);
            return false;
        }
        memcpy(wm->B->virt_addr, w->data, ggml_nbytes(w));
        wm->loaded = true;
    }
    // Bound every call: the context is shared, so another weight may have been
    // the last thing bound to it.
    int rcb;
    rcb = rknn_matmul_set_io_mem(slot->ctx, wm->B, &slot->io.B);
    if (rcb != 0) {
        return false;
    }

    const int32_t subK = (int32_t) slot->io.A.dims[2];
    const size_t need_A = (size_t) M * ((K + subK - 1) / subK) * subK * sizeof(uint16_t);
    if (need_A > slot->io.A.size) {
        GGML_LOG_ERROR("viai_rknn: %s needs %zu bytes of A, device gave %u\n",
                       node->name, need_A, slot->io.A.size);
        return false;
    }
    layout_A((const float *) a->data, (uint16_t *) slot->A->virt_addr, M, K, subK);
    int rca, rcc;
    rca = rknn_matmul_set_io_mem(slot->ctx, slot->A, &slot->io.A);
    rcc = rknn_matmul_set_io_mem(slot->ctx, slot->C, &slot->io.C);
    if (rca != 0 || rcc != 0) {
        return false;
    }
    int rcr;
    rcr = rknn_matmul_run(slot->ctx);
    if (rcr != 0) {
        return false;
    }
    const size_t need_C = (size_t) M * N * sizeof(float);
    if (need_C > ggml_nbytes(node)) {
        GGML_LOG_ERROR("viai_rknn: %s result is %zu bytes, tensor holds %zu\n",
                       node->name, need_C, ggml_nbytes(node));
        return false;
    }
    layout_C((const float *) slot->C->virt_addr, (float *) node->data,
                            M, N, slot->io.C.dims[2]);
    // Reported every 512 nodes: the shape of the per-node cost is what matters,
    // not any single one.
    return true;
}
