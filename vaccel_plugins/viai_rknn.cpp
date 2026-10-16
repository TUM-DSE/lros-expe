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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
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

// F32 row-major -> INT8 blocked by subK, one scale per row.
void layout_A_q8(const float *src, int8_t *dst, float *scale, int32_t M, int32_t K, int32_t subK) {
    const int outer = (K + subK - 1) / subK;
    for (int m = 0; m < M; m++) {
        const float *row = src + (size_t) m * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            amax = std::max(amax, std::fabs(row[k]));
        }
        scale[m] = amax / 127.0f;
        const float inv = scale[m] ? 1.0f / scale[m] : 0.0f;
        for (int k1 = 0; k1 < outer; k1++) {
            int8_t *out = dst + ((size_t) k1 * M + m) * subK;
            for (int j = 0; j < subK; j++) {
                const int ki = k1 * subK + j;
                out[j] = ki < K ? (int8_t) std::lround(row[ki] * inv) : 0;
            }
        }
    }
}

// The NPU reads B as [N/subN][K/subK][subN][subK]; ggml stores a weight as N
// rows of K. From ggml-rknnoh.
template <typename T>
void layout_B(const T *src, T *dst, int32_t K, int32_t N, int32_t subN, int32_t subK) {
    const int Nb = (N + subN - 1) / subN, Kb = (K + subK - 1) / subK;
    for (int i = 0; i < Nb; i++) {
        for (int j = 0; j < Kb; j++) {
            for (int n = 0; n < subN; n++) {
                T *out = dst + (((size_t) i * Kb + j) * subN + n) * subK;
                const int ni = i * subN + n;
                for (int k = 0; k < subK; k++) {
                    const int ki = j * subK + k;
                    out[k] = ni < N && ki < K ? src[(size_t) ni * K + ki] : T(0);
                }
            }
        }
    }
}

struct q8_block {
    ggml_fp16_t d;
    int8_t qs[32];
};

// Q8_0 blocks of 32 -> one INT8 row per output with a single scale, as the
// NPU's INT8 matmul has no per-block scales. From ggml-rknnoh.
void requant_q8(const q8_block *src, int8_t *dst, float *scale, int32_t K, int32_t N) {
    std::vector<float> row(K);
    for (int n = 0; n < N; n++) {
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            const q8_block &b = src[((size_t) n * K + k) / 32];
            row[k] = b.qs[k % 32] * ggml_fp16_to_fp32(b.d);
            amax = std::max(amax, std::fabs(row[k]));
        }
        scale[n] = amax / 127.0f;
        const float inv = scale[n] ? 1.0f / scale[n] : 0.0f;
        for (int k = 0; k < K; k++) {
            dst[(size_t) n * K + k] = (int8_t) std::lround(row[k] * inv);
        }
    }
}

// Blocked INT32 -> row-major F32 columns [col, col + N) of a row of ldc,
// scaled back by both operands' scales.
void layout_C_q8(const int32_t *src, float *dst, const float *sa, const float *sb,
                 int32_t M, int32_t N, int32_t ldc, int32_t col, int32_t subN) {
    for (int32_t m = 0; m < M; m++) {
        float *drow = dst + (size_t) m * ldc + col;
        for (int32_t n = 0; n < N; n++) {
            drow[n] = (float) src[(size_t) (n / subN) * M * subN + m * subN + n % subN] * sa[m] * sb[n];
        }
    }
}

// Blocked F32 -> row-major F32 columns [col, col + N) of a row of ldc. From
// ggml-rknnoh.
void layout_C(const float *src, float *dst, int32_t M, int32_t N, int32_t ldc, int32_t col, int32_t subN) {
    for (int32_t m = 0; m < M; m++) {
        float *drow = dst + (size_t) m * ldc + col;
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
//
// Like ggml-rknnoh, a matmul runs as column slices of N, one thread and NPU
// core per slice, each slice with its own context.
struct shape_slot {
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io = {};
    rknn_tensor_mem *A = nullptr;
    rknn_tensor_mem *C = nullptr;
};

struct weight_mem {
    std::vector<rknn_tensor_mem *> B;   // one per slice
    std::vector<float> scale;           // per output row, INT8 only
    ggml_type type = GGML_TYPE_COUNT;
    int K = 0, N = 0;
    bool loaded = false;
};

constexpr int npu_cores = 3;
constexpr int max_slice_N = 4096;

std::mutex slots_mutex;
std::map<std::tuple<int, int, int, bool, int>, shape_slot *> shapes;   // (M,K,n,q8,slice)
std::unordered_map<const void *, weight_mem> weights;                  // weight data -> B

shape_slot *shape_for(int M, int K, int n, bool q8, int slice) {
    const auto key = std::make_tuple(M, K, n, q8, slice);
    auto it = shapes.find(key);
    if (it != shapes.end()) {
        return it->second;
    }
    shape_slot *slot = new shape_slot();
    rknn_matmul_info info = {};
    info.M = M;
    info.K = K;
    info.N = n;
    info.type = q8 ? RKNN_INT8_MM_INT8_TO_INT32 : RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    // As ggml-rknnoh sets them: A and C in the performance layout, B in the
    // native one.
    info.AC_layout = 1;
    info.B_layout = 1;
    if (rknn_matmul_create(&slot->ctx, &info, &slot->io) != 0) {
        delete slot;
        return nullptr;
    }
    rknn_matmul_set_core_mask(slot->ctx, (rknn_core_mask) (1 << (slice % npu_cores)));
    slot->A = rknn_create_mem(slot->ctx, slot->io.A.size);
    slot->C = rknn_create_mem(slot->ctx, slot->io.C.size);
    if (!slot->A || !slot->C) {
        delete slot;
        return nullptr;
    }
    shapes[key] = slot;
    return slot;
}

// Slice boundaries in whole blocks of subN, at most max_slice_N wide.
std::vector<int> slice_bounds(int N, int subN) {
    const int blocks = N / subN;
    int n = std::max(adapter_threads(), (N + max_slice_N - 1) / max_slice_N);
    n = std::min(n, blocks);
    std::vector<int> b(n + 1);
    for (int t = 0; t <= n; t++) {
        b[t] = (int) ((int64_t) t * blocks / n) * subN;
    }
    return b;
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
    // F16 weights, or Q8_0 weights through the INT8 matmul, against F32 activations.
    if ((w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_Q8_0) ||
        a->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // 2D only. A batched matmul computed as 2D silently returns garbage.
    if (op->ne[2] > 1 || op->ne[3] > 1 || w->ne[2] > 1 || w->ne[3] > 1 ||
        a->ne[2] > 1 || a->ne[3] > 1) {
        return false;
    }
    // ggml-rknnoh's limits: K up to 8192, M up to 512; N is sliced.
    if (w->ne[0] > 8192 || a->ne[1] > 512) {
        return false;
    }
    // And its alignment: ggml-rknnoh asserts K % 32 for F16 and pads N to 16.
    // Handing librknnrt an unaligned shape takes the process down without a
    // diagnostic, so refuse it here where the guest can still schedule the node
    // on the CPU instead.
    if (w->ne[0] % 32 != 0 || w->ne[1] % (w->type == GGML_TYPE_Q8_0 ? 32 : 16) != 0) {
        return false;
    }
    return ggml_is_contiguous(w) && ggml_is_contiguous(a);
}

struct node_stats {
    size_t nodes = 0, loads = 0;
    double load_s = 0, run_s = 0;
};

static void report(node_stats &st, bool loaded, double load_s, double run_s) {
    static const bool on = getenv("VIAI_STATS") != nullptr;
    if (!on) {
        return;
    }
    std::lock_guard<std::mutex> lock(slots_mutex);
    st.nodes++;
    st.loads += loaded;
    st.load_s += load_s;
    st.run_s += run_s;
    if (st.nodes % 1000 == 0) {
        fprintf(stderr, "viai_rknn: %zu matmuls, %zu weight loads (%.1f s), %.3f ms per matmul\n",
                st.nodes, st.loads, st.load_s, 1e3 * st.run_s / st.nodes);
    }
}

static node_stats stats;

extern "C" bool viai_rknn_node(ggml_tensor *node) {
    if (node->op != GGML_OP_MUL_MAT) {
        return true;   // nothing to do for views and reshapes
    }
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    bool loaded = false;
    const ggml_tensor *w = node->src[0];
    const ggml_tensor *a = node->src[1];

    const int K = (int) w->ne[0];
    const int N = (int) w->ne[1];
    const int M = (int) a->ne[1];

    const bool q8 = w->type == GGML_TYPE_Q8_0;
    const size_t esize = q8 ? sizeof(int8_t) : sizeof(uint16_t);
    const int32_t subN = q8 ? 32 : 16, subK = 32;
    const std::vector<int> bounds = slice_bounds(N, subN);
    const int S = (int) bounds.size() - 1;

    if ((size_t) M * N * sizeof(float) > ggml_nbytes(node)) {
        GGML_LOG_ERROR("viai_rknn: %s result is %zu bytes, tensor holds %zu\n",
                       node->name, (size_t) M * N * sizeof(float), ggml_nbytes(node));
        return false;
    }

    std::vector<shape_slot *> slots(S);
    weight_mem *wm = nullptr;
    {
        std::lock_guard<std::mutex> lock(slots_mutex);
        for (int t = 0; t < S; t++) {
            slots[t] = shape_for(M, K, bounds[t + 1] - bounds[t], q8, t);
            if (!slots[t]) {
                GGML_LOG_ERROR("viai_rknn: no context for %s (%d x %d x %d)\n", node->name, M, K, N);
                return false;
            }
        }
        wm = &weights[w->data];
    }

    // A freed buffer's address can come back holding another weight.
    if (wm->loaded && (wm->type != w->type || wm->K != K || wm->N != N || (int) wm->B.size() != S)) {
        for (size_t t = 0; t < wm->B.size(); t++) {
            rknn_destroy_mem(slots[0]->ctx, wm->B[t]);
        }
        *wm = weight_mem();
    }
    // The weights, once per tensor, converted to the native layout and kept in
    // NPU memory for the session.
    if (!wm->loaded) {
        std::vector<int8_t> rows;
        if (q8) {
            rows.resize((size_t) N * K);
            wm->scale.resize(N);
            requant_q8((const q8_block *) w->data, rows.data(), wm->scale.data(), K, N);
        }
        wm->B.assign(S, nullptr);
        for (int t = 0; t < S; t++) {
            const int n = bounds[t + 1] - bounds[t];
            const size_t need_B = (size_t) n * ((K + subK - 1) / subK) * subK * esize;
            if (need_B > slots[t]->io.B.size) {
                GGML_LOG_ERROR("viai_rknn: %s weights need %zu bytes, device gave %u\n",
                               node->name, need_B, slots[t]->io.B.size);
                return false;
            }
            wm->B[t] = rknn_create_mem(slots[t]->ctx, slots[t]->io.B.size);
            if (!wm->B[t]) {
                GGML_LOG_ERROR("viai_rknn: no device memory for %s weights\n", node->name);
                return false;
            }
            if (q8) {
                layout_B(rows.data() + (size_t) bounds[t] * K, (int8_t *) wm->B[t]->virt_addr,
                         K, n, subN, subK);
            } else {
                layout_B((const uint16_t *) w->data + (size_t) bounds[t] * K,
                         (uint16_t *) wm->B[t]->virt_addr, K, n, subN, subK);
            }
        }
        wm->type = w->type;
        wm->K = K;
        wm->N = N;
        wm->loaded = true;
        loaded = true;
    }
    const auto t1 = clock::now();

    // A is the same for every slice: laid out once, copied into each context.
    const int32_t subKA = (int32_t) slots[0]->io.A.dims[2];
    const size_t need_A = (size_t) M * ((K + subKA - 1) / subKA) * subKA * esize;
    for (int t = 0; t < S; t++) {
        if (need_A > slots[t]->io.A.size) {
            GGML_LOG_ERROR("viai_rknn: %s needs %zu bytes of A, device gave %u\n",
                           node->name, need_A, slots[t]->io.A.size);
            return false;
        }
    }
    std::vector<float> scale_A;
    if (q8) {
        scale_A.resize(M);
        layout_A_q8((const float *) a->data, (int8_t *) slots[0]->A->virt_addr, scale_A.data(), M, K, subKA);
    } else {
        layout_A((const float *) a->data, (uint16_t *) slots[0]->A->virt_addr, M, K, subKA);
    }

    // Bound every call: a context is shared by every weight of its shape.
    std::vector<int> ok(S, 0);
    auto run = [&](int t) {
        shape_slot *slot = slots[t];
        const int n = bounds[t + 1] - bounds[t];
        if (t > 0) {
            memcpy(slot->A->virt_addr, slots[0]->A->virt_addr, need_A);
        }
        if (rknn_matmul_set_io_mem(slot->ctx, slot->A, &slot->io.A) != 0 ||
            rknn_matmul_set_io_mem(slot->ctx, wm->B[t], &slot->io.B) != 0 ||
            rknn_matmul_set_io_mem(slot->ctx, slot->C, &slot->io.C) != 0 ||
            rknn_matmul_run(slot->ctx) != 0) {
            return;
        }
        if (q8) {
            layout_C_q8((const int32_t *) slot->C->virt_addr, (float *) node->data,
                        scale_A.data(), wm->scale.data() + bounds[t], M, n, N, bounds[t],
                        slot->io.C.dims[2]);
        } else {
            layout_C((const float *) slot->C->virt_addr, (float *) node->data,
                     M, n, N, bounds[t], slot->io.C.dims[2]);
        }
        ok[t] = 1;
    };
    std::vector<std::thread> threads;
    for (int t = 1; t < S; t++) {
        threads.emplace_back(run, t);
    }
    run(0);
    for (auto &th : threads) {
        th.join();
    }
    for (int t = 0; t < S; t++) {
        if (!ok[t]) {
            GGML_LOG_ERROR("viai_rknn: %s slice %d failed on the device\n", node->name, t);
            return false;
        }
    }
    const auto t2 = clock::now();
    report(stats, loaded, std::chrono::duration<double>(t1 - t0).count(),
           std::chrono::duration<double>(t2 - t1).count());
    return true;
}
