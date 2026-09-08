// Host half of the viai backend: runs whole ggml graphs on a real
// ggml-cuda backend inside the QEMU process that dlopens this plugin.
//
// Derived from ggml-rpc's server, which is already socket-free: a class taking
// a ggml_backend_t with one method per command. Only the framing changes, from
// a byte stream to VACCEL_OP_EXEC arguments.

#include "viai-proto.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <sys/sysinfo.h>

// Two kinds of layer B adapter. CUDA hosts a real ggml backend and hands it the
// graph. RKNN drives librknnrt per node, because ggml-rknnoh is a host-side
// backend and does not run in this process. The server asks both the same two
// questions and knows nothing else about them.
#ifdef VIAI_ADAPTER_RKNN
extern "C" bool viai_rknn_supports(const ggml_tensor *op);
extern "C" bool viai_rknn_node(ggml_tensor *node);
#endif

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class viai_server {
public:
    explicit viai_server(ggml_backend_t backend) : backend(backend) {
    }
    ~viai_server();

    void alloc_buffer(const viai_msg_alloc_buffer_req & request, viai_msg_alloc_buffer_rsp & response);
    void get_alignment(viai_msg_get_alignment_rsp & response);
    void get_max_size(viai_msg_get_max_size_rsp & response);
    bool buffer_get_base(const viai_msg_buffer_get_base_req & request, viai_msg_buffer_get_base_rsp & response);
    bool free_buffer(const viai_msg_free_buffer_req & request);
    bool buffer_clear(const viai_msg_buffer_clear_req & request);
    bool set_tensor(const viai_msg_set_tensor_req & request, const void * data);
    bool mem_write(const viai_msg_mem_xfer_req & request, const void * data);
    bool mem_read(const viai_msg_mem_xfer_req & request, void * out);
    bool supports_op(const viai_msg_supports_op_req & request, viai_msg_supports_op_rsp & response);
    bool get_tensor(const viai_msg_get_tensor_req & request, void * out);
    bool copy_tensor(const viai_msg_copy_tensor_req & request, viai_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input, viai_msg_graph_compute_rsp & response);
    bool init_tensor(const viai_msg_init_tensor_req & request);
    bool get_alloc_size(const viai_msg_get_alloc_size_req & request, viai_msg_get_alloc_size_rsp & response);

private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const viai_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const viai_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    ggml_backend_t backend;
    std::unordered_set<ggml_backend_buffer_t> buffers;
};

bool viai_server::get_alloc_size(const viai_msg_get_alloc_size_req & request, viai_msg_get_alloc_size_rsp & response) {
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);

    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ggml_free(ctx);
        return false;
    }

    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backend);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft,tensor);

    ggml_free(ctx);
    return true;
}

void viai_server::alloc_buffer(const viai_msg_alloc_buffer_req & request, viai_msg_alloc_buffer_rsp & response) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n", __func__, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        GGML_LOG_ERROR("[%s] size: %" PRIu64 " -> failed\n", __func__, request.size);
    }
}

void viai_server::get_alignment(viai_msg_get_alignment_rsp & response) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    GGML_PRINT_DEBUG("[%s] alignment: %lu\n", __func__, alignment);
    response.alignment = alignment;
}

void viai_server::get_max_size(viai_msg_get_max_size_rsp & response) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    GGML_PRINT_DEBUG("[%s] max_size: %lu\n", __func__, max_size);
    response.max_size = max_size;
}

bool viai_server::buffer_get_base(const viai_msg_buffer_get_base_req & request, viai_msg_buffer_get_base_rsp & response) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool viai_server::free_buffer(const viai_msg_free_buffer_req & request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool viai_server::buffer_clear(const viai_msg_buffer_clear_req & request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * viai_server::deserialize_tensor(struct ggml_context * ctx, const viai_tensor * tensor) {
    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool viai_server::set_tensor(const viai_msg_set_tensor_req & request, const void * data) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    ggml_backend_tensor_set(tensor, data, request.offset, request.size);
    ggml_free(ctx);
    return true;
}

bool viai_server::init_tensor(const viai_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ggml_free(ctx);
        return false;
    }

    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        GGML_LOG_ERROR("Null buffer for tensor passed to init_tensor function\n");
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ggml_free(ctx);
        return false;
    }

    ggml_free(ctx);
    return true;
}

bool viai_server::get_tensor(const viai_msg_get_tensor_req & request, void * out) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    ggml_backend_tensor_get(tensor, out, request.offset, request.size);
    ggml_free(ctx);
    return true;
}

bool viai_server::copy_tensor(const viai_msg_copy_tensor_req & request, viai_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ggml_free(ctx);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_PRINT_DEBUG("[%s] out-of-bounds write in viai_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        ggml_free(ctx);
        return false;
    }

    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n",
                     __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ggml_free(ctx);
    return true;
}

ggml_tensor * viai_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const viai_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    const viai_tensor * tensor = tensor_ptrs.at(id);
    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    return result;
}

bool viai_server::graph_compute(const std::vector<uint8_t> & input, viai_msg_graph_compute_rsp & response) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(viai_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(viai_tensor)) {
        return false;
    }
    const viai_tensor * tensors = (const viai_tensor *)(input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors));
    GGML_PRINT_DEBUG("[%s] n_nodes: %u, n_tensors: %u\n", __func__, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const viai_tensor*> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        // create_node returns null for an id it cannot resolve, and ggml-rpc
        // hands that straight to the backend. A backend then dereferences it
        // and takes the whole QEMU process down with no message, which is
        // expensive to diagnose from the guest side. Say what was missing.
        if (graph->nodes[i] == nullptr) {
            GGML_LOG_ERROR("[%s] node %u (id %llx) did not resolve; %u nodes %u tensors\n",
                           __func__, i, (unsigned long long) id, n_nodes, n_tensors);
            ggml_free(ctx);
            return false;
        }
    }
#ifdef VIAI_ADAPTER_RKNN
    // Submit boundaries, so a fault can be placed between nodes of one graph or
    // between two graphs. Everything here is flushed: a fault in the plugin
    // takes QEMU down and buffered output is lost.
    ggml_status status = GGML_STATUS_SUCCESS;
    for (uint32_t i = 0; i < n_nodes; i++) {
        if (!viai_rknn_node(graph->nodes[i])) {
            GGML_LOG_ERROR("[%s] node %u (%s) failed on the device\n", __func__, i,
                           ggml_op_name(graph->nodes[i]->op));
            status = GGML_STATUS_FAILED;
            break;
        }
    }
#else
    ggml_status status = ggml_backend_graph_compute(backend, graph);
#endif
    response.result = status;
    ggml_free(ctx);
    return true;
}

bool viai_server::supports_op(const viai_msg_supports_op_req & request, viai_msg_supports_op_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * (GGML_MAX_SRC + 2),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        ggml_free(ctx);
        response.result = 0;
        return true;
    }
    // deserialize_tensor leaves src empty, and a backend's supports_op reads
    // through them. Rebuild them from the request rather than hand the device
    // a tensor with null sources.
    const uint32_t n_src = request.n_src < GGML_MAX_SRC ? request.n_src : GGML_MAX_SRC;
    for (uint32_t i = 0; i < n_src; i++) {
        tensor->src[i] = deserialize_tensor(ctx, &request.src[i]);
        if (tensor->src[i] == nullptr) {
            ggml_free(ctx);
            response.result = 0;
            return true;
        }
    }
#ifdef VIAI_ADAPTER_RKNN
    response.result = viai_rknn_supports(tensor) ? 1 : 0;
#else
    response.result = ggml_backend_dev_supports_op(ggml_backend_get_device(backend), tensor) ? 1 : 0;
#endif
    ggml_free(ctx);
    return true;
}

// Layer A: bytes in and out of a buffer, with no tensor involved. Bounds are
// checked against the buffer the caller named, so a bad offset is refused
// rather than writing over whatever follows.
static bool viai_mem_range(const std::unordered_set<ggml_backend_buffer_t> & buffers,
                           const viai_msg_mem_xfer_req & r, void ** addr) {
    ggml_backend_buffer_t buf = reinterpret_cast<ggml_backend_buffer_t>(r.remote_ptr);
    if (buffers.find(buf) == buffers.end()) {
        return false;
    }
    const uint64_t size = ggml_backend_buffer_get_size(buf);
    if (r.offset > size || r.size > size - r.offset) {
        return false;
    }
    *addr = (uint8_t *) ggml_backend_buffer_get_base(buf) + r.offset;
    return true;
}

bool viai_server::mem_write(const viai_msg_mem_xfer_req & request, const void * data) {
    void * addr = nullptr;
    if (!viai_mem_range(buffers, request, &addr)) {
        GGML_LOG_ERROR("viai: MEM_WRITE outside buffer %llx +%llu of %llu\n",
                       (unsigned long long) request.remote_ptr,
                       (unsigned long long) request.offset,
                       (unsigned long long) request.size);
        return false;
    }
    memcpy(addr, data, request.size);
    return true;
}

bool viai_server::mem_read(const viai_msg_mem_xfer_req & request, void * out) {
    void * addr = nullptr;
    if (!viai_mem_range(buffers, request, &addr)) {
        return false;
    }
    memcpy(out, addr, request.size);
    return true;
}

viai_server::~viai_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

// One CUDA backend for the process; one server, and so one buffer set, per
// vAccel session, because two guests must not see each other's buffers.
static ggml_backend_t viai_backend() {
    static ggml_backend_t backend = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
#ifdef VIAI_ADAPTER_RKNN
        // No ggml backend here: the adapter drives the NPU itself through
        // librknnrt. A CPU backend is still wanted for layer A, whose buffers
        // are ordinary host memory the NPU reads from, and it makes
        // host_buffers come out true so the guest stages instead of trying to
        // make a matmul-only device own tensors.
        backend = ggml_backend_cpu_init();
        return backend;
#else
        // Which ggml backend the adapter drives is the plugin's business, not
        // the protocol's. The server below never asks what it is.
        const char *name = getenv("VIAI_BACKEND");
        if (!name) {
            name = "CUDA";
        }
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(name);
        if (reg && ggml_backend_reg_dev_count(reg) > 0) {
            backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        }
        // A backend that takes a thread count expects to be told: llama.cpp
        // sets it on every backend it creates, and nothing else does. Left
        // unset, RKNNOH read an uninitialised count and spawned tens of
        // thousands of empty column slices. Asked for by the standard proc
        // name, so the server still does not know which backend it holds.
        if (backend && reg) {
            auto set_n_threads = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_n_threads) {
                const char *env = getenv("VIAI_BACKEND_THREADS");
                set_n_threads(backend, env ? atoi(env) : 4);
            }
        }
        if (!backend) {
            fprintf(stderr, "viai: no ggml backend '%s' in this plugin\n", name);
        }
#endif
    }
    return backend;
}

static std::mutex viai_mutex;
static std::map<uint64_t, std::unique_ptr<viai_server>> viai_servers;

static viai_server * viai_server_for(uint64_t sess_id) {
    auto it = viai_servers.find(sess_id);
    if (it != viai_servers.end()) {
        return it->second.get();
    }
    ggml_backend_t backend = viai_backend();
    if (!backend) {
        return nullptr;
    }
    auto srv = std::unique_ptr<viai_server>(new viai_server(backend));
    viai_server * raw = srv.get();
    viai_servers.emplace(sess_id, std::move(srv));
    return raw;
}

// read[0] is the command, read[1] the fixed-size header if the command has one,
// read[2] the bulk payload if it has one. write[0] is the reply.
extern "C" int va_ggml_exec(uint64_t sess_id,
                            const void **rd, const unsigned int *rd_size, unsigned int nr_rd,
                            void **wr, const unsigned int *wr_size, unsigned int nr_wr)
{
    if (nr_rd < 1 || rd_size[0] < sizeof(uint32_t)) {
        return 1;
    }
    uint32_t cmd;
    memcpy(&cmd, rd[0], sizeof(cmd));

    const void *hdr = nr_rd > 1 ? rd[1] : nullptr;
    const unsigned int hdr_size = nr_rd > 1 ? rd_size[1] : 0;
    const void *bulk = nr_rd > 2 ? rd[2] : nullptr;
    void *out = nr_wr > 0 ? wr[0] : nullptr;
    const unsigned int out_size = nr_wr > 0 ? wr_size[0] : 0;

    // Answered before a backend exists, so a mismatch is reported rather than
    // crashing somewhere later.
    // Adapter negotiation, answered before a backend exists so a mismatch is
    // reported rather than crashing somewhere later. This is the only place the
    // host decides which engine a session speaks; everything in layer A is
    // answered the same way regardless.
    if (cmd == VIAI_CMD_HELLO) {
        if (out_size < sizeof(viai_msg_hello_rsp) ||
            hdr_size < sizeof(viai_msg_hello_req)) {
            return 1;
        }
        const viai_msg_hello_req *req = (const viai_msg_hello_req *) hdr;
        viai_msg_hello_rsp *rsp = (viai_msg_hello_rsp *) out;
        memset(rsp, 0, sizeof(*rsp));
        rsp->proto_version = VIAI_PROTO_VERSION;
        snprintf(rsp->runtime, VIAI_RT_NAME_LEN, "%s", VIAI_RT_GGML);
        rsp->abi_hash = viai_ggml_abi_hash();
        rsp->accepted = (strncmp(req->runtime, VIAI_RT_GGML, VIAI_RT_NAME_LEN) == 0);
        // Whether this accelerator can own tensors, or only borrow operands.
        ggml_backend_t b = viai_backend();
        rsp->host_buffers = b && ggml_backend_buft_is_host(
                                     ggml_backend_get_default_buffer_type(b));
        return 0;
    }


    std::lock_guard<std::mutex> lock(viai_mutex);
    viai_server *srv = viai_server_for(sess_id);
    if (!srv) {
        return 1;
    }

#define NEED_HDR(T) if (hdr_size < sizeof(T)) { return 1; }
#define NEED_OUT(T) if (out_size < sizeof(T)) { return 1; }

    // Layer A. Nothing below this switch mentions ggml: it is the device
    // allocator and its properties, and it would serve any engine unchanged.
    // This is the half that belongs in vAccel proper, and the half the OS
    // accounts and admits on.
    switch (cmd) {
    case VIAI_CMD_MEM_ALLOC: {
        NEED_HDR(viai_msg_alloc_buffer_req); NEED_OUT(viai_msg_alloc_buffer_rsp);
        srv->alloc_buffer(*(const viai_msg_alloc_buffer_req *) hdr,
                          *(viai_msg_alloc_buffer_rsp *) out);
        return 0;
    }
    case VIAI_CMD_MEM_WRITE: {
        NEED_HDR(viai_msg_mem_xfer_req);
        const viai_msg_mem_xfer_req *r = (const viai_msg_mem_xfer_req *) hdr;
        if (!bulk || rd_size[2] < r->size) {
            return 1;
        }
        return srv->mem_write(*r, bulk) ? 0 : 1;
    }
    case VIAI_CMD_MEM_READ: {
        NEED_HDR(viai_msg_mem_xfer_req);
        const viai_msg_mem_xfer_req *r = (const viai_msg_mem_xfer_req *) hdr;
        if (!out || out_size < r->size) {
            return 1;
        }
        return srv->mem_read(*r, out) ? 0 : 1;
    }
    case VIAI_CMD_GET_ALIGNMENT: {
        NEED_OUT(viai_msg_get_alignment_rsp);
        srv->get_alignment(*(viai_msg_get_alignment_rsp *) out);
        return 0;
    }
    case VIAI_CMD_GET_MAX_SIZE: {
        NEED_OUT(viai_msg_get_max_size_rsp);
        srv->get_max_size(*(viai_msg_get_max_size_rsp *) out);
        return 0;
    }
    case VIAI_CMD_MEM_GET_BASE: {
        NEED_HDR(viai_msg_buffer_get_base_req); NEED_OUT(viai_msg_buffer_get_base_rsp);
        return srv->buffer_get_base(*(const viai_msg_buffer_get_base_req *) hdr,
                                    *(viai_msg_buffer_get_base_rsp *) out) ? 0 : 1;
    }
    case VIAI_CMD_MEM_FREE: {
        NEED_HDR(viai_msg_free_buffer_req);
        return srv->free_buffer(*(const viai_msg_free_buffer_req *) hdr) ? 0 : 1;
    }
    case VIAI_CMD_MEM_CLEAR: {
        NEED_HDR(viai_msg_buffer_clear_req);
        return srv->buffer_clear(*(const viai_msg_buffer_clear_req *) hdr) ? 0 : 1;
    }
    // Layer B. Everything from here interprets the engine's own payloads and
    // would be replaced wholesale by another adapter.
    case VIAI_CMD_RT_SET_TENSOR: {
        NEED_HDR(viai_msg_set_tensor_req);
        const viai_msg_set_tensor_req *r = (const viai_msg_set_tensor_req *) hdr;
        if (!bulk || rd_size[2] < r->size) {
            return 1;
        }
        return srv->set_tensor(*r, bulk) ? 0 : 1;
    }
    case VIAI_CMD_RT_GET_TENSOR: {
        NEED_HDR(viai_msg_get_tensor_req);
        const viai_msg_get_tensor_req *r = (const viai_msg_get_tensor_req *) hdr;
        if (!out || out_size < r->size) {
            return 1;
        }
        return srv->get_tensor(*r, out) ? 0 : 1;
    }
    case VIAI_CMD_RT_COPY_TENSOR: {
        NEED_HDR(viai_msg_copy_tensor_req); NEED_OUT(viai_msg_copy_tensor_rsp);
        return srv->copy_tensor(*(const viai_msg_copy_tensor_req *) hdr,
                                *(viai_msg_copy_tensor_rsp *) out) ? 0 : 1;
    }
    case VIAI_CMD_RT_SUBMIT: {
        NEED_OUT(viai_msg_graph_compute_rsp);
        std::vector<uint8_t> input((const uint8_t *) hdr,
                                   (const uint8_t *) hdr + hdr_size);
        return srv->graph_compute(input, *(viai_msg_graph_compute_rsp *) out) ? 0 : 1;
    }
    case VIAI_CMD_GET_DEVICE_MEMORY: {
        NEED_OUT(viai_msg_get_device_memory_rsp);
        viai_msg_get_device_memory_rsp *rsp = (viai_msg_get_device_memory_rsp *) out;
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(viai_backend()), &f, &t);
        // An accelerator that shares system memory has no pool of its own to
        // report, and the CPU backend answers zero. llama.cpp reads that as
        // "nowhere to put anything" and offloads no layers at all, so answer
        // with the memory the device can actually reach.
        if (t == 0) {
            struct sysinfo si;
            if (sysinfo(&si) == 0) {
                t = (size_t) si.totalram * si.mem_unit;
                f = (size_t) si.freeram * si.mem_unit;
            }
        }
        rsp->free_mem = f;
        rsp->total_mem = t;
        return 0;
    }
    case VIAI_CMD_RT_INIT_TENSOR: {
        NEED_HDR(viai_msg_init_tensor_req);
        return srv->init_tensor(*(const viai_msg_init_tensor_req *) hdr) ? 0 : 1;
    }
    case VIAI_CMD_RT_ALLOC_SIZE: {
        NEED_HDR(viai_msg_get_alloc_size_req); NEED_OUT(viai_msg_get_alloc_size_rsp);
        return srv->get_alloc_size(*(const viai_msg_get_alloc_size_req *) hdr,
                                   *(viai_msg_get_alloc_size_rsp *) out) ? 0 : 1;
    }
    case VIAI_CMD_RT_SUPPORTS: {
        NEED_HDR(viai_msg_supports_op_req); NEED_OUT(viai_msg_supports_op_rsp);
        return srv->supports_op(*(const viai_msg_supports_op_req *) hdr,
                                *(viai_msg_supports_op_rsp *) out) ? 0 : 1;
    }
    default:
        return 1;
    }
#undef NEED_HDR
#undef NEED_OUT
}
