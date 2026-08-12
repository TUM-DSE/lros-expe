#include "vaccel.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"

#define debug(fmt, ...) vaccel_debug("[noop] " fmt, ##__VA_ARGS__)
#define error(fmt, ...) vaccel_error("[noop] " fmt, ##__VA_ARGS__)


static int va_rknn_matmul_create(struct vaccel_session *sess, vaccel_matmul_ctx *ctx,
			 vaccel_matmul_info *info,
			 vaccel_matmul_io_attr *io_attr){

	return rknn_matmul_create(ctx, (rknn_matmul_info*)info, (rknn_matmul_io_attr *)io_attr);

}

static int va_rknn_create_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx,
		      uint32_t size, vaccel_tensor_mem *result)
{
	rknn_tensor_mem *rknn_tensor_memory = rknn_create_mem(ctx, size);

	if (rknn_tensor_memory == NULL) {
		return VACCEL_ENOMEM;
	}

	result->virt_addr = rknn_tensor_memory->virt_addr;
	result->handle= (vaccel_tensor_mem_handle *)rknn_tensor_memory;
	return VACCEL_OK;
}

static int va_rknn_destroy_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx,
		       vaccel_tensor_mem *mem)
{
	return rknn_destroy_mem(ctx, (rknn_tensor_mem *)mem->handle);

}

static int va_rknn_matmul_destroy(struct vaccel_session *sess, vaccel_matmul_ctx ctx)
{
	return rknn_matmul_destroy(ctx);
}

static int va_rknn_matmul_set_io_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx,
			     vaccel_tensor_mem_handle *mem,
			     vaccel_matmul_tensor_attr *attr)
{
	return rknn_matmul_set_io_mem(ctx, (rknn_tensor_mem *)mem, (rknn_matmul_tensor_attr *)attr);
}

static int va_rknn_matmul_set_core_mask(struct vaccel_session *sess,
				vaccel_matmul_ctx ctx,
				vaccel_core_mask core_mask)
{
	return rknn_matmul_set_core_mask(ctx, core_mask);
}

static int va_rknn_matmul_run(struct vaccel_session *sess, vaccel_matmul_ctx ctx)
{
	return rknn_matmul_run(ctx);
}

static int va_rknn_matmul_set_matrix(struct vaccel_session *sess,
			     vaccel_tensor_mem_handle *dst, void *src,
			     size_t nbytes){
	memcpy(((rknn_tensor_mem *)dst)->virt_addr, src, nbytes);
	return VACCEL_OK;
}

static int va_rknn_matmul_get_matrix(struct vaccel_session *sess, void *dst,
			     vaccel_tensor_mem_handle *src, size_t nbytes){
        memcpy(dst, ((rknn_tensor_mem *)src)->virt_addr, nbytes);
        return VACCEL_OK;

}

static int va_rknn_matmul_get_props(struct vaccel_session *sess, char *props, size_t nbytes) {
    if (nbytes == 0) {
        return VACCEL_EINVAL;
    }

    const int maxProps = 1 + 1;
    int nprops = nbytes > maxProps ? maxProps : nbytes;

    switch (nprops) {
        case 2:
            props[1] = 1; // Prefer matrix transforms: 0:no !0:yes
        case 1:
            props[0] = nprops - maxProps;
    }

    return VACCEL_OK;
}

struct vaccel_op ops[] = {
	VACCEL_OP_INIT(ops[0], VACCEL_OP_MATMUL_CREATE, va_rknn_matmul_create),
	VACCEL_OP_INIT(ops[1], VACCEL_OP_CREATE_MEM, va_rknn_create_mem),
	VACCEL_OP_INIT(ops[2], VACCEL_OP_DESTROY_MEM, va_rknn_destroy_mem),
	VACCEL_OP_INIT(ops[3], VACCEL_OP_MATMUL_DESTROY, va_rknn_matmul_destroy),
	VACCEL_OP_INIT(ops[4], VACCEL_OP_MATMUL_SET_IO, va_rknn_matmul_set_io_mem),
	VACCEL_OP_INIT(ops[5], VACCEL_OP_MATMUL_SET_CORE_MASK, va_rknn_matmul_set_core_mask),
	VACCEL_OP_INIT(ops[6], VACCEL_OP_MATMUL_RUN, va_rknn_matmul_run),
	VACCEL_OP_INIT(ops[7], VACCEL_OP_MATMUL_SET_MATRIX, va_rknn_matmul_set_matrix),
	VACCEL_OP_INIT(ops[8], VACCEL_OP_MATMUL_GET_MATRIX, va_rknn_matmul_get_matrix),
	VACCEL_OP_INIT(ops[9], VACCEL_OP_MATMUL_GET_PROPS, va_rknn_matmul_get_props),
};

static int init(void)
{
	return vaccel_plugin_register_ops(ops, sizeof(ops) / sizeof(ops[0]));
}

static int fini(void)
{
	return VACCEL_OK;
}

VACCEL_PLUGIN(.name = "rknn", .version = VACCEL_VERSION,
	      .vaccel_version = VACCEL_VERSION, .type = VACCEL_PLUGIN_GPU,
	      .init = init, .fini = fini)
