/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/generic/sycl/ref_inner_product.hpp"

namespace dnnl::impl::gpu::generic::sycl {

namespace detail {
status_t init_matmul_pd(impl::engine_t *engine,
        const primitive_attr_t *attributes, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc,
        std::shared_ptr<primitive_desc_t> &matmul_pd) {

    matmul_desc_t matmul_desc;
    CHECK(matmul_desc_init(
            &matmul_desc, src_desc, weights_desc, bias_desc, dst_desc));

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<op_desc_t *>(&matmul_desc), attributes, nullptr);

    if (!it.is_initialized()) return status::invalid_arguments;
    while (++it != it.end()) {
        if (*it) {
            matmul_pd = *it;
            break;
        }
    }
    return status::success;
}

status_t flatten_md(const memory_desc_t &desc, memory_desc_t &flattened_md,
        format_tag_t format_tag) {
    // Always flattens from a nD to a 2D memory layout, with batch as the first dimension
    assert(format_tag == format_tag::ab || format_tag == format_tag::ba);
    const auto &dimensions = desc.dims;
    int64_t flattened_dimension = 1;
    for (int i = 1; i < desc.ndims; i++) {
        flattened_dimension *= dimensions[i];
    }
    dims_t reshaped_dims;
    if (format_tag == format_tag::ab) {
        reshaped_dims[0] = desc.dims[0];
        reshaped_dims[1] = flattened_dimension;
    } else {
        reshaped_dims[0] = flattened_dimension;
        reshaped_dims[1] = desc.dims[0];
    }
    CHECK(memory_desc_init_by_tag(
            flattened_md, 2, reshaped_dims, desc.data_type, format_tag));
    return status::success;
}

} // namespace detail

status_t ref_inner_product_fwd_t::init(impl::engine_t *engine) {
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;
    return matmul_primitive->init(engine);
}

status_t ref_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested, matmul_primitive);
    exec_ctx_t copied_ctx(ctx);
    copied_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());
    return matmul_primitive->execute(copied_ctx);
}

status_t ref_inner_product_bwd_data_t::init(impl::engine_t *engine) {
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;
    return status::success;
}

status_t ref_inner_product_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested, matmul_primitive);

    exec_args_t args_copy(ctx.args());
    // Map src and dst to diff_dst and diff_src respectively
    args_copy[DNNL_ARG_SRC] = args_copy[DNNL_ARG_DIFF_DST];
    args_copy[DNNL_ARG_DST] = args_copy[DNNL_ARG_DIFF_SRC];
    exec_ctx_t copied_ctx(ctx.stream(), std::move(args_copy));

    copied_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());

    return matmul_primitive->execute(copied_ctx);
}

status_t ref_inner_product_bwd_weights_t::init(impl::engine_t *engine) {
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;

    if (pd()->with_bias()) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t>
                p_reduction;
        CHECK(pd()->reduction_pd->create_primitive_nested(p_reduction, engine));
        reduction_primitive = p_reduction.first;
    }

    return status::success;
}

status_t ref_inner_product_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested_multiple, matmul_primitive);

    exec_args_t args_copy(ctx.args());
    // Map src and dst to diff_dst and diff_src respectively
    auto src_memory_arg = args_copy[DNNL_ARG_SRC];
    args_copy[DNNL_ARG_SRC] = args_copy[DNNL_ARG_DIFF_DST];
    args_copy[DNNL_ARG_WEIGHTS] = src_memory_arg;
    args_copy[DNNL_ARG_DST] = args_copy[DNNL_ARG_DIFF_WEIGHTS];
    exec_ctx_t copied_ctx(ctx.stream(), std::move(args_copy));

    copied_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());
    // calcules dL/dW;
    CHECK(matmul_primitive->execute(copied_ctx));

    if (pd()->with_bias()) {
        //calculates dL/dB
        nested_scratchpad_t reduction_scratchpad(ctx,
                memory_tracking::names::key_nested_multiple + 1,
                reduction_primitive);
        exec_args_t args_copy_reduction(ctx.args());
        args_copy_reduction[DNNL_ARG_SRC]
                = args_copy_reduction[DNNL_ARG_DIFF_DST];
        args_copy_reduction[DNNL_ARG_DST]
                = args_copy_reduction[DNNL_ARG_DIFF_BIAS];
        exec_ctx_t copied_ctx_reduction(
                ctx.stream(), std::move(args_copy_reduction));

        copied_ctx_reduction.set_scratchpad_grantor(
                reduction_scratchpad.grantor());
        CHECK(reduction_primitive->execute(copied_ctx_reduction));
    }
    return status::success;
}

} // namespace dnnl::impl::gpu::generic::sycl
