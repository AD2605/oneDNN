/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/generic/sycl/ref_matmul.hpp"
#include "gpu/generic/sycl/matmul_kernels.hpp"
#include "gpu/generic/sycl/specialization_constants.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

void ref_matmul_t::pd_t::init_conf() {
    conf_ = sycl_matmul_conf_t();

    conf_.do_scale_data
            = !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.do_scale_weights
            = !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values();
    conf_.do_scale_dst
            = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();
    conf_.single_weights_scale
            = attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ == 0;

    conf_.use_data_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC_0);
    conf_.use_weights_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS_0);
    conf_.use_dst_zeropoints
            = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);

    conf_.use_dropout = !attr()->dropout_.has_default_values();
    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());

    memory_desc_wrapper src_d = src_md();
    memory_desc_wrapper weights_d = weights_md();
    memory_desc_wrapper dst_d = dst_md();
    memory_desc_wrapper bias_d = weights_md(1);
    for (const auto &mdw : {src_d, weights_d, dst_d, bias_d}) {
        if (mdw.has_runtime_dims()) {
            any_runtime_params_ = true;
            return;
        }
    }
    init_rt_conf(conf_, data_md_t, dst_md_t, weights_md_t, bias_md_t, src_d,
            weights_d, dst_d, bias_d);
}

void ref_matmul_t::pd_t::init_rt_conf(sycl_matmul_conf_t &conf,
        xpu::sycl::md_t_spec_const &data_md_t_,
        xpu::sycl::md_t_spec_const &dst_md_t_,
        xpu::sycl::md_t_spec_const &weights_md_t_,
        xpu::sycl::md_t_spec_const &bias_md_t_, const memory_desc_wrapper src_d,
        const memory_desc_wrapper weights_d, const memory_desc_wrapper dst_d,
        const memory_desc_wrapper bias_d) const {

    // Lambda because this function will not be used anywhere else
    auto init_md_t_sc_from_md = [=](xpu::sycl::md_t_spec_const &md_t_sc,
                                        const memory_desc_t *md) -> void {
        // copied from types.hpp::md_t::md_t(memory_desc_t*)
        constexpr int max_dims = 6;
        using dim32_t = int32_t;
        using dims32_t = dim32_t[max_dims];

        memory_desc_wrapper mdw(md);

        assert(mdw.format_kind() == format_kind::blocked);
        assert(mdw.ndims() <= max_dims);

        const auto &blk = mdw.blocking_desc();

        md_t_sc.data_type_ = mdw.data_type();
#define CHECK_AND_ASSIGN(lhs, rhs) \
    assert((rhs) <= INT32_MAX); \
    (lhs) = static_cast<dim32_t>(rhs)

        CHECK_AND_ASSIGN(md_t_sc.ndims_, mdw.ndims());
        CHECK_AND_ASSIGN(md_t_sc.offset0_, mdw.offset0());
        CHECK_AND_ASSIGN(md_t_sc.inner_nblks_, blk.inner_nblks);

        for (int d = 0; d < mdw.ndims(); d++) {
            CHECK_AND_ASSIGN(md_t_sc.dims_[d], mdw.dims()[d]);
            CHECK_AND_ASSIGN(md_t_sc.padded_dims_[d], mdw.padded_dims()[d]);
            CHECK_AND_ASSIGN(
                    md_t_sc.padded_offsets_[d], mdw.padded_offsets()[d]);
            CHECK_AND_ASSIGN(md_t_sc.strides_[d], blk.strides[d]);
            CHECK_AND_ASSIGN(md_t_sc.inner_blks_[d], blk.inner_blks[d]);
            CHECK_AND_ASSIGN(md_t_sc.inner_idxs_[d], blk.inner_idxs[d]);
        }
#undef CHECK_AND_ASSIGN
    };

    int matmul_dim_1 = ndims() - 2;
    int matmul_dim_2 = ndims() - 1;

    memory_desc_t data_md_copy = *src_d.md_;
    auto &data_strides = data_md_copy.format_desc.blocking.strides;
    if (data_strides[matmul_dim_1] < data_strides[matmul_dim_2]) {
        std::swap(data_strides[matmul_dim_1], data_strides[matmul_dim_2]);
        std::swap(data_md_copy.dims[matmul_dim_1],
                data_md_copy.dims[matmul_dim_2]);
        conf.transpose_data = true;
    }
    //conf.data_md = xpu::sycl::md_t(&data_md_copy);
    init_md_t_sc_from_md(data_md_t_, &data_md_copy);

    memory_desc_t weights_md_copy = *weights_d.md_;
    auto &weights_strides = weights_md_copy.format_desc.blocking.strides;
    if (weights_strides[matmul_dim_1] < weights_strides[matmul_dim_2]) {
        std::swap(weights_strides[matmul_dim_1], weights_strides[matmul_dim_2]);
        std::swap(weights_md_copy.dims[matmul_dim_1],
                weights_md_copy.dims[matmul_dim_2]);
        conf.transpose_weights = true;
    }
    //conf.weights_md = xpu::sycl::md_t(&weights_md_copy);
    init_md_t_sc_from_md(weights_md_t_, &weights_md_copy);

    memory_desc_t dst_md_copy = *dst_d.md_;
    auto &dst_strides = dst_md_copy.format_desc.blocking.strides;
    if (dst_strides[matmul_dim_1] < dst_strides[matmul_dim_2]) {
        std::swap(dst_strides[matmul_dim_1], dst_strides[matmul_dim_2]);
        std::swap(
                dst_md_copy.dims[matmul_dim_1], dst_md_copy.dims[matmul_dim_2]);
        conf.transpose_dst = true;
    }
    //conf.dst_md = xpu::sycl::md_t(&dst_md_copy);
    init_md_t_sc_from_md(dst_md_t_, &dst_md_copy);

    if (with_bias()) {
        memory_desc_t bias_md_copy = *bias_d.md_;
        auto &bias_strides = bias_md_copy.format_desc.blocking.strides;
        if (bias_strides[matmul_dim_1] < bias_strides[matmul_dim_2]) {
            std::swap(bias_strides[matmul_dim_1], bias_strides[matmul_dim_2]);
            std::swap(bias_md_copy.dims[matmul_dim_1],
                    bias_md_copy.dims[matmul_dim_2]);
            conf.transpose_bias = true;
        }
        //conf.bias_md = xpu::sycl::md_t(&bias_md_copy);
        init_md_t_sc_from_md(bias_md_t_, &bias_md_copy);
    }

    dims_t dst_blocks;
    for (int i = 0; i < matmul_kernel_fwd_t::max_supported_ndims; i++) {
        if (i < dst_md_t.ndims_) {
            dst_blocks[i] = dst_md_t.dims_[i];
        } else {
            dst_blocks[i] = 1;
        }
    }
    dst_blocks[matmul_dim_1] = math::div_up(
            dst_blocks[matmul_dim_1], matmul_kernel_fwd_t::register_block_N);
    dst_blocks[matmul_dim_2] = math::div_up(
            dst_blocks[matmul_dim_2], matmul_kernel_fwd_t::register_block_M);
    int n_blocks = 1;
    for (int i = 0; i < matmul_kernel_fwd_t::max_supported_ndims; i++) {
        n_blocks *= dst_blocks[i];
    }
    conf.wk_size = n_blocks;

    int high_two_bits = 3 << (ndims() - 2);
    // last two dimensions of data and weights are never broadcast
    conf.data_mask = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims())
            | high_two_bits;
    conf.weights_mask
            = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims())
            | high_two_bits;
    conf.bias_mask = utils::get_dims_mask(dst_d.dims(), bias_d.dims(), ndims());
}

status_t ref_matmul_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<matmul_kernel_fwd_t>();
    CHECK(create_kernel<detail::matmul::md_t_spec_const_id>(engine, kid,
            &kernel_,
            {{pd()->data_md_t, pd()->dst_md_t, pd()->weights_md_t,
                    pd()->bias_md_t}}));
    return status::success;
}

status_t ref_matmul_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->dst_md()).size() == 0) return status::success;

    sycl_matmul_conf_t conf = pd()->conf_;
    if (pd()->any_runtime_params_) {
        const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
        const auto weights_d
                = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
        const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
        const auto bias_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));
        //Need to use const cast here
        pd()->init_rt_conf(conf, pd()->data_md_t, pd()->dst_md_t,
                pd()->weights_md_t, pd()->bias_md_t, src_d, weights_d, dst_d,
                bias_d);
    }

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        matmul_kernel_fwd_t matmul_kernel(conf, cgh, ctx);

        const int block_size = 32;
        const int wg_size = 32;

        const int t_work = conf.wk_size;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = utils::div_up(t_work, wg_work);

        cgh.parallel_for(
                ::sycl::nd_range<1>(wg_cnt * wg_size, wg_size), matmul_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
