/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
* Copyright 2024-2025 Codeplay Software Limited
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

#ifndef GPU_GENERIC_SYCL_REF_INNER_PRODUCT_HPP
#define GPU_GENERIC_SYCL_REF_INNER_PRODUCT_HPP

#include "common/primitive_desc_iterator.hpp"
#include "common/reduction_pd.hpp"
#include "gpu/generic/sycl/ref_matmul.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl::impl::gpu::generic::sycl {

namespace detail {
status_t init_matmul_pd(impl::engine_t *engine,
        const primitive_attr_t *attributes, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc,
        std::shared_ptr<primitive_desc_t> &matmul_pd);

status_t flatten_md(const memory_desc_t &desc, memory_desc_t &flattened_md,
        format_tag_t format_tag);

} // namespace detail

struct ref_inner_product_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;
        using sm = primitive_attr_t::skip_mask_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto src_dt = arg_md(DNNL_ARG_SRC)->data_type;
            auto weights_dt = arg_md(DNNL_ARG_WEIGHTS)->data_type;
            auto dst_dt = arg_md(DNNL_ARG_DST)->data_type;
            auto bias_dt = with_bias() ? arg_md(DNNL_ARG_BIAS)->data_type
                                       : data_type::undef;

            const bool ok = (set_default_params() == status::success)
                    && is_fwd()
                    && check_if_dtypes_valid(
                            src_dt, dst_dt, bias_dt, weights_dt)
                    && sycl_post_ops_t::post_ops_ok(attr())
                    && (attr_.set_default_formats(dst_md()) == status::success)
                    && attr()->has_default_values(sm::scales_runtime
                            | sm::zero_points_runtime | sm::post_ops
                            | sm::dropout | sm::scales_runtime_data_type
                            | sm::zero_points_runtime_data_type)
                    && memory_desc_wrapper(src_md()).is_plain()
                    && memory_desc_wrapper(dst_md())
                               .is_plain(); // Blocked memory formats are not supported

            if (not ok) { return status::unimplemented; }

            memory_desc_t src_reshaped;
            memory_desc_t weights_reshaped;
            memory_desc_t bias_reshaped;
            CHECK(detail::flatten_md(
                    *arg_md(DNNL_ARG_SRC), src_reshaped, format_tag::ab));
            CHECK(detail::flatten_md(*arg_md(DNNL_ARG_WEIGHTS),
                    weights_reshaped, format_tag::ba));
            const auto bias_md = arg_md(DNNL_ARG_BIAS);
            //Reshape bias to 1 x OC;
            dims_t reshaped_bias_dims {1, bias_md->dims[0]};
            CHECK(memory_desc_init_by_tag(bias_reshaped, 2, reshaped_bias_dims,
                    bias_md->data_type, format_tag::ab));

            CHECK(gpu::generic::sycl::detail::init_matmul_pd(engine, attr(),
                    &src_reshaped, &weights_reshaped, &bias_reshaped,
                    arg_md(DNNL_ARG_DST), matmul_pd));

            // book scratchpad for the matmul
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd->scratchpad_registry());
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd;

    private:
        bool check_if_dtypes_valid(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &bias_dt,
                const data_type_t &weight_dt) const {
            using namespace data_type;
            return (utils::one_of(src_dt, f32) && utils::one_of(weight_dt, f32)
                           && utils::one_of(dst_dt, f32)
                           && utils::one_of(bias_dt, f32, undef))
                    || (utils::one_of(src_dt, f16)
                            && utils::one_of(weight_dt, f16)
                            && utils::one_of(dst_dt, f16, f32, s8, u8)
                            && utils::one_of(bias_dt, f16, f32, undef))
                    || (utils::one_of(src_dt, u8, s8)
                            && utils::one_of(weight_dt, s8)
                            && utils::one_of(dst_dt, u8, s8, s32, bf16, f32)
                            && utils::one_of(
                                    bias_dt, u8, s8, s32, bf16, f32, undef))
                    || (utils::one_of(src_dt, bf16)
                            && utils::one_of(weight_dt, bf16)
                            && utils::one_of(dst_dt, f32, bf16)
                            && utils::one_of(bias_dt, f32, bf16, undef));
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_primitive;
};

struct ref_inner_product_bwd_data_t : public gpu::generic::sycl::primitive_t {

    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        using gpu_inner_product_bwd_data_pd_t::gpu_inner_product_bwd_data_pd_t;
        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_bwd_data_t);

        status_t init(impl::engine_t *engine) {
            auto src_dt = arg_md(DNNL_ARG_DIFF_DST)->data_type;
            auto weights_dt = arg_md(DNNL_ARG_WEIGHTS)->data_type;
            auto dst_dt = arg_md(DNNL_ARG_DIFF_SRC)->data_type;

            bool ok = !is_fwd() && (set_default_params() == status::success)
                    && check_bwd_data_dtypes(src_dt, dst_dt, weights_dt)
                    && attr()->has_default_values() // no post-op is supported
                    && memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_DST)).is_plain()
                    && memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_SRC))
                               .is_plain(); // Blocked memory formats are not supported
            if (not ok) { return status::unimplemented; }

            // dL/dX = (dL/dY) x W (hence no transpose required here)
            auto empty_bias_desc = types::
                    zero_md(); // empty memory descriptor to signify bias is not applied

            // Temporary memory descriptors to initialize matmul_pd; diff_dst will always be 2D
            memory_desc_t reshaped_diff_src_md;
            memory_desc_t reshaped_weights_md;
            CHECK(detail::flatten_md(*arg_md(DNNL_ARG_DIFF_SRC),
                    reshaped_diff_src_md, format_tag::ab));
            CHECK(detail::flatten_md(*arg_md(DNNL_ARG_WEIGHTS),
                    reshaped_weights_md, format_tag::ab));

            CHECK(gpu::generic::sycl::detail::init_matmul_pd(engine, attr(),
                    arg_md(DNNL_ARG_DIFF_DST), &reshaped_weights_md,
                    &empty_bias_desc, &reshaped_diff_src_md, matmul_pd));

            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd->scratchpad_registry());
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd;

    private:
        bool check_bwd_data_dtypes(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &weight_dt) {
            using namespace data_type;
            return (utils::one_of(src_dt, f32)
                           && utils::one_of(dst_dt, f32, f16, bf16)
                           && utils::one_of(weight_dt, f32, bf16, f16))
                    || (utils::one_of(src_dt, bf16)
                            && utils::one_of(dst_dt, bf16)
                            && utils::one_of(weight_dt, bf16))
                    || (utils::one_of(src_dt, f16) && utils::one_of(dst_dt, f16)
                            && utils::one_of(weight_dt, f16));
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_primitive;
};

struct ref_inner_product_bwd_weights_t
    : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_bwd_weights_pd_t {
        using gpu_inner_product_bwd_weights_pd_t::
                gpu_inner_product_bwd_weights_pd_t;

        status_t init(engine_t *engine) {
            auto src_dt = arg_md(DNNL_ARG_DIFF_DST)->data_type;
            auto weights_dt = arg_md(DNNL_ARG_SRC)->data_type;
            auto dst_dt = arg_md(DNNL_ARG_DIFF_WEIGHTS)->data_type;
            auto bias_dt = arg_md(DNNL_ARG_DIFF_BIAS)->data_type;

            bool ok = !is_fwd() && (set_default_params() == status::success)
                    && check_bwd_weights_dtypes(
                            src_dt, dst_dt, weights_dt, bias_dt)
                    && memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_DST)).is_plain()
                    && memory_desc_wrapper(arg_md(DNNL_ARG_SRC)).is_plain()
                    && memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_WEIGHTS))
                               .is_plain();
            //Usage of post ops does not influence dL/dW and dL/dB;

            if (not ok) { return status::unimplemented; };

            memory_desc_t reshaped_src_md;
            memory_desc_t reshaped_diff_wt_md;
            memory_desc_t reshaped_diff_dst_md;
            auto empty_bias_desc = types::
                    zero_md(); // empty memory descriptor to signify bias is not applied
            // (dL / dW) = (dL/dY) ^ T x X;
            CHECK(detail::flatten_md(
                    *arg_md(DNNL_ARG_SRC), reshaped_src_md, format_tag::ab));
            CHECK(detail::flatten_md(*arg_md(DNNL_ARG_DIFF_DST),
                    reshaped_diff_dst_md, format_tag::ba));
            CHECK(detail::flatten_md(*arg_md(DNNL_ARG_DIFF_WEIGHTS),
                    reshaped_diff_wt_md, format_tag::ab));

            // Create matmul_pd for dL/dW
            CHECK(detail::init_matmul_pd(engine, attr(), &reshaped_diff_dst_md,
                    &reshaped_src_md, &empty_bias_desc, &reshaped_diff_wt_md,
                    matmul_pd));
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested_multiple,
                    matmul_pd->scratchpad_registry());

            //Create reduction_pd for dL/dB
            if (with_bias()) {
                CHECK(init_reduction_pd(engine, arg_md(DNNL_ARG_DIFF_DST),
                        arg_md(DNNL_ARG_DIFF_BIAS)));
                // book scratchpad for reduction
                scratchpad.book(memory_tracking::names::key_nested_multiple + 1,
                        reduction_pd->scratchpad_registry());
            }
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd;
        std::shared_ptr<primitive_desc_t> reduction_pd;

    private:
        bool check_bwd_weights_dtypes(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &weight_dt,
                const data_type_t &bias_dt) {
            using namespace data_type;
            return (utils::one_of(src_dt, f32) && utils::one_of(dst_dt, f32)
                           && utils::one_of(weight_dt, f32)
                           && utils::one_of(bias_dt, f32, undef))
                    || (utils::one_of(src_dt, bf16)
                            && utils::one_of(dst_dt, bf16)
                            && utils::one_of(weight_dt, f32, bf16)
                            && utils::one_of(bias_dt, f32, bf16, undef))
                    || (utils::one_of(src_dt, f16) && utils::one_of(dst_dt, f16)
                            && utils::one_of(weight_dt, f32, f16)
                            && utils::one_of(bias_dt, f32, f16, undef));
        }

        status_t init_reduction_pd(engine_t *engine,
                const memory_desc_t *src_desc, const memory_desc_t *dest_desc) {
            reduction_desc_t reduction_descriptor;
            CHECK(reduction_desc_init(&reduction_descriptor,
                    alg_kind::reduction_sum, src_desc, dest_desc, 0.0f, 0.0f));
            primitive_desc_iterator_t it(engine,
                    reinterpret_cast<op_desc_t *>(&reduction_pd), attr(),
                    nullptr);

            if (!it.is_initialized()) return status::invalid_arguments;
            while (++it != it.end()) {
                if (*it) {
                    reduction_pd = *it;
                    break;
                }
            }
            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_primitive;
    std::shared_ptr<impl::primitive_t> reduction_primitive;
};

} // namespace dnnl::impl::gpu::generic::sycl

#endif
