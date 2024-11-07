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

#include "gpu/generic/sycl/ref_matmul.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl::impl::gpu::generic::sycl {
struct ref_inner_product_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto src_dt = src_md()->data_type;
            auto weights_dt = weights_md(0)->data_type;
            auto dst_dt = dst_md()->data_type;
            auto bias_dt = weights_md(1)->data_type;
            const bool ok = (set_default_params() == status::success)
                    && is_fwd()
                    && check_if_dtypes_valid(
                            src_dt, dst_dt, bias_dt, weights_dt)
                    && sycl_post_ops_t::post_ops_ok(attr());
            if (not ok) { return status::unimplemented; }
            return init_matmul(engine);
        }

        std::shared_ptr<primitive_desc_t> matmul_pd;

    private:
        bool check_if_dtypes_valid(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &bias_dtype,
                const data_type_t &weight_dtype) const {
            using namespace data_type;
            return (utils::one_of(src_dt, f32)
                           && utils::one_of(weight_dtype, f32)
                           && utils::one_of(dst_dt, f32)
                           && utils::one_of(bias_dtype, f32))
                    || (utils::one_of(src_dt, f16)
                            && utils::one_of(weight_dtype, f16)
                            && utils::one_of(dst_dt, f16, f32, s8, u8)
                            && utils::one_of(bias_dtype, f16, f32))
                    || (utils::one_of(src_dt, u8, s8)
                            && utils::one_of(weight_dtype, s8)
                            && utils::one_of(dst_dt, u8, s8, s32, bf16, f32)
                            && utils::one_of(
                                    bias_dtype, u8, s8, s32, bf16, f32))
                    || (utils::one_of(src_dt, bf16)
                            && utils::one_of(weight_dtype, bf16)
                            && utils::one_of(dst_dt, f32, bf16)
                            && utils::one_of(bias_dtype, f32, f16));
        }

        status_t init_matmul(impl::engine_t *engine);
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
    std::shared_ptr<impl::primitive_t> matmul_primitive;
};
} // namespace dnnl::impl::gpu::generic::sycl

#endif
