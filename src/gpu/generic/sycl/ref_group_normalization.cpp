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

#include "gpu/generic/sycl/ref_group_normalization.hpp"

namespace dnnl::impl::gpu::generic::sycl {
status_t ref_group_normalization_fwd_t::pd_t::init(impl::engine_t *engine) {
    bool ok = set_default_formats_common();
    if (not ok) { return status::unimplemented; }

    auto src_mdw = memory_desc_wrapper(arg_md(DNNL_ARG_SRC));
    auto dst_mdw = memory_desc_wrapper(arg_md(DNNL_ARG_DST));

    auto prop_kind_ = desc()->prop_kind;
    if (not utils::one_of(prop_kind_, prop_kind::forward_training,
                prop_kind::forward_inference)) {
        return status::invalid_arguments;
    }

    auto src_dt = src_mdw.data_type();
    auto dst_dt = dst_mdw.data_type();
    bool src_dt_supported = utils::one_of(src_dt, data_type::f32,
            data_type::f16, data_type::bf16, data_type::s8);
    bool dst_dt_supported = utils::one_of(dst_dt, data_type::f32,
            data_type::f16, data_type::bf16, data_type::s8);
    if (not(src_dt_supported && dst_dt_supported)) {
        return status::invalid_arguments;
    }
    const auto &dims = src_mdw.dims();
    const auto num_groups = desc()->groups;

    if (dims[1] % num_groups != 0) { return status::invalid_arguments; }

    auto batch_size = static_cast<std::size_t>(dims[0]);
    auto group_range = static_cast<std::size_t>(desc()->groups);
    // using fully qualified namespace for clarity
    auto device
            = dynamic_cast<dnnl::impl::gpu::generic::sycl::engine_t *>(engine)
                      ->device();
    // To avoid using excess registers error
    auto local_range = std::max(std::size_t(64),
            device.get_info<::sycl::info::device::max_work_group_size>() / 8);
    launch_range = ::sycl::nd_range<2>(
            {batch_size * local_range, group_range}, {local_range, 1});
    conf_ = sycl_group_norm_conf_t {desc()->prop_kind,
            xpu::sycl::md_t(arg_md(DNNL_ARG_SRC)),
            xpu::sycl::md_t(arg_md(DNNL_ARG_DST)), stats_is_src(),
            static_cast<int32_t>(group_range),
            static_cast<int32_t>(dims[1] / group_range), use_scale(),
            use_shift(), desc()->group_norm_epsilon};
    return status::success;
}

status_t ref_group_normalization_fwd_t::init(impl::engine_t *engine) {
    auto kid = ::sycl::get_kernel_id<group_norm_fwd_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_group_normalization_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto conf_ = pd()->conf_;
    auto launch_range = pd()->launch_range;

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        ::sycl::local_accessor<float, 1> local_memory(
                launch_range.get_local_range()[0] + 2, cgh);
        cgh.parallel_for(
                launch_range, group_norm_fwd_t(conf_, local_memory, cgh, ctx));
    });
    return status::success;
}

} // namespace dnnl::impl::gpu::generic::sycl