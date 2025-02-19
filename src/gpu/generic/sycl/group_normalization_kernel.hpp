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

#ifndef GPU_GENERIC_SYCL_GROUP_NORMALIZATION_KERNEL_HPP
#define GPU_GENERIC_SYCL_GROUP_NORMALIZATION_KERNEL_HPP

#include <sycl/nd_item.hpp>

#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"

namespace dnnl::impl::gpu::generic::sycl {
struct group_norm_fwd_t {
    using sycl_dims_t = int32_t[6];

    group_norm_fwd_t(const sycl_group_norm_conf_t &conf_,
            ::sycl::local_accessor<float, 1> &local_memory,
            ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf_)
        , src(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , scale(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SCALE))
        , shift(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SHIFT))
        , dst(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , mean(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_MEAN))
        , variance(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_VARIANCE))
        , local_memory(local_memory) {}

    inline void operator()(::sycl::nd_item<2> it) const {
        auto batch = it.get_group(0);
        auto group_num = it.get_group(1);
        // Only one workgroup gets assigned to each group.
        // Purposefully not implementing optimal reduction / group_norm for the sake of
        // simplicity.

        auto src_wrapper = conf_.src_desc;
        auto dst_wrapper = conf_.dst_desc;
        dim_t num_elements_to_reduce = 1;
        const auto &dims = src_wrapper.dims();
        // As Batch and channels are dim 0 and 1 respectively
        for (int i = 2; i < src_wrapper.ndims(); i++) {
            num_elements_to_reduce *= dims[i];
        }
        num_elements_to_reduce *= conf_.num_channels_per_group;

        // precomputed inclusive scan to get the logical index of the tensor
        dims_t inclusive_scan;
        inclusive_scan[0] = 1;
        // Number of dimensions flattened in addition to the C dimension
        for (int i = 0; i < src_wrapper.ndims() - 2; i++) {
            inclusive_scan[i + 1]
                    = inclusive_scan[i - 1] * dims[src_wrapper.ndims() - i - 1];
        }

        for (auto i = group_num; i < static_cast<std::size_t>(conf_.num_groups);
                i += it.get_group_range(1)) {
            // as accumulation will always be in float.
            float accum = 0;
            float mean_value;
            float std_value;
            if (not conf_.use_global_stats) {
                for (dim_t j = it.get_local_linear_id();
                        j < num_elements_to_reduce;
                        j += static_cast<dim_t>(it.get_local_range(0))) {
                    dims_t logical_index;
                    logical_index[0] = batch;
                    auto index = get_offset(src_wrapper, inclusive_scan, dims,
                            batch, group_num, j);
                    accum += load_float_value(
                            src_wrapper.data_type(), src.get_pointer(), index);
                }

                workgroup_reduce(it, accum, it.get_local_range(0));
                // Divide by total elements to get the mean
                if (it.get_local_linear_id() == 0) {
                    local_memory[it.get_local_range(0)]
                            /= num_elements_to_reduce;
                    store_float_value(data_type::f32,
                            local_memory[it.get_local_range(0)],
                            mean.get_pointer(),
                            batch * conf_.num_groups + group_num);
                }
                ::sycl::group_barrier(it.get_group());
                //write each value to local memory;
                // start accum for standard deviation
                float mean = local_memory[it.get_local_range(0)];
                mean_value = mean;
                accum = 0;
                for (dim_t j = it.get_local_linear_id();
                        j < num_elements_to_reduce;
                        j += static_cast<dim_t>(it.get_local_range(0))) {
                    auto index = get_offset(src_wrapper, inclusive_scan, dims,
                            batch, group_num, j);
                    accum += ::sycl::pown(
                            (load_float_value(src_wrapper.data_type(),
                                     src.get_pointer(), index)
                                    - mean),
                            2);
                }
                workgroup_reduce(it, accum, it.get_local_range(0) + 1);
                // calculate std + eta
                if (it.get_local_linear_id() == 0) {
                    float variance_val
                            = local_memory[it.get_local_linear_id() + 1];
                    variance_val /= num_elements_to_reduce;
                    float std = ::sycl::sqrt(variance_val + conf_.eta);
                    local_memory[it.get_local_linear_id() + 1] = std;
                    store_float_value(data_type::f32, variance_val,
                            variance.get_pointer(),
                            batch * conf_.num_groups + group_num);
                }
                ::sycl::group_barrier(it.get_group());
                std_value = local_memory[it.get_local_linear_id() + 1];
            } else {
                mean_value
                        = load_float_value(data_type::f32, mean.get_pointer(),
                                batch * conf_.num_groups + group_num);
                std_value = load_float_value(data_type::f32,
                        variance.get_pointer(),
                        batch * conf_.num_groups + group_num);
                std_value = ::sycl::sqrt(std_value + conf_.eta);
            }

            //Now start the normalization
            for (dim_t j = it.get_local_linear_id(); j < num_elements_to_reduce;
                    j += static_cast<dim_t>(it.get_local_range(0))) {
                auto index = get_offset(
                        src_wrapper, inclusive_scan, dims, batch, group_num, j);
                float value = load_float_value(
                        src_wrapper.data_type(), src.get_pointer(), index);
                float normalized_value = (value - mean_value) / std_value;
                int32_t channel_value
                        = (j / inclusive_scan[src_wrapper.ndims() - 3])
                        % conf_.num_channels_per_group;
                if (conf_.use_scale) {
                    normalized_value *= load_float_value(data_type::f32,
                            scale.get_pointer(),
                            conf_.num_channels_per_group * group_num
                                    + channel_value);
                }
                if (conf_.use_shift) {
                    normalized_value += load_float_value(data_type::f32,
                            shift.get_pointer(),
                            conf_.num_channels_per_group * group_num
                                    + channel_value);
                }

                auto dst_index = get_offset(
                        dst_wrapper, inclusive_scan, dims, batch, group_num, j);
                store_float_value(dst_wrapper.data_type(), normalized_value,
                        dst.get_pointer(), dst_index);
            }
        }
    }

private:
    inline dim_t get_offset(const xpu::sycl::md_t &wrapper,
            const dims_t &inclusive_scan, const sycl_dims_t &dims, dim_t batch,
            dim_t group_num, dim_t flattened_index) const {

        dims_t logical_index;
        auto num_flattened_dimensions = wrapper.ndims() - 2;
        logical_index[0] = batch;
        // Calculate Channel index
        logical_index[1] = group_num * conf_.num_channels_per_group
                + ((flattened_index
                           / inclusive_scan[num_flattened_dimensions - 1])
                        % conf_.num_channels_per_group);
        for (int i = 0; i < num_flattened_dimensions - 1; i++) {
            //TODO: check this
            logical_index[i + 2]
                    = (flattened_index
                              / inclusive_scan[num_flattened_dimensions - 2
                                      - i])
                    % dims[i + 2];
        }
        return wrapper.off_v(logical_index);
    }

    inline void workgroup_reduce(::sycl::nd_item<2> &it,
            float workitem_accum_value, int32_t idx_to_write_red_value) const {
        local_memory[it.get_local_linear_id()] = workitem_accum_value;

        ::sycl::group_barrier(it.get_group());

        // Group leader accumulates;
        if (it.get_local_linear_id() == 0) {
            float total_sum = 0;
            for (auto i = std::size_t(0); i < it.get_local_range(0); i++) {
                total_sum += local_memory[i];
            }
            local_memory[idx_to_write_red_value] = total_sum;
        }
    }

    sycl_group_norm_conf_t conf_;
    xpu::sycl::in_memory_arg_t src;
    xpu::sycl::in_memory_arg_t scale;
    xpu::sycl::in_memory_arg_t shift;
    xpu::sycl::inout_memory_arg_t dst;
    xpu::sycl::inout_memory_arg_t mean;
    xpu::sycl::inout_memory_arg_t variance;
    ::sycl::local_accessor<float, 1> local_memory;
};
} // namespace dnnl::impl::gpu::generic::sycl

#endif
