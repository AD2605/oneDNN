/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef CPU_X64_GEMM_BF16_COMMON_S16_HPP
#define CPU_X64_GEMM_BF16_COMMON_S16_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

class jit_avx512_core_s16_48x8_copy_an_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_48x8_copy_an_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_48x8_copy_an_kern_t();
};

class jit_avx512_core_s16_48x8_copy_at_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_48x8_copy_at_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_48x8_copy_at_kern_t();
};

class jit_avx512_core_s16_48x8_copy_bn_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_48x8_copy_bn_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_48x8_copy_bn_kern_t();
};

class jit_avx512_core_s16_48x8_copy_bt_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_48x8_copy_bt_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_48x8_copy_bt_kern_t();
};

class jit_avx512_core_s16_24x8_copy_an_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_24x8_copy_an_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_24x8_copy_an_kern_t();
};

class jit_avx512_core_s16_24x8_copy_at_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_24x8_copy_at_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_24x8_copy_at_kern_t();
};

class jit_avx512_core_s16_24x8_copy_bn_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_24x8_copy_bn_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_24x8_copy_bn_kern_t();
};

class jit_avx512_core_s16_24x8_copy_bt_kern_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_24x8_copy_bt_kern_t);
    void generate() override;

public:
    jit_avx512_core_s16_24x8_copy_bt_kern_t();
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif // CPU_X64_GEMM_BF16_COMMON_S16_HPP
