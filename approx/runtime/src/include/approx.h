 //===--- approx.h - Approx public API ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines the public accessible API to the approximate runtime system.
///
//===----------------------------------------------------------------------===//
#ifndef __APPROX_INC__
#define __APPROX_INC__
#include <stdint.h>
#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef enum __aprox_datatype_hdf5{
    HUINT8 = 0,
    HINT8,
    HINT, 
    HFLOAT,
    HDOUBLE
}HPACDType;

bool __approx_skip_iteration(unsigned int i, float pr);
void __approx_exec_call(void (*accurateFN)(void *), void (*perfoFN)(void *),
                        void *arg, bool cond, const char *region_name,
                        void *perfoArgs, int memo_type, int petru_type, 
                        int sp_type, int ml_type, void *inputs, int num_inputs, 
                        void *outputs, int num_outputs);
const float approx_rt_get_percentage();
const int approx_rt_get_step();

extern float __approx_perfo_rate__;
extern int __approx_perfo_step__;

void HPACRegisterApplicationInput(void *ptr, size_t numBytes, const char *name, HPACDType dType);
void HPACRegisterApplicationOutput(void *ptr, size_t numBytes, const char *name, HPACDType dType);

#ifdef __cplusplus
}
#endif
#endif