//===-- Runtime/ragged.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_RAGGED_H_
#define FORTRAN_RUNTIME_RAGGED_H_

#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {
extern "C" {
// Helper for allocation of ragged array buffer blocks.
void *RTNAME(RaggedArrayAllocate)(
    void *, bool, std::int64_t, std::int64_t, std::int64_t *);
// Helper for deallocation of ragged array buffers.
void RTNAME(RaggedArrayDeallocate)(void *);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_RAGGED_H_
