//===-- Assign.cpp -- generate assignment runtime API calls ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Runtime/Assign.h"
#include "../../../runtime/assign.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Runtime/RTBuilder.h"

using namespace Fortran::runtime;

void fir::runtime::genAssign(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value destBox, mlir::Value sourceBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Assign)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}
