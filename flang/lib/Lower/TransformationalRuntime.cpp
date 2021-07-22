//===-- TransformationalRuntime.cpp -- runtime for transformational intrinsics//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/TransformationalRuntime.h"
#include "../../runtime/matmul.h"
#include "../../runtime/transformational.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Generate call to Cshift intrinsic
void Fortran::lower::genCshift(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value arrayBox, mlir::Value shiftBox,
                               mlir::Value dimBox) {
  auto cshiftFunc =
      Fortran::lower::getRuntimeFunc<mkRTKey(Cshift)>(loc, builder);
  auto fTy = cshiftFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      Fortran::lower::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                      shiftBox, dimBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, cshiftFunc, args);
}

/// Generate call to the vector version of the Cshift intrinsic
void Fortran::lower::genCshiftVector(Fortran::lower::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value resultBox,
                                     mlir::Value arrayBox,
                                     mlir::Value shiftBox) {
  auto cshiftFunc =
      Fortran::lower::getRuntimeFunc<mkRTKey(CshiftVector)>(loc, builder);
  auto fTy = cshiftFunc.getType();

  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, arrayBox, shiftBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, cshiftFunc, args);
}

/// Generate call to Matmul intrinsic runtime routine.
void Fortran::lower::genMatmul(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value matrixABox, mlir::Value matrixBBox) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Matmul)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      Fortran::lower::createArguments(builder, loc, fTy, resultBox, matrixABox,
                                      matrixBBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Reshape intrinsic runtime routine.
void Fortran::lower::genReshape(Fortran::lower::FirOpBuilder &builder,
                                mlir::Location loc, mlir::Value resultBox,
                                mlir::Value sourceBox, mlir::Value shapeBox,
                                mlir::Value padBox, mlir::Value orderBox) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Reshape)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(6));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, resultBox,
                                              sourceBox, shapeBox, padBox,
                                              orderBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Spread intrinsic runtime routine.
void Fortran::lower::genSpread(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value sourceBox, mlir::Value dim,
                               mlir::Value ncopies) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Spread)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      Fortran::lower::createArguments(builder, loc, fTy, resultBox, sourceBox,
                                      dim, ncopies, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Transpose intrinsic runtime routine.
void Fortran::lower::genTranspose(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value resultBox,
                                  mlir::Value sourceBox) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Transpose)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, sourceBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Unpack intrinsic runtime routine.
void Fortran::lower::genUnpack(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value vectorBox, mlir::Value maskBox,
                               mlir::Value fieldBox) {
  auto unpackFunc =
      Fortran::lower::getRuntimeFunc<mkRTKey(Unpack)>(loc, builder);
  auto fTy = unpackFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, resultBox,
                                              vectorBox, maskBox, fieldBox,
                                              sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, unpackFunc, args);
}
