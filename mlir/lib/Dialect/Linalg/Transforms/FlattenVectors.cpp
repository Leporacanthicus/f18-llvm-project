//===- FlattenVectors.cpp - Implementation of linalg vectors flattening
//----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Vector Flattening transformations.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_LINALGFLATTENVECTORS

#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-flatten-vectors"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class FlattenRewriter
    : public OpRewritePattern<linalg::DepthwiseConv1DNwcWcOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::DepthwiseConv1DNwcWcOp op,
                                PatternRewriter &rewriter) const override {

    Value inputShaped = op.getDpsInputOperand(0)->get();
    Value filterShaped = op.getDpsInputOperand(1)->get();
    Value outputShaped = op.getDpsInitOperand(0)->get();

    ShapedType inputShapedType = dyn_cast<ShapedType>(inputShaped.getType());
    ShapedType filterShapedType = dyn_cast<ShapedType>(filterShaped.getType());
    ShapedType outputShapedType = dyn_cast<ShapedType>(outputShaped.getType());

    Type lhsEltType = inputShapedType.getElementType();
    Type filterEltType = filterShapedType.getElementType();
    Type outputEltType = outputShapedType.getElementType();

    mlir::Location loc = op.getLoc();

    // kernel{fw, fc}
    int64_t fwSize = filterShapedType.getShape()[0];
    int64_t fcSize = filterShapedType.getShape()[1];

    // out{n, w, c}
    int64_t onSize = outputShapedType.getShape()[0];
    int64_t owSize = outputShapedType.getShape()[1];
    int64_t ocSize = outputShapedType.getShape()[2];

    // input(n, w, c}
    int64_t inSize = inputShapedType.getShape()[0];
    int64_t iwSize = inputShapedType.getShape()[1];
    int64_t icSize = inputShapedType.getShape()[2];

    assert(fcSize == ocSize && icSize == ocSize && "Channel size should match");
    assert(onSize == inSize && "Batch size should match");

    // TODO: Need to fill these values with something sensible.
    int64_t strideW = 1;
    int64_t dilationW = 1;

    // iw = (ow * sw + kw *  dw - 1) * c
    //   (i.e. 16 convolved with 3 (@stride 1 dilation 1) -> 14)
    int64_t calcWSize =
        (((owSize - 1) * strideW + 1) + ((fwSize - 1) * dilationW + 1) - 1);

    RankedTensorType filterType =
        RankedTensorType::get({fwSize}, filterEltType);
    RankedTensorType inputType = RankedTensorType::get({calcWSize}, lhsEltType);
    RankedTensorType outputType =
        RankedTensorType::get({owSize}, outputEltType);

    RankedTensorType filterSingleType =
        RankedTensorType::get({fwSize, 1}, filterEltType);
    RankedTensorType inputSingleType =
        RankedTensorType::get({1, iwSize, 1}, lhsEltType);
    RankedTensorType outputSingleType =
        RankedTensorType::get({1, owSize, 1}, outputEltType);

    // Extract the channel as a vector:
    // We need constant values for the
    IntegerAttr zero = rewriter.getI64IntegerAttr(0);
    IntegerAttr one = rewriter.getI64IntegerAttr(1);
    IntegerAttr iwVal = rewriter.getI64IntegerAttr(iwSize);
    IntegerAttr fwVal = rewriter.getI64IntegerAttr(fwSize);
    IntegerAttr owVal = rewriter.getI64IntegerAttr(owSize);
    SmallVector<OpFoldResult> inOutStrides = {one, one, one};
    SmallVector<OpFoldResult> filterStrides = {one, one};
    SmallVector<OpFoldResult> inputSizes = {one, iwVal, one};
    SmallVector<OpFoldResult> outputSizes = {one, owVal, one};
    SmallVector<OpFoldResult> filterSizes = {fwVal, one};
    SmallVector<ReassociationIndices> filterReassociation = {{0, 1}};
    SmallVector<ReassociationIndices> inOutReassociation = {{0, 1, 2}};

    Value newOutput = outputShaped;

    for (int64_t batchNo = 0; batchNo < onSize; batchNo++) {
      IntegerAttr niVal = rewriter.getI64IntegerAttr(batchNo);
      for (int64_t channelIndex = 0; channelIndex < ocSize; channelIndex++) {
        IntegerAttr ciVal = rewriter.getI64IntegerAttr(channelIndex);
        SmallVector<OpFoldResult> inOutOffsets = {niVal, zero, ciVal};
        SmallVector<OpFoldResult> filterOffsets = {zero, ciVal};
        Value inputSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, inputSingleType, inputShaped, inOutOffsets, inputSizes,
            inOutStrides);
        Value inputFlat = rewriter.create<tensor::CollapseShapeOp>(
            loc, inputType, inputSlice, inOutReassociation);
        Value filterSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, filterSingleType, filterShaped, filterOffsets, filterSizes,
            filterStrides);
        Value filterFlat = rewriter.create<tensor::CollapseShapeOp>(
            loc, filterType, filterSlice, filterReassociation);

        Value tempOut = rewriter.create<tensor::ExtractSliceOp>(
            loc, outputSingleType, outputShaped, inOutOffsets, outputSizes,
            inOutStrides);

        Value tempFlat = rewriter.create<tensor::CollapseShapeOp>(
            loc, outputType, tempOut, inOutReassociation);

        // Add new operation, which will replace the original convolution.
        Operation *conv1D = rewriter.create<linalg::Conv1DOp>(
            loc, outputType, ValueRange{inputFlat, filterFlat},
            ValueRange{tempFlat});
        Value convOutput = conv1D->getResult(0);
        Value convOutExpanded = rewriter.create<tensor::ExpandShapeOp>(
            loc, outputSingleType, convOutput, inOutReassociation);

        // Re-expand shape for the result
        newOutput = rewriter.create<tensor::InsertSliceOp>(
            loc, convOutExpanded, newOutput, inOutOffsets, outputSizes,
            inOutStrides);
      }
    }
    rewriter.replaceOp(op, newOutput);
    return success();
  }
};
#if 0
#map_scalar = affine_map < (d0, d1, d2)->()>
#map_array = affine_map < (d0, d1, d2)->(d0, d1, d2)>

#map_vec_scalar = affine_map < (d0)->()>
#map_vec_vector = affine_map < (d0)->(d0)>

module attributes {llvm.target_triple = "x86_64-none-linux-gnu"} {
  func.func @decomposed(%arg0: tensor<1x2076x1xi32>, %arg1: tensor<29x1xi32>) -> tensor<1x2048x1xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<1x2048x1xi32>
    %1 = linalg.generic {indexing_maps = [#map_scalar, #map_array], iterator_types = ["parallel", "parallel", "parallel"]} ins(%c0_i32 : i32) outs(%0 : tensor<1x2048x1xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x2048x1xi32>
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c4_index = arith.constant 1 : index
    %output = scf.for %channel_index = %c0_index to %c4_index step %c1_index iter_args(%output=%1) -> tensor<1x2048x1xi32> {
      // Get a 1x2076x1 slice of the input
      %input = tensor.extract_slice %arg0[0, 0, %channel_index][1, 2076, 1][1, 1, 1] : tensor<1x2076x1xi32> to tensor<1x2076x1xi32>
      %input_collapsed = tensor.collapse_shape %input[[0,1,2]] : tensor<1x2076x1xi32> into tensor<2076xi32>
      // Get a 29x1 slice of the filter
      %filter = tensor.extract_slice %arg1[0, %channel_index][29, 1][1, 1]: tensor<29x1xi32> to tensor<29x1xi32>
      %filter_collapsed = tensor.collapse_shape %filter[[0,1]] : tensor<29x1xi32> into tensor<29xi32>
      %te = tensor.extract_slice %output[0, 0, %channel_index][1, 2048, 1][1, 1, 1] : tensor<1x2048x1xi32> to tensor<1x2048x1xi32>
      %te_collapsed = tensor.collapse_shape %te[[0,1,2]] : tensor<1x2048x1xi32> into tensor<2048xi32>
n//      %te = tensor.empty() : tensor<2048xi32>
      %te0 = linalg.generic {indexing_maps = [#map_vec_scalar, #map_vec_vector], iterator_types = ["parallel"]} ins(%c0_i32 : i32) outs(%te_collapsed : tensor<2048xi32>) {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<2048xi32>
      %conv_output = linalg.conv_1d ins(%input_collapsed, %filter_collapsed: tensor<2076xi32>, tensor<29xi32>) outs(%te0 : tensor<2048xi32>) -> tensor<2048xi32>
      %conv_output_expanded = tensor.expand_shape %conv_output[[0, 1, 2]] : tensor<2048xi32> into tensor<1x2048x1xi32>
      %conv_output_inserted = tensor.insert_slice %conv_output_expanded into %output[0, 0, %channel_index][1, 2048, 1][1,1,1] : tensor<1x2048x1xi32> into tensor<1x2048x1xi32>
      scf.yield %conv_output_inserted : tensor<1x2048x1xi32>
    }
    return %output : tensor<1x2048x1xi32>
  }
}
#endif

static bool hasDynamicShape(OpOperand *v) {
  ShapedType t = dyn_cast<ShapedType>(v->get().getType());
  return !t.hasStaticShape();
}

struct LinalgFlattenVectorsPass
    : public impl::LinalgFlattenVectorsBase<LinalgFlattenVectorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    LDBG("Starting the pass");
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.insert<FlattenRewriter>(context);

    target.addDynamicallyLegalOp<linalg::DepthwiseConv1DNwcWcOp>(
        [](linalg::DepthwiseConv1DNwcWcOp op) {
          return llvm::any_of(op.getDpsInputOperands(), hasDynamicShape) ||
                 llvm::any_of(op.getDpsInitOperands(), hasDynamicShape);
        });
    target.addLegalOp<tensor::CollapseShapeOp, tensor::InsertSliceOp,
                      tensor::ExtractSliceOp, tensor::ExpandShapeOp,
                      linalg::Conv1DOp, func::ReturnOp, arith::ConstantOp>();

    mlir::Operation *op = getOperation();
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      LDBG("Failure in the pass");
      return signalPassFailure();
    }
    LDBG("Ending the pass");
  }
};

std::unique_ptr<Pass> mlir::createLinalgFlattenVectors() {
  return std::make_unique<LinalgFlattenVectorsPass>();
}
