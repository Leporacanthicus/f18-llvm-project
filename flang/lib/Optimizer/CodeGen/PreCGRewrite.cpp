//===-- PreCGRewrite.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace fir;

namespace {

/// Convert fir.embox to the extended form where necessary.
class EmboxConversion : public mlir::OpRewritePattern<EmboxOp> {
public:
  using OpRewritePattern::OpRewritePattern;


  mlir::LogicalResult
  matchAndRewrite(EmboxOp embox,
                  mlir::PatternRewriter &rewriter) const override {
    auto shapeVal = embox.getShape();
    // If the embox does not include a shape, then do not convert it
    if (shapeVal)
      return rewriteDynamicShape(embox, rewriter, shapeVal);
    if (auto boxTy = embox.getType().dyn_cast<BoxType>())
      if (auto seqTy = boxTy.getEleTy().dyn_cast<SequenceType>())
        if (seqTy.hasConstantShape())
          return rewriteStaticShape(embox, rewriter, seqTy);
    return mlir::failure();
  }

  mlir::LogicalResult rewriteStaticShape(EmboxOp embox,
                                         mlir::PatternRewriter &rewriter,
                                         SequenceType seqTy) const {
    auto loc = embox.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    auto idxTy = rewriter.getIndexType();
    for (auto ext : seqTy.getShape()) {
      auto iAttr = rewriter.getIndexAttr(ext);
      auto extVal = rewriter.create<mlir::ConstantOp>(loc, idxTy, iAttr);
      shapeOpers.push_back(extVal);
    }
    auto xbox = rewriter.create<cg::XEmboxOp>(
        loc, embox.getType(), embox.memref(), shapeOpers, llvm::None,
        llvm::None, llvm::None, embox.typeparams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }

  mlir::LogicalResult rewriteDynamicShape(EmboxOp embox,
                                          mlir::PatternRewriter &rewriter,
                                          mlir::Value shapeVal) const {
    auto loc = embox.getLoc();
    auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp());
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (shapeOp) {
      populateShape(shapeOpers, shapeOp);
    } else {
      auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp());
      assert(shiftOp && "shape is neither fir.shape nor fir.shape_shift");
      populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    if (auto s = embox.getSlice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }
    auto xbox = rewriter.create<cg::XEmboxOp>(
        loc, embox.getType(), embox.memref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, embox.typeparams());
    LLVM_DEBUG(llvm::dbgs() << "rewriting " << embox << " to " << xbox << '\n');
    rewriter.replaceOp(embox, xbox.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert all fir.array_coor to the extended form.
class ArrayCoorConversion : public mlir::OpRewritePattern<ArrayCoorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCoorOp arrCoor,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = arrCoor.getLoc();
    llvm::SmallVector<mlir::Value> shapeOpers;
    llvm::SmallVector<mlir::Value> shiftOpers;
    if (auto shapeVal = arrCoor.shape()) {
      if (auto shapeOp = dyn_cast<ShapeOp>(shapeVal.getDefiningOp()))
        populateShape(shapeOpers, shapeOp);
      else if (auto shiftOp = dyn_cast<ShapeShiftOp>(shapeVal.getDefiningOp()))
        populateShapeAndShift(shapeOpers, shiftOpers, shiftOp);
      else if (auto shiftOp = dyn_cast<ShiftOp>(shapeVal.getDefiningOp()))
        populateShift(shiftOpers, shiftOp);
      else
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> sliceOpers;
    llvm::SmallVector<mlir::Value> subcompOpers;
    if (auto s = arrCoor.slice())
      if (auto sliceOp = dyn_cast_or_null<SliceOp>(s.getDefiningOp())) {
        sliceOpers.append(sliceOp.triples().begin(), sliceOp.triples().end());
        subcompOpers.append(sliceOp.fields().begin(), sliceOp.fields().end());
      }
    auto xArrCoor = rewriter.create<cg::XArrayCoorOp>(
        loc, arrCoor.getType(), arrCoor.memref(), shapeOpers, shiftOpers,
        sliceOpers, subcompOpers, arrCoor.indices(), arrCoor.typeparams());
    LLVM_DEBUG(llvm::dbgs()
               << "rewriting " << arrCoor << " to " << xArrCoor << '\n');
    rewriter.replaceOp(arrCoor, xArrCoor.getOperation()->getResults());
    return mlir::success();
  }
};

/// Convert FIR structured control flow ops to CFG ops.
class CodeGenRewrite : public CodeGenRewriteBase<CodeGenRewrite> {
public:
  void runOnFunction() override final {
    auto &context = getContext();
    mlir::OwningRewritePatternList patterns(&context);
    patterns.insert<EmboxConversion, ArrayCoorConversion>(&context);
    mlir::ConversionTarget target(context);
    target.addLegalDialect<FIROpsDialect, mlir::StandardOpsDialect>();
    target.addIllegalOp<ArrayCoorOp>();
    target.addDynamicallyLegalOp<EmboxOp>(
        [](EmboxOp embox) { return !embox.getDims(); });

    // Do the conversions.
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(&context),
                      "error in running the pre-codegen conversions");
      signalPassFailure();
    }

    // Erase any residual.
    simplifyRegion(getFunction().getBody());
  }

  // Clean up the region.
  void simplifyRegion(mlir::Region &region) {
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions() != 0)
          for (auto &reg : op.getRegions())
            simplifyRegion(reg);
        maybeEraseOp(&op);
      }

    for (auto *op : opsToErase)
      op->erase();
    opsToErase.clear();
  }

  void maybeEraseOp(mlir::Operation *op) {
    // Erase any embox that was replaced.
    if (auto embox = dyn_cast_or_null<EmboxOp>(op))
      if (embox.getDims()) {
        assert(op->use_empty());
        opsToErase.push_back(op);
      }

    // Erase all fir.array_coor.
    if (auto arrCoor = dyn_cast_or_null<ArrayCoorOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }

    // Erase all fir.gendims ops.
    if (auto genDims = dyn_cast_or_null<GenDimsOp>(op)) {
      assert(op->use_empty());
      opsToErase.push_back(op);
    }
  }

private:
  std::vector<mlir::Operation *> opsToErase;
};

} // namespace

/// Convert FIR's structured control flow ops to CFG ops.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> fir::createFirCodeGenRewritePass() {
  return std::make_unique<CodeGenRewrite>();
}
