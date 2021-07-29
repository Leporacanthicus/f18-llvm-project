//===-- MaskExpr.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_MASKEXPR_H
#define FORTRAN_LOWER_MASKEXPR_H

#include "StatementContext.h"
#include "flang/Lower/FIRBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace llvm {
class raw_ostream;
}

namespace Fortran {
namespace evaluate {
struct SomeType;
template <typename>
class Expr;
} // namespace evaluate

namespace lower {

using MaskAddrAndShape = std::pair<mlir::Value, mlir::Value>;
using FrontEndExpr = const evaluate::Expr<evaluate::SomeType> *;

template <typename A>
class StackableConstructExpr {
public:
  bool empty() const { return stack.empty(); }

  void growStack() {
    if (empty())
      stmtCtx.reset();
    stack.push_back(A{});
  }

  void shrinkStack() {
    assert(!empty());
    stack.pop_back();
    if (empty())
      stmtCtx.finalize();
  }

  void bind(FrontEndExpr e, mlir::Value v, mlir::Value shape) {
    vmap.try_emplace(e, v, shape);
  }
  void bind(FrontEndExpr e, const MaskAddrAndShape &p) { vmap.insert({e, p}); }
  mlir::Value getBinding(FrontEndExpr e) const {
    return getBindingWithShape(e).first;
  }
  MaskAddrAndShape getBindingWithShape(FrontEndExpr e) const {
    assert(vmap.count(e) && "key not already in map");
    return vmap.lookup(e);
  }
  bool isLowered(FrontEndExpr e) const { return vmap.count(e); }

  StatementContext &stmtContext() { return stmtCtx; }

protected:
  // The stack for the construct information.
  llvm::SmallVector<A> stack;

  // Map each mask expression back to the temporary holding the initial
  // evaluation results.
  llvm::DenseMap<FrontEndExpr, MaskAddrAndShape> vmap;

  // Inflate the statement context for the entire construct. We have to cache
  // the mask expression results, which are always evaluated first, across the
  // entire construct.
  StatementContext stmtCtx;
};

class MaskExpr;
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const MaskExpr &);

/// Collect WHERE construct mask expressions in the bridge and forward lowering
/// results in an "evaluate once" semantics. See 10.2.3.2p3, 10.2.3.2p13, etc.
class MaskExpr
    : public StackableConstructExpr<llvm::SmallVector<FrontEndExpr>> {
public:
  using FrontEndMaskExpr = FrontEndExpr;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const MaskExpr &);

  LLVM_DUMP_METHOD void dump() const;

  void append(FrontEndMaskExpr e) {
    assert(!empty());
    getMasks().back().push_back(e);
  }

  llvm::SmallVector<FrontEndMaskExpr> getExprs() const {
    auto maskList = getMasks()[0];
    for (unsigned i = 1, d = getMasks().size(); i < d; ++i)
      maskList.append(getMasks()[i].begin(), getMasks()[i].end());
    return maskList;
  }

private:
  // Stack of WHERE constructs, each building a list of mask expressions.
  llvm::SmallVector<llvm::SmallVector<FrontEndMaskExpr>> &getMasks() {
    return stack;
  }
  const llvm::SmallVector<llvm::SmallVector<FrontEndMaskExpr>> &
  getMasks() const {
    return stack;
  }
};

class IterationSpaceExpr;
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const IterationSpaceExpr &);

/// Capture a stack of lists of concurrent-control expressions to be used to
/// generate the iteration space for a set of nested FORALL constructs.
class IterationSpaceExpr
    : public StackableConstructExpr<std::pair<
          llvm::SmallVector<std::tuple<const semantics::Symbol *, FrontEndExpr,
                                       FrontEndExpr, FrontEndExpr>>,
          FrontEndExpr>> {
public:
  using FrontEndSymbol = const semantics::Symbol *;
  using IterSpaceDim =
      std::tuple<FrontEndSymbol, FrontEndExpr, FrontEndExpr, FrontEndExpr>;
  using ConcurrentSpec =
      std::pair<llvm::SmallVector<IterSpaceDim>, FrontEndExpr>;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const IterationSpaceExpr &);

  LLVM_DUMP_METHOD void dump() const;

  /// Append a new dimension to the set of iteration spaces. If `step` is a
  /// nullptr, then the step of the loop is the constant 1.
  void emplace_back(FrontEndSymbol sym, FrontEndExpr lo, FrontEndExpr hi,
                    FrontEndExpr step) {
    assert(!empty());
    dims().back().first.emplace_back(sym, lo, hi, step);
  }

  void setMaskExpr(FrontEndExpr mask) {
    assert(!dims().back().second);
    dims().back().second = mask;
  }

  /// Get a list of all curently active FORALL concurrent control expressions.
  llvm::SmallVector<IterSpaceDim> getDims() const {
    llvm::SmallVector<IterSpaceDim> dimList = dims()[0].first;
    for (unsigned i = 1, d = dims().size(); i < d; ++i)
      dimList.append(dims()[i].first.begin(), dims()[i].first.end());
    return dimList;
  }

  /// Get a list of all currently active FORALL mask expressions.
  llvm::SmallVector<FrontEndExpr> getMasks() const {
    llvm::SmallVector<FrontEndExpr> res;
    for (auto &pr : dims())
      if (pr.second)
        res.push_back(pr.second);
    return res;
  }

  const llvm::SmallVector<ConcurrentSpec> &getSpecs() const { return dims(); }

  /// Return the total number of explicit controls on this stack.
  std::size_t size() const {
    std::size_t size = 0;
    for (auto &v : dims())
      size += v.first.size();
    return size;
  }

  StatementContext &markInnerContext() {
    if (!innerStmtCtx)
      innerStmtCtx = new StatementContext;
    return *innerStmtCtx;
  }

  void freeInnerContext() {
    if (innerStmtCtx)
      delete innerStmtCtx;
    innerStmtCtx = nullptr;
  }

private:
  llvm::SmallVector<ConcurrentSpec> &dims() { return stack; }
  const llvm::SmallVector<ConcurrentSpec> &dims() const { return stack; }

  StatementContext *innerStmtCtx = nullptr;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_MASKEXPR_H
