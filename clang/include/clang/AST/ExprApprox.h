//===--- ExprApprox.h - Classes for representing Approx ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ApproxExpr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPRAPPROX_H
#define LLVM_CLANG_AST_EXPRAPPROX_H

#include "clang/AST/ComputeDependence.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Debug.h"

namespace clang {

class ApproxSliceExpr : public Expr {
  enum { START, STOP, STEP, END_EXPR };
  Stmt *SubExprs[END_EXPR];
  SourceLocation ColonLocFirst;
  SourceLocation ColonLocSecond;
  SourceLocation LBracketLoc;
  SourceLocation RBracketLoc;

public:
  ApproxSliceExpr(Expr *Start, Expr *Stop, Expr *Step, QualType Type,
                  ExprValueKind VK, ExprObjectKind OK,
                  SourceLocation LBracketLoc, SourceLocation ColonLocFirst,
                  SourceLocation ColonLocSecond, SourceLocation RBracketLoc)
      : Expr(ApproxSliceExprClass, Type, VK, OK), ColonLocFirst(ColonLocFirst),
        ColonLocSecond(ColonLocSecond), LBracketLoc(LBracketLoc),
        RBracketLoc(RBracketLoc) {
    SubExprs[START] = Start;
    SubExprs[STOP] = Stop;
    SubExprs[STEP] = Step;

    setDependence(computeDependence(this));
  }

  explicit ApproxSliceExpr(EmptyShell Empty)
      : Expr(ApproxSliceExprClass, Empty) {}  


  Expr *getStart() { return cast_or_null<Expr>(SubExprs[START]); }
  const Expr *getStart() const { return cast_or_null<Expr>(SubExprs[START]); }
  void setStart(Expr *E) { SubExprs[START] = E; }

  Expr *getStop() { return cast_or_null<Expr>(SubExprs[STOP]); }
  const Expr *getStop() const { return cast_or_null<Expr>(SubExprs[STOP]); }
  void setStop(Expr *E) { SubExprs[STOP] = E; }

  Expr *getStep() { return cast_or_null<Expr>(SubExprs[STEP]); }
  const Expr *getStep() const { return cast_or_null<Expr>(SubExprs[STEP]); }
  void setStep(Expr *E) { SubExprs[STEP] = E; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getStart()->getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY { return RBracketLoc; }

  SourceLocation getColonLocFirst() const { return ColonLocFirst; }
  void setColonLocFirst(SourceLocation L) { ColonLocFirst = L; }
  SourceLocation getColonLocSecond() const { return ColonLocSecond; }
  void setColonLocSecond(SourceLocation L) { ColonLocSecond = L; }
  SourceLocation getLBracketLoc() const { return LBracketLoc; }
  void setLBracketLoc(SourceLocation L) { LBracketLoc = L; }
  SourceLocation getRBracketLoc() const { return RBracketLoc; }
  void setRBracketLoc(SourceLocation L) { RBracketLoc = L; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getStart()->getExprLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ApproxSliceExprClass;
  }

  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[END_EXPR]);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[0], &SubExprs[END_EXPR]);
  }
};

class ApproxArraySliceExpr final
: public Expr,
private llvm::TrailingObjects<ApproxArraySliceExpr, Expr*> {
  friend TrailingObjects;
  unsigned numDims = 0;
  SourceLocation RBracketLoc;

    ApproxArraySliceExpr(Expr *Base, llvm::ArrayRef<Expr *> DSlices,
                         QualType Type, ExprValueKind VK, ExprObjectKind OK,
                         SourceLocation RBLoc)
        : Expr(ApproxArraySliceExprClass, Type, VK, OK), RBracketLoc{RBLoc} {
      numDims = DSlices.size();
      setBase(Base);
      setDimensionSlices(DSlices);
    setDependence(computeDependence(this));
    }

  public:
  explicit ApproxArraySliceExpr(EmptyShell Empty)
      : Expr(ApproxArraySliceExprClass, Empty) {}

  static ApproxArraySliceExpr *Create(const ASTContext &C, Expr *Base,
                                      llvm::ArrayRef<Expr *> DSlices,
                                      QualType Type, SourceLocation RBLoc) {
    void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(1 + DSlices.size()),
                           alignof(ApproxArraySliceExpr));
    return new (Mem) ApproxArraySliceExpr(Base, DSlices, Type, VK_LValue,
                                          OK_Ordinary, RBLoc);
  }

  const Expr *getBase() const { return getTrailingObjects<Expr *>()[0];}
  Expr *getBase() { return getTrailingObjects<Expr *>()[0];}

  QualType getBaseOriginalType(const Expr *Base);
  void setBase(Expr *E) { getTrailingObjects<Expr *>()[0] = E;}
  void setDimensionSlices(llvm::ArrayRef<Expr *> DSlices) {
    assert(DSlices.size() == numDims && "Wrong number of dimension slices");
    llvm::copy(DSlices, getTrailingObjects<Expr *>() + 1);
  }

  unsigned getNumDimensionSlices() const { return numDims; }
  void setNumDimensionSlices(unsigned N) { numDims = N; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getTrailingObjects<Expr *>()[0]->getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY {return RBracketLoc;}
  void setEndLoc(SourceLocation L) { RBracketLoc = L; }

  unsigned numTrailingObjects(OverloadToken<Expr *>) const { return numDims + 1; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getTrailingObjects<Expr *>()[0]->getBeginLoc();
  }

  ArrayRef<Expr *> getSlices() { return llvm::ArrayRef(getTrailingObjects<Expr *>() + 1, numDims); }

  child_range children() {
    Stmt **Begin = reinterpret_cast<Stmt **>(getTrailingObjects<Expr *>());
    return child_range(Begin, Begin + numDims + 1);
  }

  const_child_range children() const { 
    Stmt *const *Begin = reinterpret_cast<Stmt *const *>(getTrailingObjects<Expr *>());
    return const_child_range(Begin, Begin + numDims + 1);
  }
};

class ApproxIndexVarRefExpr : public Expr {
  IdentifierInfo *Identifier;
  SourceLocation Loc;

  public:
  ApproxIndexVarRefExpr(IdentifierInfo *II, QualType Type, ExprValueKind VK,
                      ExprObjectKind OK, SourceLocation Loc)
      : Expr(ApproxIndexVarRefExprClass, Type, VK, OK), Loc(Loc) {
    assert(II && "No identifier provided!");
    Identifier = II;
    setDependence(computeDependence(this));
    }

    explicit ApproxIndexVarRefExpr(EmptyShell Shell)
        : Expr(ApproxIndexVarRefExprClass, Shell) {}

    child_range children() { return child_range(child_iterator(), child_iterator()); }
    const_child_range children() const { return const_child_range(const_child_iterator(), const_child_iterator()); }

    SourceLocation getBeginLoc() const LLVM_READONLY { return Loc; }
    SourceLocation getEndLoc() const LLVM_READONLY { return Loc; }
    SourceLocation getExprLoc() const LLVM_READONLY { return Loc; }

    void setBeginLoc(SourceLocation L) { Loc = L; }
    void setEndLoc(SourceLocation L) { Loc = L; }
    void setExprLoc(SourceLocation L) { Loc = L; }

    void setIdentifierInfo(IdentifierInfo *II) { Identifier = II; }
    IdentifierInfo *getIdentifier() const { return Identifier; }

    llvm::StringRef getName() const { return Identifier->getName(); }


    static bool classof(const Stmt *T) {
      return T->getStmtClass() == ApproxIndexVarRefExprClass;
    }
  };

class ApproxArraySectionExpr : public Expr {
  enum { BASE, LOWER_BOUND, LENGTH, END_EXPR };
  Stmt *SubExprs[END_EXPR];
  SourceLocation ColonLoc;
  SourceLocation RBracketLoc;

public:
  ApproxArraySectionExpr(Expr *Base, Expr *LowerBound, Expr *Length, QualType Type,
                      ExprValueKind VK, ExprObjectKind OK,
                      SourceLocation ColonLoc, SourceLocation RBracketLoc)
      : Expr(ApproxArraySectionExprClass, Type, VK, OK), ColonLoc(ColonLoc),
        RBracketLoc(RBracketLoc) {
    SubExprs[BASE] = Base;
    SubExprs[LOWER_BOUND] = LowerBound;
    SubExprs[LENGTH] = Length;
    setDependence(computeDependence(this));
  }

  /// Create an empty array section expression.
  explicit ApproxArraySectionExpr(EmptyShell Shell)
      : Expr(ApproxArraySectionExprClass, Shell) {}

  /// An array section can be written only as Base[LowerBound:Length].

  /// Get base of the array section.
  Expr *getBase() { return cast<Expr>(SubExprs[BASE]); }
  const Expr *getBase() const { return cast<Expr>(SubExprs[BASE]); }
  /// Set base of the array section.
  void setBase(Expr *E) { SubExprs[BASE] = E; }

  /// Return original type of the base expression for array section.
  static QualType getBaseOriginalType(const Expr *Base);

  /// Get lower bound of array section.
  Expr *getLowerBound() { return cast_or_null<Expr>(SubExprs[LOWER_BOUND]); }
  const Expr *getLowerBound() const {
    return cast_or_null<Expr>(SubExprs[LOWER_BOUND]);
  }
  /// Set lower bound of the array section.
  void setLowerBound(Expr *E) { SubExprs[LOWER_BOUND] = E; }

  /// Get length of array section.
  Expr *getLength() { return cast_or_null<Expr>(SubExprs[LENGTH]); }
  const Expr *getLength() const { return cast_or_null<Expr>(SubExprs[LENGTH]); }
  /// Set length of the array section.
  void setLength(Expr *E) { SubExprs[LENGTH] = E; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getBase()->getBeginLoc();
  }
  SourceLocation getEndLoc() const LLVM_READONLY { return RBracketLoc; }

  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  SourceLocation getRBracketLoc() const { return RBracketLoc; }
  void setRBracketLoc(SourceLocation L) { RBracketLoc = L; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getBase()->getExprLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ApproxArraySectionExprClass;
  }

  child_range children() {
    return child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }
};

} // namespace clang

#endif
