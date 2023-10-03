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

template<typename ExprClass>
bool anyChildHasType(const Expr *E) {
  for (const Stmt *SubStmt : E->children()) {
    if (isa<ExprClass>(SubStmt))
      return true;
    return anyChildHasType<ExprClass>(cast<Expr>(SubStmt));
  }
  return false;
}

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

  // we want to know if this slice has
  // any children that contain ApproxIndexVarRefExprs.
  // There are 3 different cases that affect codegen/shape analysis:
  // 1. the AIVRE is standalone, e.g. [i]
  //    In this case, we want to expand the shape to [i,1]
  // 2. Case 2: the AIVRE is part of a binary expression, e.g. [i*3:i*3+3]
  //    In this case, we want to expand the shape to [i,3]
  // 3. Case 3: The slice has no AIVRE. Nothing special happens here.
  enum class AIVREChildKind {
    STANDALONE,
    BINARY_EXPR,
    NONE
  };

  AIVREChildKind AIVREChild = AIVREChildKind::NONE;

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

  AIVREChildKind getAIVREChildKind() const { return AIVREChild; }
  void setAIVREChildKind(AIVREChildKind K) { AIVREChild = K; }

  static AIVREChildKind discoverChildKind(Expr *Start, Expr *Stop, Expr* Step) {
    assert(Start && Stop && Step && "Start, Stop, and Step must be non-null");
    Start = Start->IgnoreParenImpCasts();
    // we need only check start
    if(isa<ApproxIndexVarRefExpr>(Start)) {
      // if start is an AIVRE, we're in case 1: [i]
      return AIVREChildKind::STANDALONE;
    }
    if(anyChildHasType<ApproxIndexVarRefExpr>(Start)) {
      // if any child has an AIVRE, we're in case 2: [i*3:i*3+3]
      return AIVREChildKind::BINARY_EXPR;
    }
    return AIVREChildKind::NONE;
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
  int indirection_depth = 0;
  SourceLocation RBracketLoc;

    ApproxArraySliceExpr(Expr *Base, llvm::ArrayRef<Expr *> DSlices,
                         QualType Type, ExprValueKind VK, ExprObjectKind OK,
                         SourceLocation RBLoc, int indirection_depth)
        : Expr(ApproxArraySliceExprClass, Type, VK, OK), RBracketLoc{RBLoc} {
      numDims = DSlices.size();
      this->indirection_depth = indirection_depth;

      setBase(Base);
      setDimensionSlices(DSlices);
    setDependence(computeDependence(this));
    }

  public:
  explicit ApproxArraySliceExpr(EmptyShell Empty)
      : Expr(ApproxArraySliceExprClass, Empty) {}

  static ApproxArraySliceExpr *Create(const ASTContext &C, Expr *Base,
                                      llvm::ArrayRef<Expr *> DSlices,
                                      QualType Type, SourceLocation RBLoc,
                                      int indirection_depth) {
    void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(1 + DSlices.size()),
                           alignof(ApproxArraySliceExpr));
    return new (Mem) ApproxArraySliceExpr(Base, DSlices, Type, VK_LValue,
                                          OK_Ordinary, RBLoc, indirection_depth);
  }

  const Expr *getBase() const { return getTrailingObjects<Expr *>()[0];}
  Expr *getBase() { return getTrailingObjects<Expr *>()[0];}

  bool hasBase() const { return getBase() != nullptr;}
  int getIndirectionDepth() const { return indirection_depth; }

  QualType getBaseOriginalType(const Expr *Base);
  void setBase(Expr *E) { getTrailingObjects<Expr *>()[0] = E;}
  void setDimensionSlices(llvm::ArrayRef<Expr *> DSlices) {
    assert(DSlices.size() == numDims && "Wrong number of dimension slices");
    llvm::copy(DSlices, getTrailingObjects<Expr *>() + 1);
  }

  unsigned getNumDimensionSlices() const { return numDims; }
  void setNumDimensionSlices(unsigned N) { numDims = N; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    if(hasBase())
      return getBase()->getBeginLoc();
    return getTrailingObjects<Expr *>()[1]->getBeginLoc();
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
  std::optional<VarDecl*> Decl;
  
  // when we want to identify an index variable in a shape,
  // we need some way to identify it. We choose negative integers,
  // as they are not valid within shapes. Each index variable
  // is given a unique negative integer used to represent all instances
  // of that index variable
  static std::unordered_map<std::string, int> shapeReprMap;
  static int nextShapeRepr;

  void setShapeRepr(llvm::StringRef Name) {
    std::string NameStr = Name.str();
    if(shapeReprMap.find(NameStr) == shapeReprMap.end()) {
      shapeReprMap[NameStr] = nextShapeRepr;
      nextShapeRepr--;
    }
  }

  public:
  ApproxIndexVarRefExpr(IdentifierInfo *II, QualType Type, ExprValueKind VK,
                      ExprObjectKind OK, SourceLocation Loc)
      : Expr(ApproxIndexVarRefExprClass, Type, VK, OK), Loc(Loc) {
    assert(II && "No identifier provided!");
    Identifier = II;
    setDependence(computeDependence(this));
    setShapeRepr(II->getName());
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
    llvm::StringRef getDeclName() const {
    assert(hasDecl() && "Attempt to get Decl Name of Index var without decl");
    return getDecl().value()->getName();
    }

    int getShapeRepresentation() const {
      std::string NameStr  = std::string(getName());
      return shapeReprMap[NameStr];
    }

    void setDecl(VarDecl *D) { Decl.emplace(D); }
    bool hasDecl() const { return Decl.has_value(); }
    std::optional<VarDecl*> getDecl() const { return Decl; }


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
