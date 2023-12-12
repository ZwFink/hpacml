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

// collect all expressions of a given type
// if Targettype expressions are nested, collects only 
// the top level
template <typename TargetType> class TargetExprCollector {
public:
  std::vector<Expr *> Targets;

  template<typename StopFn>
  void Collect(clang::Expr *E, StopFn ShouldStop) {
    if (E == nullptr) {
      return;
    }

    if (auto *Target = llvm::dyn_cast_or_null<TargetType>(E)) {
      Targets.push_back(E);
    } else {
      if (ShouldStop(E)) {
        return;
      }
      // Otherwise, recurse on its children.
      for (auto it = E->child_begin(), end = E->child_end(); it != end; ++it) {
        if (clang::Expr *child = llvm::dyn_cast_or_null<clang::Expr>(*it)) {
          Collect(child, ShouldStop);
        }
      }
    }
  }

  template<typename OutputIt>
  void CopyTo(OutputIt It) {
    for (auto *Target : Targets) {
      *It++ = Target;
    }
  }

  llvm::ArrayRef<Expr*> getCollectedExprs() {
    return Targets;
  }
};

class ApproxArraySliceExpr final
: public Expr,
private llvm::TrailingObjects<ApproxArraySliceExpr, Expr*> {
  friend TrailingObjects;
  unsigned numDims = 0;
  int num_indirections = 0;
  int indirection_depth = 0;
  SourceLocation RBracketLoc;

    ApproxArraySliceExpr(llvm::ArrayRef<Expr*> Indirections, llvm::ArrayRef<Expr *> DSlices,
                         QualType Type, ExprValueKind VK, ExprObjectKind OK,
                         SourceLocation RBLoc, int indirection_depth)
        : Expr(ApproxArraySliceExprClass, Type, VK, OK), RBracketLoc{RBLoc} {
      numDims = DSlices.size();
      this->num_indirections = Indirections.size();
      this->indirection_depth = indirection_depth;

      setIndirections(Indirections);
      setDimensionSlices(DSlices);
    setDependence(computeDependence(this));
    }

  ArrayRef<Expr *> getIndirectionsFromTrailing() const { return llvm::ArrayRef(getTrailingObjects<Expr *>(), num_indirections); }
  ArrayRef<Expr *> getSlicesFromTrailing() const {
    // we may get either slices either from a child indirection or directly from
    // the indirections. We supply both here so they can be found in the AST traversal
    return llvm::ArrayRef(getTrailingObjects<Expr *>(), numDims + num_indirections);
  }

  public:
  explicit ApproxArraySliceExpr(EmptyShell Empty)
      : Expr(ApproxArraySliceExprClass, Empty) {}

  static ApproxArraySliceExpr *Create(const ASTContext &C, llvm::ArrayRef<Expr*> Indirections,
                                      llvm::ArrayRef<Expr *> DSlices,
                                      QualType Type, SourceLocation RBLoc,
                                      int indirection_depth) {
    void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(Indirections.size() + DSlices.size()),
                           alignof(ApproxArraySliceExpr));
    return new (Mem) ApproxArraySliceExpr(Indirections, DSlices, Type, VK_LValue,
                                          OK_Ordinary, RBLoc, indirection_depth);
  }

  bool hasIndirections() const { return num_indirections > 0; }
  int getIndirectionDepth() const { return indirection_depth; }

  void setIndirections(llvm::ArrayRef<Expr*> Indirections) {
    assert(Indirections.size() == static_cast<size_t>(num_indirections) && "Wrong number of indirections");
    llvm::copy(Indirections, getTrailingObjects<Expr *>());
  }
  QualType getBaseOriginalType(const Expr *Base);
  void setDimensionSlices(llvm::ArrayRef<Expr *> DSlices) {
    assert(DSlices.size() == static_cast<size_t>(numDims) && "Wrong number of dimension slices");
    llvm::copy(DSlices, getTrailingObjects<Expr *>() + num_indirections);
  }

  unsigned getNumDimensionSlices() {
    return getSlices().size();
  }
  void setNumDimensionSlices(unsigned N) { numDims = N; }

  unsigned getNumIndirections() const { return getIndirections().size(); }
  void setNumIndirections(unsigned N) { num_indirections = N; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getTrailingObjects<Expr *>()[0]->getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY {return RBracketLoc;}
  void setEndLoc(SourceLocation L) { RBracketLoc = L; }

  unsigned numTrailingObjects(OverloadToken<Expr *>) const { return numDims + num_indirections; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getTrailingObjects<Expr *>()[0]->getBeginLoc();
  }

  llvm::SmallVector<Expr*, 8> getIndirections() const { 
    auto StopOnApproxSlice = [](const Expr *E) {return isa<ApproxSliceExpr>(E);};
    auto Base = getIndirectionsFromTrailing();
    llvm::SmallVector<Expr*, 8> Result;
    TargetExprCollector<DeclRefExpr> IndirectionCollector;
    for(auto E : Base) {
      IndirectionCollector.Collect(E, StopOnApproxSlice);
    }

    IndirectionCollector.CopyTo(std::back_inserter(Result)); 
    return Result;
  }

  llvm::SmallVector<Expr*, 8> getSlices() {
    auto StopOnNull = [](const Expr *E) {return false;};
    auto Base = getSlicesFromTrailing();
    llvm::SmallVector<Expr*, 8> Result;
    TargetExprCollector<ApproxSliceExpr> SliceCollector;
    for(auto E : Base) {
      SliceCollector.Collect(E, StopOnNull);
    }

    SliceCollector.CopyTo(std::back_inserter(Result)); 
    return Result;
  }

  child_range children() {
    Stmt **Begin = reinterpret_cast<Stmt **>(getTrailingObjects<Expr *>());
    return child_range(Begin, Begin + numDims + num_indirections);
  }

  const_child_range children() const { 
    Stmt *const *Begin = reinterpret_cast<Stmt *const *>(getTrailingObjects<Expr *>());
    return const_child_range(Begin, Begin + numDims + num_indirections);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ApproxArraySliceExprClass;
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

class ApproxCompoundExpr final : public Expr, 
private llvm::TrailingObjects<ApproxCompoundExpr, Decl*, Expr*> {
  friend TrailingObjects;
  unsigned num_decls = 0;
  unsigned num_exprs = 0;
  public:
  ApproxCompoundExpr(llvm::ArrayRef<Decl *> Decls, 
                     llvm::ArrayRef<Expr *> Exprs, QualType Type,
                     ExprValueKind VK, ExprObjectKind OK)
      : Expr(ApproxCompoundExprClass, Type, VK, OK) {
    num_decls = Decls.size();
    num_exprs = Exprs.size();
    setDeclarations(Decls);
    setExpressions(Exprs);
    setDependence(computeDependence(this));
  }

  public:
  static ApproxCompoundExpr *Create(const ASTContext &C, llvm::ArrayRef<Decl *> Decls,
                             llvm::ArrayRef<Expr *> Exprs, QualType Type) {
    auto DeclAlloc = totalSizeToAlloc<Decl *, Expr*>(Decls.size(), Exprs.size());
    void *Mem = C.Allocate(DeclAlloc,
                           alignof(ApproxCompoundExpr));
    return new (Mem) ApproxCompoundExpr(Decls, Exprs, Type, VK_LValue,
                                          OK_Ordinary);
                             }


  explicit ApproxCompoundExpr(EmptyShell Empty)
      : Expr(ApproxCompoundExprClass, Empty) {}

  ArrayRef<Expr *> getExpressions() const {
    return llvm::ArrayRef(getTrailingObjects<Expr *>(), num_exprs);
  }

  ArrayRef<Decl *> getDeclarations() const {
    return llvm::ArrayRef(getTrailingObjects<Decl *>(), num_decls);
  }

  unsigned numTrailingObjects(OverloadToken<Decl *>) const { return num_decls;}
  unsigned numTrailingObjects(OverloadToken<Expr *>) const { return num_exprs;}

  SourceLocation getBeginLoc() const LLVM_READONLY {
    if(num_decls > 0)
      return getDeclarations()[0]->getBeginLoc();
    else
      return getExpressions()[0]->getBeginLoc();
  }

  SourceLocation getEndLoc() const LLVM_READONLY {
    return getExpressions()[num_exprs - 1]->getEndLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ApproxCompoundExprClass;
  }

  child_range children() {
    Stmt **Begin = reinterpret_cast<Stmt **>(getTrailingObjects<Decl *>());
    return child_range(Begin, Begin + num_decls + num_exprs);
  }

  const_child_range children() const {
    Stmt *const *Begin = reinterpret_cast<Stmt *const *>(getTrailingObjects<Decl *>());
    return const_child_range(Begin, Begin + num_decls + num_exprs);
  }

  void setExpressions(llvm::ArrayRef<Expr *> Exprs) {
    assert(Exprs.size() == num_exprs && "Wrong number of expressions");
    llvm::copy(Exprs, getTrailingObjects<Expr *>());
  }

  void setDeclarations(llvm::ArrayRef<Decl *> Decls) {
    assert(Decls.size() == num_decls && "Wrong number of declarations");
    llvm::copy(Decls, getTrailingObjects<Decl *>());
  }

  void setNumExpressions(unsigned N) { num_exprs = N; }
  void setNumDeclarations(unsigned N) { num_decls = N; }

  unsigned getNumExpressions() const { return num_exprs; }
  unsigned getNumDeclarations() const { return num_decls; }

  SourceLocation getExprLoc() const {
    return getEndLoc();
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
