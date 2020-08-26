//===--- ExprApprox.cpp - Approximate Expression AST Node Implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ExprApprox class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include "clang/AST/ExprApprox.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/ComputeDependence.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DependenceFlags.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LiteralSupport.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
using namespace clang;

QualType ApproxArraySectionExpr::getBaseOriginalType(const Expr *Base) {
  unsigned ArraySectionCount = 0;
  while (auto *OASE = dyn_cast<ApproxArraySectionExpr>(Base->IgnoreParens())) {
    Base = OASE->getBase();
    ++ArraySectionCount;
  }
  while (auto *ASE =
             dyn_cast<ArraySubscriptExpr>(Base->IgnoreParenImpCasts())) {
    Base = ASE->getBase();
    ++ArraySectionCount;
  }
  Base = Base->IgnoreParenImpCasts();
  auto OriginalTy = Base->getType();
  if (auto *DRE = dyn_cast<DeclRefExpr>(Base))
    if (auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl()))
      OriginalTy = PVD->getOriginalType().getNonReferenceType();

  for (unsigned Cnt = 0; Cnt < ArraySectionCount; ++Cnt) {
    if (OriginalTy->isAnyPointerType())
      OriginalTy = OriginalTy->getPointeeType();
    else {
      assert (OriginalTy->isArrayType());
      OriginalTy = OriginalTy->castAsArrayTypeUnsafe()->getElementType();
    }
  }
  return OriginalTy;
}

void ApproxArrayShapingExpr::setDimensions(ArrayRef<Expr *> Dims) {
  assert(
      NumDims == Dims.size() &&
      "Preallocated number of dimensions is different from the provided one.");
  llvm::copy(Dims, getTrailingObjects<Expr *>());
}

void ApproxArrayShapingExpr::setBracketsRanges(ArrayRef<SourceRange> BR) {
  assert(
      NumDims == BR.size() &&
      "Preallocated number of dimensions is different from the provided one.");
  llvm::copy(BR, getTrailingObjects<SourceRange>());
}

ApproxArrayShapingExpr::ApproxArrayShapingExpr(QualType ExprTy, Expr *Op,
                                         SourceLocation L, SourceLocation R,
                                         ArrayRef<Expr *> Dims)
    : Expr(ApproxArrayShapingExprClass, ExprTy, VK_LValue, OK_Ordinary), LPLoc(L),
      RPLoc(R), NumDims(Dims.size()) {
  setBase(Op);
  setDimensions(Dims);
  setDependence(computeDependence(this));
}

ApproxArrayShapingExpr *
ApproxArrayShapingExpr::Create(const ASTContext &Context, QualType T, Expr *Op,
                            SourceLocation L, SourceLocation R,
                            ArrayRef<Expr *> Dims,
                            ArrayRef<SourceRange> BracketRanges) {
  assert(Dims.size() == BracketRanges.size() &&
         "Different number of dimensions and brackets ranges.");
  void *Mem = Context.Allocate(
      totalSizeToAlloc<Expr *, SourceRange>(Dims.size() + 1, Dims.size()),
      alignof(ApproxArrayShapingExpr));
  auto *E = new (Mem) ApproxArrayShapingExpr(T, Op, L, R, Dims);
  E->setBracketsRanges(BracketRanges);
  return E;
}

ApproxArrayShapingExpr *ApproxArrayShapingExpr::CreateEmpty(const ASTContext &Context,
                                                      unsigned NumDims) {
  void *Mem = Context.Allocate(
      totalSizeToAlloc<Expr *, SourceRange>(NumDims + 1, NumDims),
      alignof(ApproxArrayShapingExpr));
  return new (Mem) ApproxArrayShapingExpr(EmptyShell(), NumDims);
}

void ApproxIteratorExpr::setIteratorDeclaration(unsigned I, Decl *D) {
  assert(I < NumIterators &&
         "Idx is greater or equal the number of iterators definitions.");
  getTrailingObjects<Decl *>()[I] = D;
}

void ApproxIteratorExpr::setAssignmentLoc(unsigned I, SourceLocation Loc) {
  assert(I < NumIterators &&
         "Idx is greater or equal the number of iterators definitions.");
  getTrailingObjects<
      SourceLocation>()[I * static_cast<int>(RangeLocOffset::Total) +
                        static_cast<int>(RangeLocOffset::AssignLoc)] = Loc;
}

void ApproxIteratorExpr::setIteratorRange(unsigned I, Expr *Begin,
                                       SourceLocation ColonLoc, Expr *End,
                                       SourceLocation SecondColonLoc,
                                       Expr *Step) {
  assert(I < NumIterators &&
         "Idx is greater or equal the number of iterators definitions.");
  getTrailingObjects<Expr *>()[I * static_cast<int>(RangeExprOffset::Total) +
                               static_cast<int>(RangeExprOffset::Begin)] =
      Begin;
  getTrailingObjects<Expr *>()[I * static_cast<int>(RangeExprOffset::Total) +
                               static_cast<int>(RangeExprOffset::End)] = End;
  getTrailingObjects<Expr *>()[I * static_cast<int>(RangeExprOffset::Total) +
                               static_cast<int>(RangeExprOffset::Step)] = Step;
  getTrailingObjects<
      SourceLocation>()[I * static_cast<int>(RangeLocOffset::Total) +
                        static_cast<int>(RangeLocOffset::FirstColonLoc)] =
      ColonLoc;
  getTrailingObjects<
      SourceLocation>()[I * static_cast<int>(RangeLocOffset::Total) +
                        static_cast<int>(RangeLocOffset::SecondColonLoc)] =
      SecondColonLoc;
}

Decl *ApproxIteratorExpr::getIteratorDecl(unsigned I) {
  return getTrailingObjects<Decl *>()[I];
}

ApproxIteratorExpr::IteratorRange ApproxIteratorExpr::getIteratorRange(unsigned I) {
  IteratorRange Res;
  Res.Begin =
      getTrailingObjects<Expr *>()[I * static_cast<int>(
                                           RangeExprOffset::Total) +
                                   static_cast<int>(RangeExprOffset::Begin)];
  Res.End =
      getTrailingObjects<Expr *>()[I * static_cast<int>(
                                           RangeExprOffset::Total) +
                                   static_cast<int>(RangeExprOffset::End)];
  Res.Step =
      getTrailingObjects<Expr *>()[I * static_cast<int>(
                                           RangeExprOffset::Total) +
                                   static_cast<int>(RangeExprOffset::Step)];
  return Res;
}

SourceLocation ApproxIteratorExpr::getAssignLoc(unsigned I) const {
  return getTrailingObjects<
      SourceLocation>()[I * static_cast<int>(RangeLocOffset::Total) +
                        static_cast<int>(RangeLocOffset::AssignLoc)];
}

SourceLocation ApproxIteratorExpr::getColonLoc(unsigned I) const {
  return getTrailingObjects<
      SourceLocation>()[I * static_cast<int>(RangeLocOffset::Total) +
                        static_cast<int>(RangeLocOffset::FirstColonLoc)];
}

SourceLocation ApproxIteratorExpr::getSecondColonLoc(unsigned I) const {
  return getTrailingObjects<
      SourceLocation>()[I * static_cast<int>(RangeLocOffset::Total) +
                        static_cast<int>(RangeLocOffset::SecondColonLoc)];
}

void ApproxIteratorExpr::setHelper(unsigned I, const ApproxIteratorHelperData &D) {
  getTrailingObjects<ApproxIteratorHelperData>()[I] = D;
}

ApproxIteratorHelperData &ApproxIteratorExpr::getHelper(unsigned I) {
  return getTrailingObjects<ApproxIteratorHelperData>()[I];
}

const ApproxIteratorHelperData &ApproxIteratorExpr::getHelper(unsigned I) const {
  return getTrailingObjects<ApproxIteratorHelperData>()[I];
}

ApproxIteratorExpr::ApproxIteratorExpr(
    QualType ExprTy, SourceLocation IteratorKwLoc, SourceLocation L,
    SourceLocation R, ArrayRef<ApproxIteratorExpr::IteratorDefinition> Data,
    ArrayRef<ApproxIteratorHelperData> Helpers)
    : Expr(ApproxIteratorExprClass, ExprTy, VK_LValue, OK_Ordinary),
      IteratorKwLoc(IteratorKwLoc), LPLoc(L), RPLoc(R),
      NumIterators(Data.size()) {
  for (unsigned I = 0, E = Data.size(); I < E; ++I) {
    const IteratorDefinition &D = Data[I];
    setIteratorDeclaration(I, D.IteratorDecl);
    setAssignmentLoc(I, D.AssignmentLoc);
    setIteratorRange(I, D.Range.Begin, D.ColonLoc, D.Range.End,
                     D.SecondColonLoc, D.Range.Step);
    setHelper(I, Helpers[I]);
  }
  setDependence(computeDependence(this));
}

ApproxIteratorExpr *
ApproxIteratorExpr::Create(const ASTContext &Context, QualType T,
                        SourceLocation IteratorKwLoc, SourceLocation L,
                        SourceLocation R,
                        ArrayRef<ApproxIteratorExpr::IteratorDefinition> Data,
                        ArrayRef<ApproxIteratorHelperData> Helpers) {
  assert(Data.size() == Helpers.size() &&
         "Data and helpers must have the same size.");
  void *Mem = Context.Allocate(
      totalSizeToAlloc<Decl *, Expr *, SourceLocation, ApproxIteratorHelperData>(
          Data.size(), Data.size() * static_cast<int>(RangeExprOffset::Total),
          Data.size() * static_cast<int>(RangeLocOffset::Total),
          Helpers.size()),
      alignof(ApproxIteratorExpr));
  return new (Mem) ApproxIteratorExpr(T, IteratorKwLoc, L, R, Data, Helpers);
}

ApproxIteratorExpr *ApproxIteratorExpr::CreateEmpty(const ASTContext &Context,
                                              unsigned NumIterators) {
  void *Mem = Context.Allocate(
      totalSizeToAlloc<Decl *, Expr *, SourceLocation, ApproxIteratorHelperData>(
          NumIterators, NumIterators * static_cast<int>(RangeExprOffset::Total),
          NumIterators * static_cast<int>(RangeLocOffset::Total), NumIterators),
      alignof(ApproxIteratorExpr));
  return new (Mem) ApproxIteratorExpr(EmptyShell(), NumIterators);
}
