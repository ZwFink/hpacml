//===--- DeclApprox.cpp - Declaration Approx AST Node Implementation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements ApproxCaptureDecl
/// classes.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclApprox.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// ApproxCapturedExprDecl Implementation.
//===----------------------------------------------------------------------===//

void ApproxCapturedExprDecl::anchor() {}

ApproxCapturedExprDecl *
ApproxCapturedExprDecl::Create(ASTContext &C, DeclContext *DC,
                               IdentifierInfo *Id, QualType T,
                               SourceLocation StartLoc) {
  return new (C, DC) ApproxCapturedExprDecl(
      C, DC, Id, T, C.getTrivialTypeSourceInfo(T), StartLoc);
}

ApproxCapturedExprDecl *
ApproxCapturedExprDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID)
      ApproxCapturedExprDecl(C, nullptr, nullptr, QualType(),
                             /*TInfo=*/nullptr, SourceLocation());
}

SourceRange ApproxCapturedExprDecl::getSourceRange() const {
  assert(hasInit());
  return SourceRange(getInit()->getBeginLoc(), getInit()->getEndLoc());
}

ApproxDeclareTensorDecl *ApproxDeclareTensorDecl::Create(
    ASTContext &C, DeclContext *DC, SourceRange SR, DeclarationName TensorName,
    QualType T, Decl *TensorFunctor, llvm::ArrayRef<Expr *> ArraySlices) {

  SourceLocation Begin = SR.getBegin();
  SourceLocation End = SR.getEnd();

  return new (C, DC)
  ApproxDeclareTensorDecl(Begin, End, TensorName, DC, T, TensorFunctor, ArraySlices);
}

ApproxDeclareTensorFunctorDecl *ApproxDeclareTensorFunctorDecl::Create(
    ASTContext &C, DeclContext *DC, SourceRange SR, DeclarationName FunctorName,
    QualType T, ApproxNDTensorSlice LHSSlice,
    ApproxNDTensorSliceCollection RHSSlices) {

  SourceLocation Begin = SR.getBegin();
  SourceLocation End = SR.getEnd();
  auto * Ret = new (C, DC)
  ApproxDeclareTensorFunctorDecl( Begin, End, FunctorName, DC, T, LHSSlice, RHSSlices);
  return Ret;
}