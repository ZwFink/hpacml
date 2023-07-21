//===--- DeclApprox.h - Approx Declarations---------------------------------*-
// C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some Approx-specific declarative directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLAPPROX_H
#define LLVM_CLANG_AST_DECLAPPROX_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"
#include "clang/Basic/Approx.h"

// TOOD: This is a hack, this must match the definitions in Parser.h
using ApproxNDTensorSlice = llvm::SmallVector<clang::Expr *, 8>;
using ApproxNDTensorSliceCollection = llvm::SmallVector<ApproxNDTensorSlice, 16>;

namespace clang {

class ApproxDecl {
  SourceLocation StartLoc;
  SourceLocation EndLoc;
  friend class ASTDeclReader;

  approx::DeclKind Kind;

  protected:
  ApproxDecl(approx::DeclKind Kind, SourceLocation StartLoc, SourceLocation EndLoc)
      : StartLoc(StartLoc), EndLoc(EndLoc), Kind(Kind) {}

  public:
  static const std::string Name[approx::DK_END];

  SourceLocation getBeginLoc() const { return StartLoc; }
  SourceLocation getEndLoc() const { return EndLoc; }
  SourceRange getSourceRange() const { return SourceRange(StartLoc, EndLoc); }

  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  approx::DeclKind getDeclKind() const { return Kind; }
  std::string getAsString() const { return Name[Kind]; }

  using child_iterator = StmtIterator;
  using const_child_iterator = ConstStmtIterator;
  using child_range = llvm::iterator_range<child_iterator>;
  using const_child_range = llvm::iterator_range<const_child_iterator>;

  child_range children();
  const_child_range children() const {
    auto Children = const_cast<ApproxDecl *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }

  static bool classof(const ApproxDecl *) { return true; }
};

class ApproxDeclareTensorFunctorDecl final : public ApproxDecl, public ValueDecl {

  std::string FunctorName;
  ApproxNDTensorSlice LHSSlice;
  ApproxNDTensorSliceCollection RHSSlices;

    ApproxDeclareTensorFunctorDecl(SourceLocation StartLoc, SourceLocation EndLoc,
                            DeclarationName FunctorName,
                            DeclContext *DC,
                            QualType T,
                            ApproxNDTensorSlice LHSSlice,
                            ApproxNDTensorSliceCollection RHSSlices)
        : ApproxDecl(approx::DK_TF, StartLoc, EndLoc),
          ValueDecl{Decl::Kind::ApproxDeclareTensorFunctor, DC, StartLoc, FunctorName, T},
          FunctorName{FunctorName.getAsString()}, LHSSlice{LHSSlice}, RHSSlices{RHSSlices} {}

    // build an empty clause 
    ApproxDeclareTensorFunctorDecl()
        : ApproxDecl(approx::DK_TF, SourceLocation(), SourceLocation()),
        ValueDecl{Decl::Kind::ApproxDeclareTensorFunctor, nullptr, SourceLocation(), DeclarationName(), QualType()},
        FunctorName{}, LHSSlice{}, RHSSlices{} {}

  public:
    static ApproxDeclareTensorFunctorDecl *
    Create(ASTContext &C, DeclContext *DC, SourceRange SR,
           DeclarationName FunctorName, QualType T,
           ApproxNDTensorSlice LHSSlice,
           ApproxNDTensorSliceCollection RHSSlices);

    static bool classof(const ApproxDecl *T) {
      return T->getDeclKind() == approx::DK_TF;
    }

    child_range children() {
      llvm_unreachable("Children not implemented for TFDeclClause");
      return child_range(child_iterator(), child_iterator());
    }

    const_child_range children() const {
      llvm_unreachable("Const children not implemented for TFDeclClause");
      return const_child_range(const_child_iterator(), const_child_iterator());
    }

    child_range used_children() {
      llvm_unreachable("Used children not implemented for TFDeclClause");
      return child_range(child_iterator(), child_iterator());
    }
    const_child_range used_children() const {
      llvm_unreachable("Const used children not implemented for TFDeclClause");
      return const_child_range(const_child_iterator(), const_child_iterator());
    }

    llvm::StringRef getFunctorName() const {return FunctorName;}

    llvm::ArrayRef<Expr*> getLHSSlice() const {return LHSSlice;}
    ApproxNDTensorSliceCollection &getRHSSlices() {return RHSSlices;}

    static bool classof(const Decl *D) {
      return classofKind(D->getKind());
    }
    static bool classofKind(Decl::Kind K) { return K == Decl::Kind::ApproxDeclareTensorFunctor; }

};

class ApproxDeclareTensorDecl final : public ApproxDecl, public ValueDecl {
  std::string TFName;
  std::string TensorName;
  llvm::SmallVector<Expr*, 8> ArraySlices;
    ApproxDeclareTensorDecl(SourceLocation StartLoc, SourceLocation EndLoc,
                            DeclarationName TensorName, DeclContext *DC,
                            QualType T, IdentifierInfo *TFName,
                            llvm::ArrayRef<Expr *> ArraySlices)
        : ApproxDecl(approx::DK_T, StartLoc, EndLoc),
          ValueDecl{Decl::Kind::ApproxDeclareTensor, DC, StartLoc, TensorName,
                    T},
          TFName{TFName->getName()}, TensorName{TensorName.getAsString()} {
      this->ArraySlices.append(ArraySlices.begin(), ArraySlices.end());
    }

  // build an empty decl
  ApproxDeclareTensorDecl()
      : ApproxDecl(approx::DK_T, SourceLocation(), SourceLocation()),
        ValueDecl{Decl::Kind::ApproxDeclareTensor, nullptr, SourceLocation(), DeclarationName(), QualType()},
        TFName{}, TensorName{}, ArraySlices{} {}

  public:
  static ApproxDeclareTensorDecl *Create(ASTContext &C, DeclContext *DC,
                                         SourceRange SR,
                                         DeclarationName TensorName,
                                         QualType T, IdentifierInfo *TFName,
                                         llvm::ArrayRef<Expr *> ArraySlices);

  static bool classof(const Decl *D) {
    return classofKind(D->getKind());
  }
  static bool classofKind(Decl::Kind K) { return K == Decl::Kind::ApproxDeclareTensor; }

  child_range children() {
    llvm_unreachable("Children not implemented for TensorDeclClause");
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    llvm_unreachable("Const children not implemented for TensorDeclClause");
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    llvm_unreachable("Used children not implemented for TensorDeclClause");
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    llvm_unreachable("Const used children not implemented for TensorDeclClause");
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  llvm::StringRef getTensorName() const {return TensorName;} 
  llvm::StringRef getTFName() const {return TFName;} 

  llvm::ArrayRef<Expr*> getArraySlices() {return ArraySlices;}
};


class ApproxCapturedExprDecl final : public VarDecl {
  friend class ASTDeclReader;
  void anchor() override;

  ApproxCapturedExprDecl(ASTContext &C, DeclContext *DC, IdentifierInfo *Id,
                         QualType Type, TypeSourceInfo *TInfo,
                         SourceLocation StartLoc)
      : VarDecl(ApproxCapturedExpr, C, DC, StartLoc, StartLoc, Id, Type, TInfo,
                SC_None) {
    setImplicit();
  }

public:
  static ApproxCapturedExprDecl *Create(ASTContext &C, DeclContext *DC,
                                        IdentifierInfo *Id, QualType T,
                                        SourceLocation StartLoc);

  static ApproxCapturedExprDecl *CreateDeserialized(ASTContext &C, unsigned ID);

  SourceRange getSourceRange() const override LLVM_READONLY;

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ApproxCapturedExpr; }
};

} // namespace clang

#endif
