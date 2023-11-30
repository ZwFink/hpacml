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
#include "clang/AST/ExprApprox.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Approx.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"
#include <unordered_set>

// TOOD: This is a hack, this must match the definitions in Parser.h
using ApproxNDTensorSlice = llvm::SmallVector<clang::Expr *, 8>;
using ApproxNDTensorSliceCollection = llvm::SmallVector<ApproxNDTensorSlice, 16>;

namespace clang {

  class LeafExprCollector {
public:
  std::vector<clang::Expr*> Leafs;

  void Collect(clang::Expr *E) {
    if (E == nullptr) {
      return;
    }

    // If the expression has no children, it's a leaf.
    if (E->child_begin() == E->child_end()) {
      Leafs.push_back(E);
    } else {
      // Otherwise, recurse on its children.
      for (auto it = E->child_begin(), end = E->child_end(); it != end; ++it) {
        if (clang::Expr *child = llvm::dyn_cast_or_null<clang::Expr>(*it)) {
          Collect(child);
        }
      }
    }
  }
};

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
  Expr *LHSSliceExpr = nullptr;
  llvm::SmallVector<Expr*, 16> RHSSliceExprs;
  mutable std::optional<llvm::SmallVector<Expr *, 16>> SymbolicVars;

    ApproxDeclareTensorFunctorDecl(SourceLocation StartLoc, SourceLocation EndLoc,
                            DeclarationName FunctorName,
                            DeclContext *DC,
                            QualType T,
                            Expr *LHSSlice,
                            llvm::ArrayRef<Expr*> RHSSlices)
        : ApproxDecl(approx::DK_TF, StartLoc, EndLoc),
          ValueDecl{Decl::Kind::ApproxDeclareTensorFunctor, DC, StartLoc, FunctorName, T},
          FunctorName{FunctorName.getAsString()}, LHSSliceExpr{LHSSlice}, RHSSliceExprs{RHSSlices} {
            copySlicesFromAASE(this->LHSSlice, LHSSlice);
            initializeRHSExprs(RHSSlices);
            // createSymbolicVarDecls();
          }

    template<typename SV>
    void copySlicesFromAASE(SV& Vec, Expr *E) {
      ApproxArraySliceExpr *AASE = cast<ApproxArraySliceExpr>(E);
      for (auto *Slice : AASE->getSlices()) {
        Vec.push_back(Slice);
      }
    }

    void initializeRHSExprs(llvm::ArrayRef<Expr*> RHSSlices) {
      for (auto *E : RHSSlices) {
        ApproxNDTensorSlice Slice;
        copySlicesFromAASE(Slice, E);
        this->RHSSlices.push_back(Slice);
      }
    }

    ApproxNDTensorSliceCollection getSlicesFromExprCollection(llvm::ArrayRef<Expr*> Exprs) {
      ApproxNDTensorSliceCollection Slices;
      for (auto *E : Exprs) {
        ApproxNDTensorSlice Slice;
        LeafExprCollector LeafCollector;
        LeafCollector.Collect(E);
        for (auto *Leaf : LeafCollector.Leafs) {
          Slice.push_back(Leaf);
        }
        Slices.push_back(Slice);
      }
      return Slices;
    }

    // build an empty clause 
    ApproxDeclareTensorFunctorDecl()
        : ApproxDecl(approx::DK_TF, SourceLocation(), SourceLocation()),
        ValueDecl{Decl::Kind::ApproxDeclareTensorFunctor, nullptr, SourceLocation(), DeclarationName(), QualType()},
        FunctorName{}, LHSSlice{}, RHSSlices{} {}



    VarDecl *_DeclareSymbolicVar(Expr *Var) {
      ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(Var);
      assert(IndexVar && "Attempt to declare a non-index variable as symbolic");
      IdentifierInfo *Id = createIdentifierForSymbolicVar(getASTContext(), Var);
      ASTContext &C = getASTContext();
      DeclContext *DC = getDeclContext();
      VarDecl *SymbolicVar = VarDecl::Create(getASTContext(), DC, IndexVar->getBeginLoc(),
                                             IndexVar->getEndLoc(), Id, C.getIntTypeForBitwidth(64, false), nullptr, SC_None);
      return SymbolicVar;
    }

    #if 0
    // this implementation incorrectly assumes that all appearances of symbolic vars 
    // on the RHS must have the same range. Under this assumption, it creates one 
    // declaration for each symbolic var. However, the true requirement is that
    // all appearances of symbolic vars on the RHS must have the same /shape/.
    void createSymbolicVarDecls() {
      std::unordered_map<std::string, VarDecl *> SymbolicVarDecls;
      for (auto *Var : getUniqueSymbolicVars()) {
        auto *Decl = DeclareSymbolicVar(Var);
        ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(Var);
        SymbolicVarDecls[std::string(IndexVar->getName())] = Decl;
      }

      for(auto *Var : getSymbolicVars()) {
        ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(Var);
        auto *Decl = SymbolicVarDecls[std::string(IndexVar->getName())];
        IndexVar->setDecl(Decl);
      }
    }
    #endif

    void createSymbolicVarDecls() {

      for(auto *Var : getSymbolicVars()) {
        auto *Decl = _DeclareSymbolicVar(Var);
        ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(Var);
        IndexVar->setDecl(Decl);
      }
    }

    static IdentifierInfo *createIdentifierForSymbolicVar(ASTContext &C, Expr * E) {
      ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(E);
      static int Counter = 0;
      assert(IndexVar && "Attempt to declare a non-index variable as symbolic");
      auto DeclName = std::string(IndexVar->getIdentifier()->getName());
      // we'll let clang disambiguate when the same symbolic var has multiple declarations
      std::string Name = "symbolicRefExpr_" + DeclName + "_" + std::to_string(Counter++);
      return &C.Idents.get(Name);
    }

  public:
    static ApproxDeclareTensorFunctorDecl *
    Create(ASTContext &C, DeclContext *DC, SourceRange SR,
           DeclarationName FunctorName, QualType T,
           Expr *LHSSlice,
           llvm::ArrayRef<Expr*> RHSSlices);

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
    llvm::StringRef getName() const { return getFunctorName(); }

    llvm::ArrayRef<Expr*> getLHSSlice() const {return LHSSlice;}
    ApproxNDTensorSliceCollection &getRHSSlices() {return RHSSlices;}
    llvm::ArrayRef<Expr*> getRHSSliceExprs() const {
      return RHSSliceExprs;
    }

    static bool classof(const Decl *D) {
      return classofKind(D->getKind());
    }
    static bool classofKind(Decl::Kind K) { return K == Decl::Kind::ApproxDeclareTensorFunctor; }

    llvm::SmallVector<Expr *, 16> getSymbolicVars() const {
      if(SymbolicVars.has_value()) {
        return *SymbolicVars;
      }
      llvm::SmallVector<Expr *, 16> Vars;
      LeafExprCollector Collector;
      // for(auto *Expr : LHSSlice) {
        // Collector.Collect(Expr);
      // }
      for (auto Slice : RHSSlices) {
        for (auto *E : Slice) {
          Collector.Collect(E);
        }
      }

      auto &Leafs = Collector.Leafs;
      for(auto *Leaf : Leafs) {
        if (isa<ApproxIndexVarRefExpr>(Leaf)) {
          Vars.push_back(Leaf);
        }
      }
      SymbolicVars = std::make_optional(Vars);
      return *SymbolicVars;
    }

    static llvm::SmallVector<Expr*, 4> getSymbolicVarsFromExpression(Expr *E) {
      llvm::SmallVector<Expr*, 4> Vars;
      LeafExprCollector Collector;
      Collector.Collect(E);
      auto &Leafs = Collector.Leafs;
      for(auto *Leaf : Leafs) {
        if (isa<ApproxIndexVarRefExpr>(Leaf)) {
          Vars.push_back(Leaf);
        }
      }
      return Vars;
    } 

    static llvm::SmallVector<Expr *, 4> getDeclaredSymbolicVarsFromExpression(Expr *E) {
      auto Vars = getSymbolicVarsFromExpression(E);
      llvm::SmallVector<Expr *, 4> DeclaredVars;
      for(auto *Var : Vars) {
        if (cast<ApproxIndexVarRefExpr>(Var)->getDecl().has_value()) {
          DeclaredVars.push_back(Var);
        }
      }
      return DeclaredVars;
    }

    llvm::SmallVector<Expr *, 16> getUniqueSymbolicVars() const {
      llvm::SmallVector<Expr *, 16> UniqueVars;
      std::unordered_set<std::string> VarNames;
      auto Vars = getSymbolicVars();
      for (auto *Var : Vars) {
        auto Name = std::string(cast<ApproxIndexVarRefExpr>(Var)->getDecl().value()->getName());
        if (VarNames.find(Name) == VarNames.end()) {
          VarNames.insert(Name);
          UniqueVars.push_back(Var);
        }
      }
      return UniqueVars;
    }

    void createSymbolicVarDeclsForExpression(Expr *E) {
      auto Vars = getSymbolicVarsFromExpression(E);
      std::unordered_map<std::string, VarDecl *> SymbolicVarDecls;
      for(auto *Var : Vars) {
        ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(Var);
        auto Name = std::string(IndexVar->getName());
        if (SymbolicVarDecls.find(Name) == SymbolicVarDecls.end()) {
          auto *Decl = _DeclareSymbolicVar(Var);
          SymbolicVarDecls[Name] = Decl;
        } else {
          auto *Decl = SymbolicVarDecls[Name];
          cast<ApproxIndexVarRefExpr>(Var)->setDecl(Decl);
        }
      }
    }

    llvm::SmallVector<Expr*, 16> getFlattenedRHSSlices() const {
      llvm::SmallVector<Expr*, 16> LinearizedSlices;
      for(auto Slice : RHSSlices) {
        for(auto *E : Slice) {
          LinearizedSlices.push_back(E);
        }
      }
      return LinearizedSlices;
    }

    template<typename InsertIterator>
    void getSymbolicVarsUniqueToExpression(InsertIterator I, Expr * E) {
      auto Vars = getSymbolicVarsFromExpression(E);
      std::unordered_set<std::string> SymbolicVarDecls;
      for(auto *Var : Vars) {
        ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(Var);
        auto Name = std::string(IndexVar->getName());
        if (SymbolicVarDecls.find(Name) == SymbolicVarDecls.end()) {
          SymbolicVarDecls.insert(Name);
          *I++ = Var;
        }
      }

    }

    llvm::SmallVector<Expr*, 16> getSymbolicVarsUniqueToEachLHSSlice() {
      llvm::SmallVector<Expr*, 16> UniqueVars;
      for(auto *E : LHSSlice) {
        getSymbolicVarsUniqueToExpression(std::back_inserter(UniqueVars), E);
      }
      return UniqueVars;
    }

    llvm::SmallVector<Expr*, 16> getSymbolicVarsUniqueToEachRHSSlice() {
      llvm::SmallVector<Expr *, 16> UniqueVars;
      for (auto NDSlice : RHSSlices) {
        for (auto *E : NDSlice) {
          getSymbolicVarsUniqueToExpression(std::back_inserter(UniqueVars), E);
        }
      }
      return UniqueVars;
    }

    llvm::SmallVector<Expr*, 16> getSymbolicVarsUniqueToEachSlice(bool RHS=true) {
      if(RHS)
        return getSymbolicVarsUniqueToEachRHSSlice();
      else
        return getSymbolicVarsUniqueToEachLHSSlice();
    }

    void DeclareSymbolicVar(Expr *E) {
      auto *Decl = _DeclareSymbolicVar(E);
      ApproxIndexVarRefExpr *IndexVar = cast<ApproxIndexVarRefExpr>(E);
      IndexVar->setDecl(Decl);
    }


};

class TensorFunctorCall {
  public:
  enum class Directionality {
    MEM_TO_TENSOR,
    TENSOR_TO_MEM
  };

  private:
  Decl *FunctorDecl;
  llvm::SmallVector<Expr*, 8> Args;
  Directionality Dir;

  public:


  TensorFunctorCall(Decl *FunctorDecl, llvm::ArrayRef<Expr*> Args)
      : FunctorDecl{FunctorDecl}, Args{Args}, Dir{Directionality::MEM_TO_TENSOR} {}

  TensorFunctorCall(Decl *FunctorDecl, llvm::ArrayRef<Expr*> Args, Directionality Dir)
      : FunctorDecl{FunctorDecl}, Args{Args}, Dir{Dir} {}

  Decl *getFunctorDecl() const {return FunctorDecl;}
  llvm::ArrayRef<Expr*> getArgs() const {return Args;}
  Directionality getDirectionality() const {return Dir;}

  void setDirectionality(Directionality Dir) {this->Dir = Dir;}
  void setFunctorDecl(Decl *FunctorDecl) {this->FunctorDecl = FunctorDecl;}
  void setArgs(llvm::ArrayRef<Expr*> Args) {this->Args.assign(Args.begin(), Args.end());}
};

class ApproxDeclareTensorDecl final : public ApproxDecl, public ValueDecl {
  public:
  using Direction = TensorFunctorCall::Directionality;

private:
  TensorFunctorCall FunctorCall;
  DeclarationName TensorName;
  llvm::SmallVector<Expr*, 8> ArraySlices;

  public:
    ApproxDeclareTensorDecl(SourceLocation StartLoc, SourceLocation EndLoc,
                            DeclarationName TensorName, DeclContext *DC,
                            QualType T, Decl *FunctorDecl,
                            llvm::ArrayRef<Expr *> ArraySlices,
                            Direction TensorDirection)
        : ApproxDecl(approx::DK_T, StartLoc, EndLoc),
          ValueDecl{Decl::Kind::ApproxDeclareTensor, DC, StartLoc, TensorName,
                    T},
          FunctorCall{FunctorDecl, ArraySlices}, TensorName{TensorName} {
            FunctorCall.setDirectionality(TensorDirection);
          }
  // build an empty decl
  ApproxDeclareTensorDecl()
      : ApproxDecl(approx::DK_T, SourceLocation(), SourceLocation()),
        ValueDecl{Decl::Kind::ApproxDeclareTensor, nullptr, SourceLocation(), DeclarationName(), QualType()},
        FunctorCall{nullptr, {}}, TensorName{} {}

  static ApproxDeclareTensorDecl *Create(ASTContext &C, DeclContext *DC,
                                         SourceRange SR,
                                         DeclarationName TensorName,
                                         QualType T, Decl *FunctorDecl,
                                         llvm::ArrayRef<Expr *> ArraySlices,
                                         Direction Dir = Direction::MEM_TO_TENSOR);

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

  llvm::StringRef getTensorName() const {return TensorName.getAsIdentifierInfo()->getName();}
  llvm::StringRef getFunctorName() const {
    assert(FunctorCall.getFunctorDecl() && "Attempt to get name of null Tensor Functor");
    return cast<ApproxDeclareTensorFunctorDecl>(FunctorCall.getFunctorDecl())->getFunctorName();
  }
  llvm::StringRef getName() const { return getTensorName(); }
  IdentifierInfo *getIdentifier() const { return TensorName.getAsIdentifierInfo(); }

  Decl *getFunctor() const {return FunctorCall.getFunctorDecl();}
  void setFunctor(Decl *Functor) {FunctorCall.setFunctorDecl(Functor);}

  llvm::ArrayRef<Expr*> getArraySlices() const {return FunctorCall.getArgs();}

  Direction getDirectionality() const {return FunctorCall.getDirectionality();}
  void setDirectionality(Direction Dir) {FunctorCall.setDirectionality(Dir);}
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
