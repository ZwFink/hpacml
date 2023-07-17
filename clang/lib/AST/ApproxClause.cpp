//===- ApproxClause.cpp - Classes for Approx clauses ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ApproxClause methods in ApproxClause.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ApproxClause.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Approx.h"
#include "llvm/Support/Debug.h"

using namespace clang;
using namespace approx;
using namespace llvm;

const std::string ApproxClause::Name[approx::CK_END] = {
    "perfo", "memo", "dt", "declare", "nn", "user", "if", "in", "out", "inout", "label", "petrubate", "ml"};

const std::string ApproxPerfoClause::PerfoName[approx::PT_END] = {
    "small", "large", "rand", "init", "fini"};

const std::string ApproxDeclClause::DeclName[approx::DT_END] = {
    "tensor_functor", "tensor"};

const std::string ApproxMemoClause::MemoName[approx::MT_END] = {
    "in", "out"};

const std::string ApproxMLClause::MLName[approx::ML_END] = {
    "online", "offline", "infer"};

const std::string ApproxPetrubateClause::PetrubateName[approx::PETRUBATE_END] = {
    "in", "out", "inout"};

ApproxInClause *ApproxInClause::Create(const ASTContext &C,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc,
                                       ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  ApproxInClause *Clause =
      new (Mem) ApproxInClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

ApproxInClause *ApproxInClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) ApproxInClause(N);
}

ApproxOutClause *ApproxOutClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  ApproxOutClause *Clause =
      new (Mem) ApproxOutClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

ApproxOutClause *ApproxOutClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) ApproxOutClause(N);
}

ApproxInOutClause *ApproxInOutClause::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation LParenLoc,
                                             SourceLocation EndLoc,
                                             ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  ApproxInOutClause *Clause =
      new (Mem) ApproxInOutClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

ApproxInOutClause *ApproxInOutClause::CreateEmpty(const ASTContext &C,
                                                  unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) ApproxInOutClause(N);
}

void ApproxClausePrinter::VisitApproxPerfoClause(ApproxPerfoClause *Node) {
  OS << Node->getAsString() << " ";
}

void ApproxClausePrinter::VisitApproxTensorDeclClause(ApproxTensorDeclClause *Node) {
  llvm::ArrayRef<Expr*> Slices = Node->getArraySlices();
  OS << "declare tensor(" << Node->getTensorName() << ": " 
  << Node->getTFName() << "(";
  for(unsigned i = 0; i < Slices.size(); i++){
    if(i != 0)
      OS << ", ";
    Slices[i]->printPretty(OS, nullptr, Policy, 0);
  }
  OS << "))";
}

void ApproxClausePrinter::VisitApproxTensorFunctorDeclClause(ApproxTensorFunctorDeclClause *Node) {
  OS << "declare tensor_functor(" << Node->getFunctorName() << ": ";
  OS << "[";
  auto LHS = Node->getLHSSlice();
  for(unsigned i = 0; i < LHS.size(); i++){
    if(i != 0)
      OS << ", ";
    LHS[i]->printPretty(OS, nullptr, Policy, 0);
  }
  OS << "]";
  OS << " = (";
  ApproxNDTensorSliceCollection &RHS = Node->getRHSSlices();
  for(unsigned j = 0 ; j < RHS.size(); j++){
    auto &Slices = RHS[j];
    if(j != 0)
      OS << ", ";
    OS << "[";
    for(unsigned i = 0; i < Slices.size(); i++){
      if(i != 0)
        OS << ", ";
      Slices[i]->printPretty(OS, nullptr, Policy, 0);
    }
    OS << "]";
  }

  OS << "))";

}

void ApproxClausePrinter::VisitApproxMemoClause(ApproxMemoClause *Node) {
  OS << Node->getAsString() << "(" << Node->getMemoTypeAsString() << ") ";
}

void ApproxClausePrinter::VisitApproxMLClause(ApproxMLClause *Node) {
  OS << Node->getAsString() << "(" << Node->getMLTypeAsString() << ") ";
}

void ApproxClausePrinter::VisitApproxPetrubateClause(ApproxPetrubateClause *Node) {
  OS << Node->getAsString() << "(" << Node->getPetrubateTypeAsString() << ") ";
}

void ApproxClausePrinter::VisitApproxDTClause(ApproxDTClause *Node) {
  OS << Node->getAsString() << " ";
}

void ApproxClausePrinter::VisitApproxNNClause(ApproxNNClause *Node) {
  OS << Node->getAsString() << " ";
}

void ApproxClausePrinter::VisitApproxUserClause(ApproxUserClause *Node) {
  OS << Node->getAsString() << " ";
}

void ApproxClausePrinter::VisitApproxIfClause(ApproxIfClause *Node) {
  OS << Node->getAsString() << "(";
  Node->getCondition()->printPretty(OS,nullptr, Policy, 0);
  OS << ")";
}

template<typename T>
  void ApproxClausePrinter::VisitApproxVarList(T *Node){
    for (typename T::varlist_iterator I = Node->varlist_begin(),
                                      E = Node->varlist_end();
        I != E; ++I){
          if ( I!= Node->varlist_begin())
            OS << "," ;
          assert(*I && "Expected non-nul Stmt");
          (*I)->printPretty(OS,nullptr,Policy, 0);
        }
  }

  void ApproxClausePrinter::VisitApproxInClause(ApproxInClause *Node) {
    OS << Node->getAsString();
    if ( !Node->varlist_empty() ){
      OS << "(";
      VisitApproxVarList(Node);
      OS << ")";
    }
    else{
      OS << "()";
    }
    OS << " ";
  }

void ApproxClausePrinter::VisitApproxOutClause(ApproxOutClause *Node) {
  OS << Node->getAsString();
  if ( !Node->varlist_empty() ){
    OS << "(";
    VisitApproxVarList(Node);
    OS << ")";
  }
  else{
    OS << "()";
  }
  OS << " ";
}

void ApproxClausePrinter::VisitApproxInOutClause(ApproxInOutClause *Node) {
  OS << Node->getAsString();
  if ( !Node->varlist_empty() ){
    OS <<"(";
    VisitApproxVarList(Node);
    OS <<")";
  }
  else{
    OS << "()";
  }
  OS <<" ";
}

void ApproxClausePrinter::VisitApproxLabelClause(ApproxLabelClause *Node){
  OS << Node->getAsString();
  OS << "(";
  Node->getLabel()->printPretty(OS,nullptr, Policy, 0);
  OS << ")";
}