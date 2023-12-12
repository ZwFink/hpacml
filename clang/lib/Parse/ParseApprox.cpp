//===--- ParseApprox.cpp - Approx directives parsing ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements parsing of all Approx directives and clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ApproxClause.h"
#include "clang/Basic/Approx.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "llvm/Support/Debug.h"

#include <iostream>
#include <utility>

using namespace clang;
using namespace llvm;
using namespace approx;


static bool isMLType(Token &Tok, MLType &Kind) {
  for (unsigned i = ML_START; i < ML_END; i++) {
    enum MLType MT = (enum MLType)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxMLClause::MLName[MT])) {
      Kind = MT;
      return true;
    }
  }
  return false;
}

static bool isPerfoType(Token &Tok, PerfoType &Kind) {
  for (unsigned i = PT_START; i < PT_END; i++) {
    enum PerfoType PT = (enum PerfoType)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxPerfoClause::PerfoName[PT])) {
      Kind = PT;
      return true;
    }
  }
  return false;
}

static bool isMemoType(Token &Tok, MemoType &Kind) {
  for (unsigned i = MT_START; i < MT_END; i++) {
    enum MemoType MT = (enum MemoType)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxMemoClause::MemoName[MT])) {
      Kind = MT;
      return true;
    }
  }
  return false;
}

static bool isPetrubateType(Token &Tok, PetrubateType &Kind) {
  for (unsigned i = PETRUBATE_START ; i < PETRUBATE_END; i++) {
    enum PetrubateType PT = (enum PetrubateType) i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxPetrubateClause::PetrubateName[PT])) {
      Kind = PT;
      return true;
    }
  }
  return false;
}

static bool isDeclType(Token &Tok, DeclKind &Kind) {
  for (unsigned i = DK_START; i < DK_END; i++) {
    enum DeclKind DT = (enum DeclKind)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxDecl::Name[DT])) {
      Kind = DT;
      return true;
    }
  }
  return false;
}

Scope *getNonApproxScope(Scope *base) {
  Scope *S = base;
  while (S && S->isApproxScope())
    S = S->getParent();

  if(!S)
    llvm_unreachable("Base scope is approx?");
  return S;
}

bool Parser::ParseApproxVarList(SmallVectorImpl<Expr *> &Vars,
                                SourceLocation &ELoc) {
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after))
    return true;

  while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::colon) &&
         Tok.isNot(tok::annot_pragma_approx_end)) {
    ExprResult VarExpr =
        Actions.CorrectDelayedTyposInExpr(ParseAssignmentExpression());
        if (VarExpr.isUsable()) {
      Vars.push_back(VarExpr.get());
    } else {
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_approx_end,
                StopBeforeMatch);
      return false;
    }
    bool isComma = Tok.is(tok::comma);
    if (isComma)
      ConsumeToken();
    else if (Tok.isNot(tok::r_paren) &&
             Tok.isNot(tok::annot_pragma_approx_end) && Tok.isNot(tok::colon)) {
      Diag(Tok, diag::err_pragma_approx_expected_punc);
      SkipUntil(tok::annot_pragma_approx_end, StopBeforeMatch);
      return false;
    }
  }
  ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();
  return true;
}

ApproxClause *Parser::ParseApproxPerfoClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LParenLoc = ConsumeAnyToken();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ApproxPerfoClause::PerfoName[CK].c_str()))
    return nullptr;

  PerfoType PT;
  if (!isPerfoType(Tok, PT)){
    return nullptr;
  }
  /// Consume Perf Type
  ConsumeAnyToken();

  ///Parse ':'
  if (Tok.isNot(tok::colon)){
    return nullptr;
  }
  /// Consuming ':'
  ConsumeAnyToken();
  SourceLocation ExprLoc = Tok.getLocation();
  ExprResult Val(ParseExpression());
  Val = Actions.ActOnFinishFullExpr(Val.get(), ExprLoc, false);
  SourceLocation ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();
  ApproxVarListLocTy Locs(Loc, LParenLoc, ELoc);

  return Actions.ActOnApproxPerfoClause(CK, PT, Locs, Val.get());
}

ApproxClause *Parser::ParseApproxPetrubateClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LParenLoc = ConsumeAnyToken();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ApproxClause::Name[CK].c_str()))
    return nullptr;

  PetrubateType PT;
  if (!isPetrubateType(Tok, PT)){
    return nullptr;
  }
  /// Consume Memo Type
  ConsumeAnyToken();

  SourceLocation ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();
  ApproxVarListLocTy Locs(Loc, LParenLoc, ELoc);
  return Actions.ActOnApproxPetrubateClause(CK, PT, Locs);
}


ApproxClause *Parser::ParseApproxMemoClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LParenLoc = ConsumeAnyToken();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ApproxClause::Name[CK].c_str()))
    return nullptr;

  MemoType MT;
  if (!isMemoType(Tok, MT)){
    return nullptr;
  }
  /// Consume Memo Type
  ConsumeAnyToken();

  SourceLocation ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();
  ApproxVarListLocTy Locs(Loc, LParenLoc, ELoc);
  return Actions.ActOnApproxMemoClause(CK, MT, Locs);
}

ApproxClause *Parser::ParseApproxMLClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LParenLoc = ConsumeAnyToken();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ApproxClause::Name[CK].c_str()))
    return nullptr;

  MLType MT;
  if (!isMLType(Tok, MT)){
    return nullptr;
  }
  /// Consume Memo Type
  ConsumeAnyToken();

  SourceLocation ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();
  ApproxVarListLocTy Locs(Loc, LParenLoc, ELoc);
  return Actions.ActOnApproxMLClause(CK, MT, Locs);
}

//These claues are not used a.t.m
ApproxClause *Parser::ParseApproxDTClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation ELoc = ConsumeAnyToken();
  ApproxVarListLocTy Locs(Loc, SourceLocation(), ELoc);
  return Actions.ActOnApproxDTClause(CK, Locs);
}

StmtResult Parser::ParseApproxDecl(DeclKind DK) {
  
  // Are we declaring a tensor_functor or a tensor?
  Token DeclaredTypeToken = Tok;
  auto DeclTypeLoc = ConsumeAnyToken();

  if(!isDeclType(DeclaredTypeToken, DK)){
    return StmtError();
  }

  SourceLocation LParenLoc = Tok.getLocation();

  clang::Decl *DeclResult = nullptr;

  if(DK == approx::DeclKind::DK_T) {
    DeclResult = ParseApproxTensorDecl(DeclKind::DK_T, LParenLoc);
  }
  else if(DK == approx::DeclKind::DK_TF) {
    DeclResult =  ParseApproxTensorFunctorDecl(DeclKind::DK_TF, LParenLoc);
  }
  else {
    llvm_unreachable("Unknown DeclType");
  }

  DeclGroupPtrTy DG = Actions.ConvertDeclToDeclGroup(DeclResult);

  StmtResult Res = Actions.ActOnDeclStmt(DG, LParenLoc, DeclResult->getEndLoc());

  Res.get()->printPretty(dbgs(), nullptr, Actions.getPrintingPolicy());
  return Res;
}

ExprResult
Parser::ParseApproxTensorDeclArgs(ExprResult LHS, SourceLocation OpenBracketLoc) {
  static int DeclParseDepth = 0;
  int myParseDepth = 0;
  llvm::SmallVector<Expr*, 8> Indirections;
  llvm::SmallVector<Expr*, 8> Slices;

  Indirections.push_back(LHS.get());

  // this can be either bunch of arrayslices representing indirection,
  // or it'll be the start of an actual slice, such as '0'
  // if the current token is not a colon, then we've parsed an indirection.
  ExprResult RHS = ParseExpression();

  if(Tok.isNot(tok::colon)) {
    Indirections.push_back(RHS.get());
    return Actions.ActOnApproxArraySliceExpr(Indirections, OpenBracketLoc, Slices, Tok.getLocation(), 1);
  }

  // I've found a colon and so I have partially parsed the first slice in the ndtensor slice
  // Our parse for this so far looks like (assume some indirection): functor_name(a[b[0:
  llvm::SmallVector<Expr*, 8> SliceParts;
  SliceParts.push_back(RHS.get());
  ParseApproxNDTensorSlice(Slices, SliceParts, tok::r_square);

  return Actions.ActOnApproxArraySliceExpr(Indirections, OpenBracketLoc, Slices, Tok.getLocation(), 1);
}

ApproxDeclareTensorFunctorDecl *Parser::ParseApproxTensorFunctorDecl(DeclKind DK, SourceLocation Loc) {
  unsigned ScopeFlags = Scope::ApproxTensorFunctorDeclScope | getCurScope()->getFlags();
  ParseScope ApproxScope(this, ScopeFlags);
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, "tensor_functor"))
  {
    return nullptr;
  }

  auto LParenLoc = T.getOpenLocation();

  // get the name
  SourceLocation NameLocation = Tok.getLocation();
  auto NameID = Tok.getIdentifierInfo();

  ConsumeAnyToken(); // skip past the name
  // skip past the colon
  ConsumeAnyToken();

  // parse the LHS of the tensor functor, looks like [...] = (...)
  auto Begin = Tok.getLocation();
  ApproxNDTensorSlice Slices;
  BalancedDelimiterTracker T2(*this, tok::l_square);
  if(T2.expectAndConsume(diag::err_expected_lsquare_after, "tensor_functor"))
    return nullptr;
  ParseApproxNDTensorSlice(Slices, tok::r_square);
  auto RSQLoc = Tok.getLocation();
  llvm::SmallVector<Expr*,1> Base;
  ExprResult LHSRes = Actions.ActOnApproxArraySliceExpr(Base, Begin, Slices, RSQLoc);
  if(LHSRes.isInvalid())
    return nullptr;
  Expr *LHS = LHSRes.get();

  if(T2.consumeClose())
    llvm_unreachable("Expected a close bracket");

  // consume the token '='
  ConsumeAnyToken();

  BalancedDelimiterTracker T3(*this, tok::l_paren);
  if(T3.expectAndConsume(diag::err_expected_lparen_after, "tensor_functor"))
    return nullptr;

  // parse the RHS of the tensor functor, looks like ([...], [...], ...)
  // each element of the vector is an ApproxArraySliceExpr
  llvm::SmallVector<Expr *, 8> IptArrayExprs;
  ParseExpressionList(IptArrayExprs);

  T3.consumeClose();


  if(T.consumeClose())
    llvm_unreachable("Expected a close paren");

  ApproxVarListLocTy Locs(Loc, LParenLoc, T.getCloseLocation());

  ApproxScope.Exit();
  Scope *S = getNonApproxScope(getCurScope());
  return Actions.ActOnApproxTFDecl(DK, S, NameID, LHS, IptArrayExprs, Locs);
}

// we may have partially parsed the first slice in the ND tensor slice.
void Parser::ParseApproxNDTensorSlice(SmallVectorImpl<Expr *>& Slices, SmallVectorImpl<Expr*>& FirstSliceParts, tok::TokenKind EndToken) { 

  // if we haven't yet parsed any part of it, go ahead and parse like regular.
  if(FirstSliceParts.size() == 0) {
    ParseApproxNDTensorSlice(Slices, EndToken);
    return;
  }

  SourceLocation ColonLocFirst = SourceLocation();
  SourceLocation ColonLocSecond = SourceLocation();

  ExprResult StartResult;
  ExprResult StopResult;
  ExprResult StepResult;

  // if we have already parsed the start
  if(FirstSliceParts.size() == 1) {
    StartResult = ExprResult(FirstSliceParts[0]);
    std::tie(ColonLocFirst, StopResult) = ParsePartOfSliceExpression();
  }
  // if we have already parsed start and stop
  else if(FirstSliceParts.size() == 2) {
    StopResult = ExprResult(FirstSliceParts[1]);
  }

  if(Tok.is(tok::colon)) {
    std::tie(ColonLocSecond, StepResult) = ParsePartOfSliceExpression();
  }

  Slices.push_back(Actions
                       .ActOnApproxSliceExpr(
                           StartResult.get()->getBeginLoc(), StartResult.get(),
                           ColonLocFirst, StopResult.get(), ColonLocSecond,
                           StepResult.get(), StopResult.get()->getEndLoc())
                       .get());

  if(Tok.is(tok::comma)) {
    ConsumeAnyToken();
    ParseApproxNDTensorSlice(Slices, EndToken);
  }
}
void Parser::ParseApproxNDTensorSlice(SmallVectorImpl<Expr *>& Slices, tok::TokenKind EndToken) {
  unsigned ScopeFlags = Scope::ApproxSliceScope | getCurScope()->getFlags();
  ParseScope ApproxScope(this, ScopeFlags);

  while (Tok.isNot(EndToken) && Tok.isNot(tok::r_square)) {
    // Parse a slice expression
    auto Expr = ParseSliceExpression();
    ApproxSliceExpr *SliceRes = dyn_cast<ApproxSliceExpr>(Expr.get());

    if (Expr.isInvalid()) {
      llvm::dbgs() << "The stride expression is invalid\n";
    }

    Slices.push_back(Expr.get());

    if (Tok.is(EndToken)) {
      break;
    }

    if (Tok.isNot(tok::comma)) {
      llvm_unreachable("Expected a comma");
    }
    
    // Consume the comma
    ConsumeAnyToken();
  }

}

void Parser::ParseApproxNDTensorSliceCollection(ApproxNDTensorSliceCollection &Slices)
{
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::r_paren);
  if(T.expectAndConsume(diag::err_expected_lparen_after, "NDTensorSliceCollection"))
    llvm_unreachable("Expected a left paren");

  SourceLocation LParenLoc = T.getOpenLocation();

  while (Tok.isNot(tok::r_paren)) {
    // Parse a slice expression
    ApproxNDTensorSlice Slice;

    int depth = 0;
    while(Tok.is(tok::l_square)) {
      depth += 1;
      ConsumeAnyToken();
    }

    ParseApproxNDTensorSlice(Slice, tok::r_square);

    while(Tok.is(tok::r_square)) {
      depth -= 1;
      ConsumeAnyToken();
    }
    
    if(depth != 0)
      llvm_unreachable("Mismatched brackets");

    Slices.push_back(Slice);

    if (Tok.is(tok::r_paren)) {
      break;
    }

    // If it's not right paren, should be comma
    if(Tok.isNot(tok::comma))
    {
      llvm_unreachable("Expected a comma");
    }

    // Consume the comma between slices: [...], [...], ...
    ConsumeAnyToken();
  }

  SourceLocation RParenLoc = T.getCloseLocation();
  if(T.consumeClose())
  {
    llvm_unreachable("Expected a close paren");
  }

}

std::pair<SourceLocation, ExprResult> Parser::ParsePartOfSliceExpression() {
  ExprResult Result = ExprError();
  SourceLocation PreceedingColonLoc = SourceLocation();

  if (Tok.is(tok::colon)) {
    PreceedingColonLoc = Tok.getLocation();
    ConsumeAnyToken();
  }

  Result = ParseAssignmentExpression();
  return std::make_pair(PreceedingColonLoc, Result);
}

ExprResult Parser::ParseSliceExpression()
{
  Expr *Start = nullptr;
  Expr *Stop = nullptr;
  Expr *Step = nullptr;

  SourceLocation StartLocation = SourceLocation();
  SourceLocation ColonLocFirst = SourceLocation();
  SourceLocation StopLocation = SourceLocation();
  SourceLocation ColonLocSecond = SourceLocation();
  SourceLocation StepLocation = SourceLocation();


  // TODO: Here we are potentially parsing OpenMP array section expression because
  // We should only parse up to a colon or the ']'
  if (Tok.isNot(tok::colon)) {
    auto StartResult = ParseAssignmentExpression();
    StartLocation = StartResult.get()->getBeginLoc();
    if (StartResult.isInvalid()) {
      llvm::dbgs() << "Invalid start expression\n";
      return ExprError();
    }
    Start = StartResult.get();
  }

  if(Tok.is(tok::colon))
  {
    ColonLocFirst = Tok.getLocation();
    ConsumeAnyToken();
    auto StopResult = ParseAssignmentExpression();
    StopLocation = StopResult.get()->getBeginLoc();
    if (StopResult.isInvalid()) {
      llvm::dbgs() << "Invalid stop expression\n";
      return ExprError();
    }
    Stop = StopResult.get();

  }

  if(Tok.is(tok::colon))
  {
    ColonLocSecond = Tok.getLocation();
    ConsumeAnyToken();
    auto StepResult = ParseAssignmentExpression();
    StepLocation = StepResult.get()->getBeginLoc();
    if (StepResult.isInvalid()) {
      llvm::dbgs() << "Invalid step expression\n";
      return ExprError();
    }
    Step = StepResult.get();
  }

  return Actions.ActOnApproxSliceExpr(StartLocation, Start, ColonLocFirst,
                                      Stop, ColonLocSecond, Step,
                                      StopLocation);
}

ApproxDeclareTensorDecl *Parser::ParseApproxTensorDecl(DeclKind DK, SourceLocation Loc) {
  unsigned ScopeFlags = getCurScope()->getFlags() | Scope::ApproxTensorDeclScope;
  ParseScope ApproxScope(this, ScopeFlags);
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if(T.expectAndConsume(diag::err_expected_lparen_after, "tensor_decl"))
    return nullptr;
  
  auto LParenLoc = T.getOpenLocation();
  SourceLocation NameLocation = Tok.getLocation();
  auto TensorName = Tok.getIdentifierInfo();

  // Skip past the name
  ConsumeAnyToken();

  if(Tok.isNot(tok::colon))
    llvm_unreachable("Expected a colon");

  ConsumeAnyToken();

  SourceLocation TFNameLoc = Tok.getLocation();
  auto TFName = Tok.getIdentifierInfo();

  ConsumeAnyToken();

  BalancedDelimiterTracker T2(*this, tok::l_paren);
  T2.consumeOpen();

  llvm::SmallVector<Expr *, 8> IptArrayExprs;
  auto TFCall = ParseExpressionList(IptArrayExprs);
  T2.consumeClose();

  T.consumeClose();
  ApproxVarListLocTy Locs(Loc, LParenLoc, T.getCloseLocation());
  Scope *S = getNonApproxScope(getCurScope());
  return Actions.ActOnApproxTensorDecl(DK, S, TFName, TensorName, IptArrayExprs, Locs);
}

ApproxDeclareTensorDecl *Parser::ParseApproxTensorDeclAnonymous(DeclKind DK, SourceLocation Loc) {
  unsigned ScopeFlags = getCurScope()->getFlags() | Scope::ApproxTensorDeclScope;
  ParseScope ApproxScope(this, ScopeFlags);
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if(T.expectAndConsume(diag::err_expected_lparen_after, "out"))
    return nullptr;

  SourceLocation TFNameLoc = Tok.getLocation();
  auto TFName = Tok.getIdentifierInfo();

  ConsumeAnyToken();

  BalancedDelimiterTracker T2(*this, tok::l_paren);
  T2.consumeOpen();

  llvm::SmallVector<Expr *, 8> IptArrayExprs;
  auto TFCall = ParseExpressionList(IptArrayExprs);
  T2.consumeClose();

  ApproxVarListLocTy Locs(Loc, T2.getOpenLocation(), T2.getCloseLocation());
  // get a dummy identifier for the tensor name
  IdentifierInfo *TensorName = PP.getIdentifierInfo("anonymous_tensor");
  Scope *S = getNonApproxScope(getCurScope());

  T.consumeClose();

  return Actions.ActOnApproxTensorDecl(DK, S, TFName, TensorName, IptArrayExprs, Locs);
}

ApproxClause *Parser::ParseApproxNNClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation ELoc = ConsumeAnyToken();
  ApproxVarListLocTy Locs(Loc, SourceLocation(), ELoc);
  return Actions.ActOnApproxNNClause(CK, Locs);
}
//~These claues are not used/implemented a.t.m

ApproxClause *Parser::ParseApproxUserClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation ELoc = ConsumeAnyToken();
  ApproxVarListLocTy Locs(Loc, SourceLocation(), ELoc);
  return Actions.ActOnApproxUserClause(CK, Locs);
}

ApproxClause *Parser::ParseApproxIfClause(ClauseKind CK) {
  //Start Location
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LParenLoc = ConsumeAnyToken();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ApproxClause::Name[CK].c_str()))
    return nullptr;

  SourceLocation ExprLoc = Tok.getLocation();
  ExprResult LHS(ParseCastExpression(AnyCastExpr, false, NotTypeCast));
  ExprResult Val = ParseRHSOfBinaryExpression(LHS, prec::Conditional);
  Val = Actions.ActOnFinishFullExpr(Val.get(), ExprLoc, false );

  SourceLocation ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();

  if ( Val.isInvalid() )
    return nullptr;

  ApproxVarListLocTy Locs(Loc, LParenLoc, ELoc);
  return Actions.ActOnApproxIfClause(CK, Locs, Val.get());
}

ApproxClause *Parser::ParseApproxInClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeAnyToken();
  SourceLocation RLoc;
  SmallVector<Expr *, 8> Vars;
  unsigned ScopeFlags = Scope::ApproxArraySectionScope | getCurScope()->getFlags();
  ParseScope ApproxScope(this, ScopeFlags);
  if (!ParseApproxVarList(Vars, RLoc)) {
    return nullptr;
  }
  ApproxVarListLocTy Locs(Loc, LOpen, RLoc);
  return Actions.ActOnApproxVarList(CK, Vars, Locs);
}

ApproxClause *Parser::ParseApproxOutClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeAnyToken();
  SourceLocation RLoc;
  SmallVector<Expr *, 8> Vars;

  if(true) {
    auto *TensorDecl = ParseApproxTensorDeclAnonymous(approx::DeclKind::DK_T,
                                                      Tok.getLocation());
    CXXScopeSpec ScopeSpec;
    SourceLocation TemplateKWLoc;
    UnqualifiedId Name;
    Name.setIdentifier(TensorDecl->getIdentifier(), TensorDecl->getLocation());
    ExprResult TensorRef = Actions.ActOnIdExpression(
        getCurScope(), ScopeSpec, TemplateKWLoc, Name, false, false);

    llvm::SmallVector<Decl *, 8> Decls;
    llvm::SmallVector<Expr *, 8> Exprs;

    Decls.push_back(cast<Decl>(TensorDecl));
    Exprs.push_back(TensorRef.get());

    ExprResult Body = Actions.ActOnApproxCompoundExpr(Decls, Exprs);

    Vars.push_back(Body.get());
  } else {
    unsigned ScopeFlags =
        Scope::ApproxArraySectionScope | getCurScope()->getFlags();
    ParseScope ApproxScope(this, ScopeFlags);

    if (!ParseApproxVarList(Vars, RLoc)) {
      return nullptr;
    }
  }

  ApproxVarListLocTy Locs(Loc, LOpen, RLoc);
  return Actions.ActOnApproxVarList(CK, Vars, Locs);
}

ApproxClause *Parser::ParseApproxInOutClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeAnyToken();
  SourceLocation RLoc;
  SmallVector<Expr *, 8> Vars;
  if (!ParseApproxVarList(Vars, RLoc)) {
    return nullptr;
  }
  ApproxVarListLocTy Locs(Loc, LOpen, RLoc);
  return Actions.ActOnApproxVarList(CK, Vars, Locs);
}

ApproxClause *Parser::ParseApproxLabelClause(ClauseKind CK) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LParenLoc = ConsumeAnyToken();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, ApproxClause::Name[CK].c_str()))
    return nullptr;

  SourceLocation ExprLoc = Tok.getLocation();
  ExprResult Val(ParseExpression());
  Val = Actions.ActOnFinishFullExpr(Val.get(), ExprLoc, false);

  SourceLocation ELoc = Tok.getLocation();
  if (!T.consumeClose())
    ELoc = T.getCloseLocation();

  ApproxVarListLocTy Locs(Loc, LParenLoc, ELoc);

  return Actions.ActOnApproxLabelClause(CK, Locs, Val.get());
}

bool isApproxClause(Token &Tok, ClauseKind &Kind) {
  for (unsigned i = CK_START; i < CK_END; i++) {
    enum ClauseKind CK = (enum ClauseKind)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxClause::Name[CK])) {
      Kind = CK;
      return true;
    }
  }
  return false;
}

bool isApproxDecl(Token &Tok, DeclKind &Kind) {
  for (unsigned i = DK_START; i < DK_END; i++) {
    enum DeclKind DK = (enum DeclKind)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxDecl::Name[DK])) {
      Kind = DK;
      return true;
    }
  }
  return false;
}

StmtResult Parser::ParseApproxDirective(ParsedStmtContext StmtCtx) {
  assert(Tok.is(tok::annot_pragma_approx_start));
  /// This should be a function call;
  // assume approx array section scope
#define PARSER_CALL(method) ((*this).*(method))

  StmtResult Directive = StmtError();
  SourceLocation DirectiveStart = Tok.getLocation();
  SmallVector<ApproxClause*, CK_END> Clauses;
  SmallVector<StmtResult, DK_END> Decls;

  /// I am consuming the pragma identifier atm.
  ConsumeAnyToken();

  if(Tok.is(tok::identifier)) {
    if(Tok.getIdentifierInfo()->getName().equals("declare"))
      ConsumeAnyToken();
  }

  SourceLocation ClauseStartLocation = Tok.getLocation();

  /// we do not support just
  /// #pragma approx
  /// we need extra information. So just
  /// return with an error
  if (Tok.is(tok::eod) || Tok.is(tok::eof)) {
    PP.Diag(Tok, diag::err_pragma_approx_expected_directive);
    ConsumeAnyToken();
    return Directive;
  }

  ClauseKind CK;
  DeclKind DK;
  while (Tok.isNot(tok::annot_pragma_approx_end)) {
    if (isApproxClause(Tok, CK)) {
      ApproxClause *Clause = PARSER_CALL(ParseApproxClause[CK])(CK);
      if (!Clause) {
        SkipUntil(tok::annot_pragma_approx_end);
        return Directive;
      }
      Clauses.push_back(Clause);
    } else if(isApproxDecl(Tok, DK)) {
      auto Decl = ParseApproxDecl(DK);

      if(!Decl.get()) {
        SkipUntil(tok::annot_pragma_approx_end);
        return Directive;
      }
      Decls.push_back(Decl);
      }else {
      PP.Diag(Tok, diag::err_pragma_approx_unrecognized_directive);
      SkipUntil(tok::annot_pragma_approx_end);
      return Directive;
    }
  }

  /// Update the end location of the directive.
  SourceLocation DirectiveEnd = Tok.getLocation();
  ConsumeAnnotationToken();
  ApproxVarListLocTy Locs(DirectiveStart, ClauseStartLocation, DirectiveEnd);

  if(Decls.size() > 0) {
    assert(Decls.size() == 1 && "Only one decl is supported");
    assert(Clauses.size() == 0 && "Clauses and decls are mutually exclusive");
    return Decls[0];
  }

  Stmt *AssociatedStmtPtr = nullptr;

    // Start captured region sema, will end withing ActOnApproxDirective.
  Actions.ActOnCapturedRegionStart(Tok.getEndLoc(), getCurScope(), CR_Default,
                                   /* NumParams = */ 1);
  StmtResult AssociatedStmt =
      (Sema::CompoundScopeRAII(Actions), ParseStatement());
  AssociatedStmtPtr = AssociatedStmt.get();
  Directive = Actions.ActOnApproxDirective(AssociatedStmtPtr, Clauses, Locs);
  return Directive;
}
