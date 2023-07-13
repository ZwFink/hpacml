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

static bool isDeclType(Token &Tok, DeclType &Kind) {
  for (unsigned i = DT_START; i < DT_END; i++) {
    enum DeclType DT = (enum DeclType)i;
    if (Tok.getIdentifierInfo()->getName().equals(ApproxDeclClause::DeclName[DT])) {
      Kind = DT;
      return true;
    }
  }
  return false;
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

ApproxClause *Parser::ParseApproxDeclClause(ClauseKind CK) {
  
  // Consume 'declare'
  auto DeclareTokenLocation = ConsumeAnyToken();
  SourceLocation Loc = Tok.getLocation();

  // Are we declaring a tensor_functor or a tensor?
  Token DeclaredTypeToken = Tok;
  auto DeclTypeLoc = ConsumeAnyToken();
  auto DeclaredTypeString = DeclaredTypeToken.getIdentifierInfo()->getName();

  DeclType DT;
  if(!isDeclType(DeclaredTypeToken, DT)){
    return nullptr;
  }

  SourceLocation LParenLoc = Tok.getLocation();
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_approx_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after, DeclaredTypeString.data()))
  {
    return nullptr;
  }

  if(DT == approx::DeclType::DT_TENSOR) {
    return ParseApproxTensorDeclClause(CK, Loc, LParenLoc, T);
  }
  else if(DT == approx::DeclType::DT_TENSOR_fUNCTOR) {
    return ParseApproxTensorFunctorDeclClause(CK, Loc, LParenLoc, T);
  }
  else {
    llvm_unreachable("Unknown DeclType");
  }
}

ApproxClause *Parser::ParseApproxTensorFunctorDeclClause(ClauseKind CK, SourceLocation Loc, SourceLocation LParenLoc, BalancedDelimiterTracker T) {
  approxScope = ApproxScope::APPROX_TENSOR_SLICE;

  // get the name
  SourceLocation NameLocation = Tok.getLocation();
  auto Name = Tok.getIdentifierInfo()->getName();

  ConsumeAnyToken(); // skip past the name
  // skip past the colon
  ConsumeAnyToken();

  // parse the LHS of the tensor functor, looks like [...] = (...)
  auto Begin = Tok.getLocation();
  ApproxNDTensorSlice Slices;
  ParseApproxNDTensorSlice(Slices, tok::r_square);

  // consume the token '='
  ConsumeAnyToken();

  ApproxNDTensorSliceCollection RHSSlices;
  // parse the RHS of the tensor functor, looks like ([...], [...], ...)
  ParseApproxNDTensorSliceCollection(RHSSlices);

  if(T.consumeClose())
    llvm_unreachable("Expected a close paren");

  ApproxVarListLocTy Locs(Loc, LParenLoc, T.getCloseLocation());
  // Do we need to pass in the declared type?
  return Actions.ActOnApproxTFDeclClause(CK, Name, Slices, RHSSlices, Locs);
}

void Parser::ParseApproxNDTensorSlice(SmallVectorImpl<Expr *>& Slices, tok::TokenKind EndToken) {
  BalancedDelimiterTracker T(*this, tok::l_square, tok::r_square);
  if(T.expectAndConsume(diag::err_expected_lsquare_after, "NDTensorSlice"))
    llvm_unreachable("Expected a left bracket");

  SourceLocation LBLoc = T.getOpenLocation();

  while (Tok.isNot(EndToken) && Tok.isNot(tok::r_square)) {
    // Parse a slice expression
    auto Expr = ParseSliceExpression();

    if (Expr.isInvalid()) {
      llvm::dbgs() << "The slide expression is invalid\n";
    }

    Slices.push_back(Expr.get());

    if (Tok.is(EndToken)) {
      break;
    }

    if (Tok.isNot(tok::comma)) {
      llvm_unreachable("Expected a comma");
    }
  }

  if(T.consumeClose())
  {
    llvm_unreachable("Expected a close bracket");
  }
  SourceLocation RBLoc = T.getCloseLocation();

  ApproxSliceExpr *FirstSlice = dyn_cast<ApproxSliceExpr>(Slices.front());
  ApproxSliceExpr *LastSlice = dyn_cast<ApproxSliceExpr>(Slices.back());
  FirstSlice->setLBracketLoc(LBLoc);
  LastSlice->setRBracketLoc(RBLoc);
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
    ParseApproxNDTensorSlice(Slice, tok::r_square);

    Slices.push_back(Slice);

    if (Tok.is(tok::r_paren)) {
      break;
    }

    // If it's not right paren, should be comma
    if(Tok.isNot(tok::comma))
    {
      llvm_unreachable("Expected a comma");
    }

    ConsumeAnyToken();
  }

  SourceLocation RParenLoc = T.getCloseLocation();
  if(T.consumeClose())
  {
    llvm_unreachable("Expected a close paren");
  }

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
    auto StartResult = ParseExpression();
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
    auto StopResult = ParseExpression();
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
    auto StepResult = ParseExpression();
    StepLocation = StepResult.get()->getBeginLoc();
    if (StepResult.isInvalid()) {
      llvm::dbgs() << "Invalid step expression\n";
      return ExprError();
    }
    Step = StepResult.get();
  }

  return Actions.ActOnApproxSliceExpr(SourceLocation(), Start, ColonLocFirst,
                                      Stop, ColonLocSecond, Step,
                                      SourceLocation());
}

ApproxClause *Parser::ParseApproxTensorDeclClause(ClauseKind CK, SourceLocation Loc, SourceLocation LParenLoc, BalancedDelimiterTracker T) {
  llvm::dbgs() << "Recognized a Tensor Decl\n";
  // Consume Decl Type (Note: This might be something we want to recurse on later)
  auto TensorName = Tok;
  ConsumeAnyToken();

  // Do we need to pass in the declared type?
  ApproxVarListLocTy Locs(Loc, LParenLoc, LParenLoc);
  // return Actions.ActOnApproxDeclClause(CK, Locs);
  llvm_unreachable("Not implemented yet");
  return nullptr;
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
  if (!ParseApproxVarList(Vars, RLoc)) {
    return nullptr;
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

StmtResult Parser::ParseApproxDirective(ParsedStmtContext StmtCtx) {
  assert(Tok.is(tok::annot_pragma_approx_start));
  /// This should be a function call;
  // assume approx array section scope
  approxScope = ApproxScope::APPROX_ARRAY_SECTION;
#define PARSER_CALL(method) ((*this).*(method))

  StmtResult Directive = StmtError();
  SourceLocation DirectiveStart = Tok.getLocation();
  SmallVector<ApproxClause*, CK_END> Clauses;

  /// I am consuming the pragma identifier atm.
  ConsumeAnyToken();

  SourceLocation ClauseStartLocation = Tok.getLocation();

  /// we do not support just
  /// #pragma approx
  /// we need extra information. So just
  /// return with an error
  if (Tok.is(tok::eod) || Tok.is(tok::eof)) {
    PP.Diag(Tok, diag::err_pragma_approx_expected_directive);
    ConsumeAnyToken();
    approxScope = ApproxScope::APPROX_NONE;
    return Directive;
  }

  ClauseKind CK;
  while (Tok.isNot(tok::annot_pragma_approx_end)) {
    if (isApproxClause(Tok, CK)) {
      ApproxClause *Clause = PARSER_CALL(ParseApproxClause[CK])(CK);
      if (!Clause) {
        SkipUntil(tok::annot_pragma_approx_end);
        approxScope = ApproxScope::APPROX_NONE;
        return Directive;
      }
      Clauses.push_back(Clause);
    } else {
      PP.Diag(Tok, diag::err_pragma_approx_unrecognized_directive);
      SkipUntil(tok::annot_pragma_approx_end);
      approxScope = ApproxScope::APPROX_NONE;
      return Directive;
    }
  }

  /// Update the end location of the directive.
  SourceLocation DirectiveEnd = Tok.getLocation();
  ConsumeAnnotationToken();
  ApproxVarListLocTy Locs(DirectiveStart, ClauseStartLocation, DirectiveEnd);

  // Start captured region sema, will end withing ActOnApproxDirective.
  Actions.ActOnCapturedRegionStart(Tok.getEndLoc(), getCurScope(), CR_Default, /* NumParams = */1);
  StmtResult AssociatedStmt = (Sema::CompoundScopeRAII(Actions), ParseStatement());
  Directive = Actions.ActOnApproxDirective(AssociatedStmt.get(), Clauses, Locs);
  approxScope = ApproxScope::APPROX_NONE;
  return Directive;
}
