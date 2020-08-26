//===----- CGApproxRuntime.cpp - Interface to Approx Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for Approx runtime code generation.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/ApproxClause.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtApprox.h"
#include "clang/Basic/Approx.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace clang;
using namespace CodeGen;

static std::pair<llvm::Value *, llvm::Value *>
getPointerAndSize(CodeGenFunction &CGF, const Expr *E) {
  const auto *OASE = dyn_cast<ApproxArrayShapingExpr>(E);
  llvm::Value *Addr;
  if (OASE) {
    const Expr *Base = OASE->getBase();
    Addr = CGF.EmitScalarExpr(Base);
  } else {
    Addr = CGF.EmitLValue(E).getPointer(CGF);
  }
  llvm::Value *SizeVal;
  QualType Ty = E->getType();
  if (OASE) {
    SizeVal = CGF.getTypeSize(OASE->getBase()->getType()->getPointeeType());
    for (const Expr *SE : OASE->getDimensions()) {
      llvm::Value *Sz = CGF.EmitScalarExpr(SE);
      Sz = CGF.EmitScalarConversion(
          Sz, SE->getType(), CGF.getContext().getSizeType(), SE->getExprLoc());
      SizeVal = CGF.Builder.CreateNUWMul(SizeVal, Sz);
    }
  } else if (const auto *ASE =
                 dyn_cast<ApproxArraySectionExpr>(E->IgnoreParenImpCasts())) {
    LValue UpAddrLVal =
        CGF.EmitApproxArraySectionExpr(ASE, /*IsLowerBound=*/false);
    llvm::Value *UpAddr =
        CGF.Builder.CreateConstGEP1_32(UpAddrLVal.getPointer(CGF), /*Idx0=*/1);
    llvm::Value *LowIntPtr = CGF.Builder.CreatePtrToInt(Addr, CGF.SizeTy);
    llvm::Value *UpIntPtr = CGF.Builder.CreatePtrToInt(UpAddr, CGF.SizeTy);
    SizeVal = CGF.Builder.CreateNUWSub(UpIntPtr, LowIntPtr);
  } else {
    SizeVal = CGF.getTypeSize(Ty);
  }
  return std::make_pair(Addr, SizeVal);
}

static FieldDecl *addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                       QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
  return Field;
}

static void getPerfoInfoType(ASTContext &C, QualType &perfoInfoTy) {
  if (perfoInfoTy.isNull()) {
    RecordDecl *perfoInfoRD = C.buildImplicitRecord("approx_perfo_info_t");
    perfoInfoRD->startDefinition();
    /// The Perfo Flags Field
    addFieldToRecordDecl(C, perfoInfoRD, C.getIntTypeForBitwidth(32, false));
    /// The approx region Id
    addFieldToRecordDecl(C, perfoInfoRD, C.getIntTypeForBitwidth(32, false));
    /// The approx step
    addFieldToRecordDecl(C, perfoInfoRD, C.getIntTypeForBitwidth(32, false));
    /// The percentage of loops to skip
    addFieldToRecordDecl(C, perfoInfoRD, C.getRealTypeForBitwidth(32, false));
    perfoInfoRD->completeDefinition();
    perfoInfoTy = C.getRecordType(perfoInfoRD);
  }
  return;
}

static void getVarInfoType(ASTContext &C, QualType &VarInfoTy) {
  if (VarInfoTy.isNull()) {
    RecordDecl *VarInfoRD = C.buildImplicitRecord("approx_var_info_t");
    VarInfoRD->startDefinition();
    /// Void pointer pointing to data values
    addFieldToRecordDecl(C, VarInfoRD, C.getIntPtrType());
    /// The directionality of this region in/out/inout
    addFieldToRecordDecl(C, VarInfoRD, C.getIntTypeForBitwidth(32, false));
    /// The type of this region float/double/int etc and respectively ptr to
    /// to vectors float/double/int
    addFieldToRecordDecl(C, VarInfoRD, C.getIntTypeForBitwidth(32, false));
   /// Number of elements of vector.
    addFieldToRecordDecl(C, VarInfoRD, C.LongTy);
    /// I will need also to create a union to hold scalar values of
    /// long/int/short/char/unsigned char/float/double and any other
    /// primary type
    addFieldToRecordDecl(C, VarInfoRD, C.getIntTypeForBitwidth(32, false));
    VarInfoRD->completeDefinition();
    VarInfoTy = C.getRecordType(VarInfoRD);
  }
  return;
}

CGApproxRuntime::CGApproxRuntime(CodeGenModule &CGM)
    : CGM(CGM), approxRegions(0), StartLoc(SourceLocation()), EndLoc(SourceLocation()) {
  ASTContext &C = CGM.getContext();
  for (unsigned i = ARG_START; i < ARG_END; i++) {
    approxRTParams.push_back(nullptr);
    approxRTTypes.push_back(nullptr);
  }
  getPerfoInfoType(C, PerfoInfoTy);
  getVarInfoType(C, VarInfoTy);
}

void CGApproxRuntime::CGApproxRuntimeEnterRegion(CodeGenFunction &CGF,
                                                 CapturedStmt &CS) {
  hasPerfo = false;
  /// Rest All info of the Runtime "state machine"
  dbgs() << "Approx Size: " << ARG_END << " " << approxRTTypes.size() << "\n";
  for (unsigned i = ARG_START; i < ARG_END; i++) {
    approxRTParams[i] = nullptr;
    approxRTTypes[i] = nullptr;
  }

  Address CapStructAddr = CGF.GenerateCapturedStmtArgument(CS);
  CodeGenFunction::CGCapturedStmtInfo CGSI(CS);
  CodeGenFunction localCGF(CGM, true);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(localCGF, &CGSI);
  llvm::Function *Fn = localCGF.GenerateCapturedStmtFunction(CS);

  /// Set Types of the function call
  approxRTTypes[AccurateFn] =
      llvm::PointerType::getUnqual(Fn->getFunctionType());
  /// The type of the perforated function is the same with the accurate one
  approxRTTypes[PerfoFn] = approxRTTypes[AccurateFn];
  approxRTTypes[CapDataPtr] = CapStructAddr.getType();
  approxRTTypes[Cond] = CGF.Builder.getInt1Ty();
  approxRTTypes[PerfoDesc] = CGM.VoidPtrTy;
  approxRTTypes[DataDesc] = CGM.VoidPtrTy;
  approxRTTypes[DataSize] = CGM.Int32Ty;

  /// Fill in parameters of runtime function call
  /// Put default values on everything.
  /// EmitClause* Will replace as necessary
  approxRTParams[AccurateFn] = Fn;
  approxRTParams[PerfoFn] = llvm::ConstantPointerNull::get(
      llvm::PointerType::getUnqual(Fn->getFunctionType()));
  approxRTParams[CapDataPtr] = CapStructAddr.getPointer();
  approxRTParams[Cond] = llvm::ConstantInt::get(CGF.Builder.getInt1Ty(), true);
  approxRTParams[PerfoDesc] = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
  approxRTParams[DataDesc] = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
  approxRTParams[DataSize] = llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);

  StartLoc = CS.getBeginLoc();
  EndLoc = CS.getEndLoc();
  return;
}

void CGApproxRuntime::CGApproxRuntimeEmitPerfoInit(
    CodeGenFunction &CGF, ApproxPerfoClause &PerfoClause) {
  enum PerfoInfoFieldID { FlagsId, ApproxRegionId, StepId, RateId };
  Value *StepVal = nullptr;
  Expr *Step = nullptr;
  ASTContext &C = CGM.getContext();
  const auto *PerfoInfoRecord = PerfoInfoTy->getAsRecordDecl();
  auto *PD =
      ImplicitParamDecl::Create(C, PerfoInfoTy, ImplicitParamDecl::Other);
  CGF.EmitVarDecl(*PD);
  Address PerfoStructAddress = CGF.GetAddrOfLocalVar(PD);
  Step = PerfoClause.getStep();
  if (const auto *PreInit = cast_or_null<DeclStmt>(PerfoClause.getPreInit())) {
    for (const auto *D : PreInit->decls()) {
      CGF.EmitVarDecl(cast<VarDecl>(*D));
    }
  }
  StepVal = CGF.EmitScalarExpr(Step);
  Value *PerfoType =
      llvm::ConstantInt::get(CGM.Int32Ty, PerfoClause.getPerfoType(), false);
  Value *RGId = llvm::ConstantInt::get(CGM.Int32Ty, approxRegions, false);
  LValue BaseAddr = CGF.MakeAddrLValue(PerfoStructAddress, PerfoInfoTy);

  LValue FieldAddr = CGF.EmitLValueForField(
      BaseAddr, *std::next(PerfoInfoRecord->field_begin(), FlagsId));
  CGF.EmitStoreOfScalar(PerfoType, FieldAddr);

  FieldAddr = CGF.EmitLValueForField(
      BaseAddr, *std::next(PerfoInfoRecord->field_begin(), ApproxRegionId));
  CGF.EmitStoreOfScalar(RGId, FieldAddr);

  if (PerfoClause.getPerfoType() == approx::PT_SMALL ||
      PerfoClause.getPerfoType() == approx::PT_LARGE) {
    FieldAddr = CGF.EmitLValueForField(
        BaseAddr, *std::next(PerfoInfoRecord->field_begin(), StepId));
    CGF.EmitStoreOfScalar(StepVal, FieldAddr);
  } else {
    FieldAddr = CGF.EmitLValueForField(
        BaseAddr, *std::next(PerfoInfoRecord->field_begin(), RateId));
    CGF.EmitStoreOfScalar(StepVal, FieldAddr);
  }
  /// Cast ptr to void* and assign to respective parameter
  approxRTParams[PerfoDesc] = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      PerfoStructAddress.getPointer(), CGM.VoidPtrTy);
  hasPerfo = true;
}

void CGApproxRuntime::CGApproxRuntimeEmitIfInit(CodeGenFunction &CGF,
                                                ApproxIfClause &IfClause) {
  if (const auto *PreInit = cast_or_null<DeclStmt>(IfClause.getPreInit())) {
    for (const auto *D : PreInit->decls()) {
      CGF.EmitVarDecl(cast<VarDecl>(*D));
    }
  }
  approxRTParams[Cond] = CGF.EvaluateExprAsBool(IfClause.getCondition());
}

void CGApproxRuntime::CGApproxRuntimeEmitPerfoFn(CapturedStmt &CS) {
  if (hasPerfo) {
    CodeGenFunction::CGCapturedStmtInfo CGSI(CS);
    CodeGenFunction CGF(CGM, true);
    CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGSI);
    llvm::Function *Fn = CGF.GenerateCapturedStmtFunction(CS);
    Fn->addFnAttr("Perforate");
    approxRTParams[PerfoFn] = Fn;
  }
  return;
}

void CGApproxRuntime::CGApproxRuntimeExitRegion(CodeGenFunction &CGF) {
  /// Create Runtime function type
  llvm::FunctionType *RTFnTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                              ArrayRef<llvm::Type *>(approxRTTypes), false);
  llvm::Function *RTFn =
      Function::Create(RTFnTy, GlobalValue::ExternalLinkage,
                       "__approx_exec_call", CGM.getModule());
  CGF.EmitCallOrInvoke(RTFn, ArrayRef<llvm::Value *>(approxRTParams));
}

void CGApproxRuntime::CGApproxRuntimeRegisterInputs(ApproxInClause &InClause) {
  for (auto *V : InClause.varlist()) {
    Data.push_back(std::make_pair(V, Input));
  }
}

void CGApproxRuntime::CGApproxRuntimeRegisterOutputs(ApproxOutClause &OutClause) {
  for (auto *V : OutClause.varlist()) {
    Data.push_back(std::make_pair(V, Output));
  }
}

void CGApproxRuntime::CGApproxRuntimeRegisterInputsOutputs(ApproxInOutClause &InOutClause){
  for (auto *V : InOutClause.varlist()) {
    Data.push_back(std::make_pair(V, InputOuput));
  }
}

void CGApproxRuntime::CGApproxRuntimeEmitDataValues(CodeGenFunction &CGF){
  /// No Dependencies so exit.
  if (!Data.size())
    return;
  int numVars = Data.size();
  ASTContext &C = CGM.getContext();
  QualType VarInfoArrayTy;
  llvm::Value *NumOfElements = llvm::ConstantInt::get(CGM.Int32Ty, numVars, false);

  VarInfoArrayTy = C.getConstantArrayType(
    VarInfoTy, llvm::APInt(64,numVars), nullptr, ArrayType::Normal, 0);

  Address VarInfoArray = CGF.CreateMemTemp(VarInfoArrayTy, ".dep.approx.arr.addr");
  VarInfoArray = CGF.Builder.CreateConstArrayGEP(VarInfoArray, 0);

  const auto *VarInfoRecord = VarInfoTy->getAsRecordDecl();
  unsigned Pos = 0;
  enum VarInfoFieldID { PTR, DIR, TYPE, NUMELEMENTS, OPAQUE };
  dbgs() << "Here " << __LINE__ <<"\n";
  for (auto P : Data){
    llvm::Value *Addr;
    llvm::Value *Size;
    Expr *E  = P.first;
    Directionality Dir = P.second;
    std::tie(Addr, Size) = getPointerAndSize(CGF, E);
    LValue Base = CGF.MakeAddrLValue(CGF.Builder.CreateConstGEP(VarInfoArray, Pos), VarInfoTy);
    auto *FieldT = *std::next(VarInfoRecord->field_begin(), PTR);
    LValue BaseAddrLVal = CGF.EmitLValueForField(Base,FieldT);
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(Addr, CGF.IntPtrTy), BaseAddrLVal);
    /// NumElements now actually contains num of bytes.
    LValue LenLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), NUMELEMENTS));
    CGF.EmitStoreOfScalar(Size, LenLVal);

    Value *direction=
      llvm::ConstantInt::get(CGM.Int32Ty, Dir, false);
    LValue DirLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), DIR));
    CGF.EmitStoreOfScalar(direction, DirLVal);
    Pos++;
  }
  approxRTParams[DataDesc] = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      VarInfoArray.getPointer(), CGF.VoidPtrTy);
  approxRTParams[DataSize] = NumOfElements;
}