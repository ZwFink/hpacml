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

#include "CGApproxRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/ApproxClause.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtApprox.h"
#include "clang/Basic/Approx.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include <unordered_map>
#include <memory>

using namespace llvm;
using namespace clang;
using namespace CodeGen;



int8_t convertToApproxType(const BuiltinType *T) {
  ApproxType approxType;
  switch (T->getKind()) {
  case BuiltinType::Kind::Bool:
    approxType = BOOL;
    break;
  case BuiltinType::Kind::Char_U:
  case BuiltinType::Kind::UChar:
  case BuiltinType::Kind::WChar_U:
    approxType = UCHAR;
    break;
  case BuiltinType::Kind::UShort:
    approxType = USHORT;
    break;
  case BuiltinType::Kind::UInt:
    approxType = UINT;
    break;
  case BuiltinType::Kind::ULong:
    approxType = ULONG;
    break;
  case BuiltinType::Kind::ULongLong:
    approxType = ULONGLONG;
    break;
  case BuiltinType::Kind::Char_S:
  case BuiltinType::Kind::SChar:
  case BuiltinType::Kind::WChar_S:
    approxType = SCHAR;
    break;
  case BuiltinType::Kind::Short:
    approxType = SHORT;
    break;
  case BuiltinType::Kind::Int:
    approxType = INT;
    break;
  case BuiltinType::Kind::Long:
    approxType = LONG;
    break;
  case BuiltinType::Kind::LongLong:
    approxType = LONGLONG;
    break;
  case BuiltinType::Kind::Float:
    approxType = FLOAT;
    break;
  case BuiltinType::Kind::Double:
    approxType = DOUBLE;
    break;
  case BuiltinType::Kind::LongDouble:
    approxType = LDOUBLE;
    break;
  default:
    approxType = ApproxType::INVALID;
    break;
  }
  return approxType;
}


size_t CountAIVRExpandedShapes(CodeGenFunction *CGF, llvm::ArrayRef<Expr*> Slices) {
  size_t numExpanded = 0;
  for(auto *E : Slices) {
    assert(isa<ApproxSliceExpr>(E) && "Expected a slice expression");

    ApproxSliceExpr *Slice = dyn_cast<ApproxSliceExpr>(E);
    ApproxSliceExpr::AIVREChildKind Kind = Slice->getAIVREChildKind();
    if(Kind == ApproxSliceExpr::AIVREChildKind::BINARY_EXPR ||
    Kind == ApproxSliceExpr::AIVREChildKind::STANDALONE) {
      numExpanded++;
    }
  }
  return numExpanded;
}
static std::tuple<llvm::Value *, llvm::Value *, llvm::Value *, llvm::Value *>
getPointerAndSize(CodeGenFunction &CGF, const Expr *E) {
  // Address of first Element.
  llvm::Value *Addr;
  // Total Size in Bytes.
  llvm::Value *Size;
  // Number of elements
  llvm::Value *NumElements;
  // Data Type
  llvm::Value *TypeOfElement;
  // This is actually required only for
  // user defined types. Everything else
  // should already be known by the RT system.
  llvm::Value *SizeOfElement;

  Addr = CGF.EmitLValue(E).getPointer(CGF);

  if (const auto *ASE =
          dyn_cast<ApproxArraySectionExpr>(E->IgnoreParenImpCasts())) {
    QualType BaseTy =
        ApproxArraySectionExpr::getBaseOriginalType(ASE->getBase());
    QualType ResultExprTy = BaseTy;
    int8_t TyKind = -1;

    // Drill down to find the scalar type we are point to.
    do {
      if (auto *AT = CGF.getContext().getAsArrayType(ResultExprTy))
        ResultExprTy = AT->getElementType();
      else
        ResultExprTy = ResultExprTy->getPointeeType();
    } while (ResultExprTy->isPointerType() || ResultExprTy->isReferenceType() ||
             ResultExprTy->isArrayType());

    if (const BuiltinType *T = ResultExprTy->getAs<BuiltinType>()) {
      TyKind = convertToApproxType(T);
    }

    // The array slicer does not yet support multi-dimensional slicing
    // Example:
    // int a[100];
    // #pragma omp in(a[0:10]) ---> will work correclty.
    // int a[100][100];
    // #pragma omp in(a[0:10][:]) ---> will work correclty.
    // int a[100][100];
    // #pragma omp in(a[4:2][5:4]) ---> will NOT work correclty.
    // Solution 1:
    // Describe access  dimensions with some kind of struct
    // with an access pattern, strides etc.
    // Solution 2:
    // Create code that iterats through all the outer indexes
    // and points to the inner continues memeory array and pass
    // to the runtime system multiple in/out/inout parameters.
    LValue UpAddrLVal =
        CGF.EmitApproxArraySectionExpr(ASE, /*IsLowerBound=*/false);
    Address UpAddrAddress = UpAddrLVal.getAddress(CGF);
    llvm::Value *UpAddr =
      CGF.Builder.CreateConstGEP1_32(UpAddrAddress.getElementType(), UpAddrAddress.getPointer(), /*Idx0=*/1);

    llvm::Value *LowIntPtr = CGF.Builder.CreatePtrToInt(Addr, CGF.SizeTy);
    llvm::Value *UpIntPtr = CGF.Builder.CreatePtrToInt(UpAddr, CGF.SizeTy);
    Size = CGF.Builder.CreateNUWSub(UpIntPtr, LowIntPtr);
    SizeOfElement = CGF.getTypeSize(ResultExprTy);
    NumElements = CGF.Builder.CreateUDiv(Size, SizeOfElement);
    TypeOfElement = llvm::ConstantInt::get(CGF.Builder.getInt8Ty(), TyKind);
  } else {
    QualType Ty = E->getType();
    int TyKind = -1;
    if (const BuiltinType *T = Ty->getAs<BuiltinType>()) {
      TyKind = convertToApproxType(T);
    }

    SizeOfElement = CGF.getTypeSize(Ty);
    Size = SizeOfElement;
    QualType SizeOfType = CGF.getContext().getSizeType();
    NumElements = llvm::ConstantInt::get(CGF.ConvertType(SizeOfType), 1);
    TypeOfElement = llvm::ConstantInt::get(CGF.Builder.getInt8Ty(), TyKind);
  }
  return std::make_tuple(Addr, NumElements, SizeOfElement, TypeOfElement);
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
    addFieldToRecordDecl(C, perfoInfoRD,
                         C.getRealTypeForBitwidth(32, FloatModeKind::Float));
    perfoInfoRD->completeDefinition();
    perfoInfoTy = C.getRecordType(perfoInfoRD);
  }
  return;
}

static void getSliceInfoTy(ASTContext &C, QualType &SliceInfoTy) {
  if(SliceInfoTy.isNull()) {
    RecordDecl *sliceInfoRD = C.buildImplicitRecord("approx_slice_info_t");
    sliceInfoRD->startDefinition();
    // Start Index
    addFieldToRecordDecl(C, sliceInfoRD, C.getIntTypeForBitwidth(64, false));
    // Stop Index
    addFieldToRecordDecl(C, sliceInfoRD, C.getIntTypeForBitwidth(64, false));
    // Step
    addFieldToRecordDecl(C, sliceInfoRD, C.getIntTypeForBitwidth(64, false));

    // the mode of this slice's AIVR, if any. See ApproxSliceExpr::AIVREChildKind
    addFieldToRecordDecl(C, sliceInfoRD, C.getIntTypeForBitwidth(32, false));
    // the program representation of this slice's AIVR, if any
    addFieldToRecordDecl(C, sliceInfoRD, C.getIntTypeForBitwidth(64, true));
    sliceInfoRD->completeDefinition();
    SliceInfoTy = C.getRecordType(sliceInfoRD);
  }
}

static void getNDArraySliceTy(ASTContext &C, QualType &SliceTy, QualType &ShapeTy, QualType &NDArraySliceTy) {
  if(NDArraySliceTy.isNull()) {
    assert(!SliceTy.isNull() && "SliceTy must be defined before NDArraySliceTy");

    RecordDecl *NDArraySliceRD = C.buildImplicitRecord("approx_ndarray_slice_t");
    NDArraySliceRD->startDefinition();
    // Void pointer pointer to the arrays
    addFieldToRecordDecl(C, NDArraySliceRD, C.getPointerType(C.getIntPtrType()));

    // indicator for the underlying type of each base
    addFieldToRecordDecl(C, NDArraySliceRD, C.getPointerType(C.getIntTypeForBitwidth(8, false)));

    // Number of bases
    addFieldToRecordDecl(C, NDArraySliceRD, C.getIntTypeForBitwidth(32, false));

    // Number of dimensions
    addFieldToRecordDecl(C, NDArraySliceRD, C.getIntTypeForBitwidth(32, false));

    // The slice info
    addFieldToRecordDecl(C, NDArraySliceRD, C.getPointerType(SliceTy));

    // the shape info, just an array of integers
    addFieldToRecordDecl(C, NDArraySliceRD, C.getPointerType(ShapeTy));

    // shapes after substituting approx index var ref expressions
    addFieldToRecordDecl(C, NDArraySliceRD, C.getPointerType(ShapeTy));

    // Number of dimensions before substitution: this is set by the runtime
    addFieldToRecordDecl(C, NDArraySliceRD, C.getIntTypeForBitwidth(32, false));

    NDArraySliceRD->completeDefinition();
    NDArraySliceTy = C.getRecordType(NDArraySliceRD);
  }
}

static void getTensorShapeTy(ASTContext &C, QualType &TensorShapeTy) {
  if(TensorShapeTy.isNull()) {
    RecordDecl *TensorShapeRD = C.buildImplicitRecord("tensor_shape_t");
    TensorShapeRD->startDefinition();
    // Number of dimensions
    addFieldToRecordDecl(C, TensorShapeRD, C.getIntTypeForBitwidth(32, false));

    // The shape info, just an array of integers
    addFieldToRecordDecl(C, TensorShapeRD, C.getPointerType(C.getIntTypeForBitwidth(64, true)));

    TensorShapeRD->completeDefinition();
    TensorShapeTy = C.getRecordType(TensorShapeRD);
  }
}

static void getVarInfoType(ASTContext &C, QualType &VarInfoTy) {
  if (VarInfoTy.isNull()) {
    RecordDecl *VarInfoRD = C.buildImplicitRecord("approx_var_info_t");
    VarInfoRD->startDefinition();
    /// Void pointer pointing to data values
    addFieldToRecordDecl(C, VarInfoRD, C.getIntPtrType());
    /// Void pointer pointing to the names of the variables
    addFieldToRecordDecl(C, VarInfoRD, C.getIntPtrType());
    QualType SizeOfType = C.getSizeType();
    SizeOfType = C.getCanonicalType(SizeOfType);
    /// number of elements
    addFieldToRecordDecl(C, VarInfoRD, SizeOfType);
    /// Sizeof(type)
    addFieldToRecordDecl(C, VarInfoRD, SizeOfType);
    /// Data Type can be negative.
    /// The bitwidth will depend on the way we support
    /// user types/ primary types. Keep it 8 atm.
    addFieldToRecordDecl(C, VarInfoRD, C.getIntTypeForBitwidth(8, true));
    /// The directionality of this region in/out/inout
    addFieldToRecordDecl(C, VarInfoRD, C.getIntTypeForBitwidth(8, false));
    /// Is this info_t wrapping a tensor?
    addFieldToRecordDecl(C, VarInfoRD, C.getIntTypeForBitwidth(8, false));
    VarInfoRD->completeDefinition();
    VarInfoTy = C.getRecordType(VarInfoRD);
  }
  return;
}

static void getInternalReprMetadataTy(ASTContext &C, QualType& TensorShapeTy, QualType &InternalReprTy) {
  if(InternalReprTy.isNull()) {
    assert(!TensorShapeTy.isNull() && "TensorShapeTy must be defined before InternalReprTy");
    RecordDecl *InternalReprRD = C.buildImplicitRecord("internal_tensor_metadata_t");
    InternalReprRD->startDefinition();
    // int type
    addFieldToRecordDecl(C, InternalReprRD, C.getIntTypeForBitwidth(32, false));
    // TensorShapeTy shape
    addFieldToRecordDecl(C, InternalReprRD, TensorShapeTy);
    // void *for internal data
    addFieldToRecordDecl(C, InternalReprRD, C.getIntPtrType());
    InternalReprRD->completeDefinition();
    InternalReprTy = C.getRecordType(InternalReprRD);
  }
}

CGApproxRuntime::CGApproxRuntime(CodeGenModule &CGM)
    : CGM(CGM), CallbackFnTy(nullptr), RTFnTy(nullptr), approxRegions(0),
      StartLoc(SourceLocation()), EndLoc(SourceLocation()), requiresData(false),
      requiresInputs(false) {
  ASTContext &C = CGM.getContext();
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  llvm::PointerType *CharPtrTy =
      llvm::PointerType::getUnqual(Types.ConvertType(C.CharTy));
  getPerfoInfoType(C, PerfoInfoTy);
  getVarInfoType(C, VarInfoTy);
  getSliceInfoTy(C, SurrogateInfo.SliceInfoTy);
  getTensorShapeTy(C, SurrogateInfo.TensorShapeTy);
  getNDArraySliceTy(C, SurrogateInfo.SliceInfoTy, SurrogateInfo.TensorShapeTy, SurrogateInfo.NDArraySliceTy);
  getInternalReprMetadataTy(C, SurrogateInfo.TensorShapeTy, SurrogateInfo.InternalReprMetadataTy);

  CallbackFnTy = llvm::FunctionType::get(CGM.VoidTy, {CGM.VoidPtrTy}, false);

  // This is the runtime call function type information, which mirrors the
  // types provided in the argument parameters.
  RTFnTy = llvm::FunctionType::get(
      CGM.VoidTy,
      {/* Orig. fn ptr*/ llvm::PointerType::getUnqual(CallbackFnTy),
       /* Perfo fn ptr*/ llvm::PointerType::getUnqual(CallbackFnTy),
       /* Captured data ptr*/ CGM.VoidPtrTy,
       /* Cond Value*/ llvm::Type::getInt1Ty(CGM.getLLVMContext()),
       /* Label Name */ CharPtrTy,
       /* Perfo Description */ CGM.VoidPtrTy,
       /* Memoization Type*/ CGM.Int32Ty,
       /* Petrubation Type*/ CGM.Int32Ty,
       /* ML Type*/ CGM.Int32Ty,
       /* Input Data Descr*/ CGM.VoidPtrTy,
       /* Input Data Num Elements*/ CGM.Int32Ty,
       /* Ouput Data Descr. */ CGM.VoidPtrTy,
       /* Output Data Num Elements*/ CGM.Int32Ty},
      false);

  SurrogateInfo.ConvertSliceInfoFnTy = llvm::FunctionType::get(
    CGM.VoidTy, 
    { /*Number of slices */ CGM.Int32Ty,
      /* Tensor array slices (void **) */ CGM.VoidPtrTy,
      /* Functor Array Slices */ CGM.VoidPtrTy
    }, false);

  SurrogateInfo.ConvertToHigherOrderShapeFnTy = llvm::FunctionType::get(
    CGM.VoidTy, 
    { /*Number of slices */ CGM.Int32Ty,
      /* Tensor array slices (void **) */ CGM.VoidPtrTy,
      /* Functor Array Slices */ CGM.VoidPtrTy
    }, false);

  SurrogateInfo.SubstituteAIVRInShapesFnTy = llvm::FunctionType::get(
    CGM.VoidTy, 
    { /*Number of arguments */ CGM.Int32Ty,
      /* array_info_t ** pointing to the arrays */ CGM.VoidPtrTy,
      /* internal_repr_metadata_t * with metadata about internal representation */ CGM.VoidPtrTy
    }, false);

  SurrogateInfo.ConversionMemToTensorFnTy = llvm::FunctionType::get(
    CGM.VoidPtrTy, 
    { /*Number of LHS slices */ CGM.Int32Ty,
      /* slice_info_t * for LHS */ CGM.VoidPtrTy,
      /* tensor_shape_t for LHS  */ CGM.VoidPtrTy,
      /*Number of RHS array_info_t  */ CGM.Int32Ty,
      /* array_info_t **array info for the RHS */ CGM.VoidPtrTy
    }, false);

  SurrogateInfo.ConversionTensorToMemFnTy = llvm::FunctionType::get(
    CGM.VoidPtrTy, 
    { /*Number of LHS slices */ CGM.Int32Ty,
      /* slice_info_t * for LHS */ CGM.VoidPtrTy,
      /* tensor_shape_t for LHS  */ CGM.VoidPtrTy,
      /*Number of RHS array_info_t  */ CGM.Int32Ty,
      /* array_info_t **array info for the RHS */ CGM.VoidPtrTy
    }, false);

  SurrogateInfo.TensorCleanupFnTy = llvm::FunctionType::get(
    CGM.VoidTy, {CGM.VoidPtrTy}, false);
}

void CGApproxRuntime::CGApproxRuntimeEnterRegion(CodeGenFunction &CGF,
                                                 CapturedStmt &CS) {
  // This two values (requiresInputs, requiredData) should be false.
  // currently though the compiler is forwarding everything to
  // the runtime system
  requiresInputs = true;
  requiresData = true;

  ASTContext &C = CGM.getContext();
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  llvm::PointerType *CharPtrTy =
      llvm::PointerType::getUnqual(Types.ConvertType(C.CharTy));
  /// Reset All info of the Runtime "state machine"
  Inputs.clear();
  Outputs.clear();
  for (unsigned i = ARG_START; i < ARG_END; i++)
    approxRTParams[i] = nullptr;

  Address CapStructAddr = CGF.GenerateCapturedStmtArgument(CS);
  CodeGenFunction::CGCapturedStmtInfo CGSI(CS);
  CodeGenFunction localCGF(CGM, true);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(localCGF, &CGSI);
  llvm::Function *Fn = localCGF.GenerateCapturedStmtFunction(CS);

  /// Fill in parameters of runtime function call
  /// Put default values on everything.
  /// EmitClause* Will replace as necessary
  approxRTParams[AccurateFn] =
      CGF.Builder.CreatePointerCast(Fn, CallbackFnTy->getPointerTo());
  approxRTParams[PerfoFn] = llvm::ConstantPointerNull::get(
      llvm::PointerType::getUnqual(CallbackFnTy));
  approxRTParams[CapDataPtr] =
      CGF.Builder.CreatePointerCast(CapStructAddr.getPointer(), CGM.VoidPtrTy);
  approxRTParams[Cond] = llvm::ConstantInt::get(CGF.Builder.getInt1Ty(), true);
  approxRTParams[Label] = llvm::ConstantPointerNull::get(CharPtrTy);
  approxRTParams[PerfoDesc] = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
  approxRTParams[DataDescIn] = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
  approxRTParams[DataSizeIn] =
      llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);
  approxRTParams[DataDescOut] = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
  approxRTParams[DataSizeOut] =
      llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);
  approxRTParams[MemoDescr] =
      llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);
  approxRTParams[PetruDescr] =
      llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);
  approxRTParams[MLDescr] =
      llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);

  StartLoc = CS.getBeginLoc();
  EndLoc = CS.getEndLoc();
  return;
}

// TODO: Should we add LoopExprs to the PerfoClause?
void CGApproxRuntime::CGApproxRuntimeEmitPerfoInit(
    CodeGenFunction &CGF, CapturedStmt &CS, ApproxPerfoClause &PerfoClause, const ApproxLoopHelperExprs &LoopExprs) {
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
  /// Emit Function which needs to be perforated.
  CGApproxRuntimeEmitPerfoFn(CS, LoopExprs, PerfoClause);
}

void CGApproxRuntime::CGApproxRuntimeEmitPetrubateInit(
    CodeGenFunction &CGF, ApproxPetrubateClause &PetrubateClause) {
  requiresData = true;
  if (PetrubateClause.getPetrubateType() == approx::PETRUBATE_IN) {
    requiresInputs = true;
    approxRTParams[PetruDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 1);
  } else if (PetrubateClause.getPetrubateType() == approx::PETRUBATE_OUT) {
    approxRTParams[PetruDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 2);
  } else if (PetrubateClause.getPetrubateType() == approx::PETRUBATE_INOUT) {
    approxRTParams[PetruDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 3);
  }
}


void CGApproxRuntime::CGApproxRuntimeEmitMemoInit(
    CodeGenFunction &CGF, ApproxMemoClause &MemoClause) {
  requiresData = true;
  if (MemoClause.getMemoType() == approx::MT_IN) {
    requiresInputs = true;
    approxRTParams[MemoDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 1);
  } else if (MemoClause.getMemoType() == approx::MT_OUT) {
    approxRTParams[MemoDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 2);
  }
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

/// Creates the outlined function for a perforated loop.
llvm::Function *CodeGenFunction::GeneratePerfoCapturedStmtFunction(
    const CapturedStmt &CS, const ApproxLoopHelperExprs &LoopExprs,
    const ApproxPerfoClause &PC) {
  assert(CapturedStmtInfo &&
    "CapturedStmtInfo should be set when generating the captured function");
  const CapturedDecl *CD = CS.getCapturedDecl();
  const RecordDecl *RD = CS.getCapturedRecordDecl();
  SourceLocation Loc = CS.getBeginLoc();
  assert(CD->hasBody() && "missing CapturedDecl body");

  // Build the argument list.
  ASTContext &Ctx = CGM.getContext();
  FunctionArgList Args;
  Args.append(CD->param_begin(), CD->param_end());

  // Create the function declaration.
  const CGFunctionInfo &FuncInfo =
    CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, Args);
  llvm::FunctionType *FuncLLVMTy = CGM.getTypes().GetFunctionType(FuncInfo);

  llvm::Function *F =
    llvm::Function::Create(FuncLLVMTy, llvm::GlobalValue::InternalLinkage,
                           CapturedStmtInfo->getHelperName(), &CGM.getModule());
  CGM.SetInternalFunctionAttributes(CD, F, FuncInfo);
  if (CD->isNothrow())
    F->addFnAttr(llvm::Attribute::NoUnwind);

  // Generate the function.
  StartFunction(CD, Ctx.VoidTy, F, FuncInfo, Args, CD->getLocation(),
                CD->getBody()->getBeginLoc());
  // Set the context parameter in CapturedStmtInfo.
  Address DeclPtr = GetAddrOfLocalVar(CD->getContextParam());
  CapturedStmtInfo->setContextValue(Builder.CreateLoad(DeclPtr));

  // Initialize variable-length arrays.
  LValue Base = MakeNaturalAlignAddrLValue(CapturedStmtInfo->getContextValue(),
                                           Ctx.getTagDeclType(RD));
  for (auto *FD : RD->fields()) {
    if (FD->hasCapturedVLAType()) {
      auto *ExprArg =
          EmitLoadOfLValue(EmitLValueForField(Base, FD), CS.getBeginLoc())
              .getScalarVal();
      auto VAT = FD->getCapturedVLAType();
      VLASizeMap[VAT->getSizeExpr()] = ExprArg;
    }
  }

  // If 'this' is captured, load it into CXXThisValue.
  if (CapturedStmtInfo->isCXXThisExprCaptured()) {
    FieldDecl *FD = CapturedStmtInfo->getThisFieldDecl();
    LValue ThisLValue = EmitLValueForField(Base, FD);
    CXXThisValue = EmitLoadOfLValue(ThisLValue, Loc).getScalarVal();
  }

  PGO.assignRegionCounters(GlobalDecl(CD), F);

  // Declare IV, LastIteration, LB, UB variables.
  const auto *IVExpr = cast<DeclRefExpr>(LoopExprs.IterationVarRef);
  const auto *IVDecl = cast<VarDecl>(IVExpr->getDecl());
  EmitVarDecl(*IVDecl);

  if (const auto *LIExpr = dyn_cast<DeclRefExpr>(LoopExprs.LastIteration)) {
    EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
    // Emit calculation of the iterations count.
    EmitIgnoredExpr(LoopExprs.CalcLastIteration);
  }

  const auto *LBExpr = cast<DeclRefExpr>(LoopExprs.LB);
  const auto *LBDecl = cast<VarDecl>(LBExpr->getDecl());
  EmitVarDecl(*LBDecl);

  // Emit variable declarations of PreInits.
  if (const auto *PreInits = cast_or_null<DeclStmt>(LoopExprs.PreInits)) {
    for (const auto *I : PreInits->decls())
      EmitVarDecl(cast<VarDecl>(*I));
  }

  const auto *UBExpr = cast<DeclRefExpr>(LoopExprs.UB);
  const auto *UBDecl = cast<VarDecl>(UBExpr->getDecl());
  EmitVarDecl(*UBDecl);

  // EmitIgnoredExpr(LoopExprs.EUB);
  // IV = LB;
  EmitIgnoredExpr(LoopExprs.Init);

  const auto *CounterExpr = cast<DeclRefExpr>(LoopExprs.Counter);
  const auto *CounterDecl = cast<VarDecl>(CounterExpr->getDecl());
  // Emit counter declaration and init if it is not captured.
  if (!CS.capturesVariable(CounterDecl)) {
    EmitVarDecl(*CounterDecl);
    EmitIgnoredExpr(LoopExprs.CounterInit);
  }

  if(LoopExprs.OMPParallelForDir) {
    EmitStmt(LoopExprs.OMPParallelForDir);
  }
  else {
    // Create BBs for end of the loop and condition check.
    auto LoopExit = getJumpDestInCurrentScope("approx.perfo.for.end");
    auto CondBlock = createBasicBlock("approx.perfo.for.cond");
    EmitBlock(CondBlock);
    const SourceRange R = CS.getSourceRange();

    LoopStack.push(CondBlock, SourceLocToDebugLoc(R.getBegin()),
                   SourceLocToDebugLoc(R.getEnd()));

    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    llvm::BasicBlock *LoopBody = createBasicBlock("approx.perfo.for.body");
    llvm::BasicBlock *PerfRandCondBlock =
        createBasicBlock("approx.perfo.for.rand.cond");
    auto *LoopCond = LoopExprs.Cond;
    // Emit condition.
    EmitBranchOnBoolExpr(LoopCond, PerfRandCondBlock, ExitBlock,
                         getProfileCount(&CS));
    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(LoopExit);
    }

    // Create a block for the increment.
    JumpDest Continue = getJumpDestInCurrentScope("approx.perfo.for.inc");
    BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

    EmitBlock(PerfRandCondBlock);
    // Emit perfo rand cond basic block.
    #if 0
    if (PC.getPerfoType() == approx::PT_RAND) {
      // Skip iteration if true.
      StringRef FnName("__approx_skip_iteration");
      llvm::Function *Fn = CGM.getModule().getFunction(FnName);
      llvm::FunctionType *FnTy =
          llvm::FunctionType::get(llvm::Type::getInt1Ty(CGM.getLLVMContext()),
                                  {CGM.Int32Ty, CGM.FloatTy},
                                  /* VarArgs */ false);
      if (!Fn) {
        Fn = Function::Create(FnTy, GlobalValue::ExternalLinkage, FnName,
                              CGM.getModule());
      }

      llvm::Value *IV = EmitLoadOfScalar(EmitLValue(LoopExprs.IterationVarRef),
                                         SourceLocation());
      llvm::Value *Pr = nullptr;

      // Emit Pr expression, either loading from a captured DRE or evaluating
      // it.
      if (dyn_cast<DeclRefExpr>(LoopExprs.PerfoStep)) {
        Pr =
            EmitLoadOfScalar(EmitLValue(LoopExprs.PerfoStep), SourceLocation());
      } else
        Pr = EmitScalarExpr(LoopExprs.PerfoStep);

      assert(Pr != nullptr && "Expected a non-null Pr value");

      llvm::FunctionCallee FnCallee({FnTy, Fn});
      llvm::Value *Ret = EmitRuntimeCall(FnCallee, {IV, Pr});
      Builder.CreateCondBr(Ret, Continue.getBlock(), LoopBody);
    } else {
      EmitBranch(LoopBody);
    }
    #endif

    EmitBlock(LoopBody);
    incrementProfileCounter(&CS);

    // Emit counter update.
    EmitIgnoredExpr(LoopExprs.CounterUpdate);

    auto emitBody = [&](auto &&emitBody, const Stmt *S,
                        const Stmt *LoopS) -> void {
      const Stmt *SimplifiedS = S->IgnoreContainers();
      if (const auto *CompS = dyn_cast<CompoundStmt>(SimplifiedS)) {
        // Keep track of the current cleanup stack depth, including debug
        // scopes.
        // CodeGenFunction::LexicalScope Scope(CGF, S->getSourceRange());
        for (const Stmt *CurStmt : CompS->body())
          emitBody(emitBody, CurStmt, LoopS);
        return;
      }

      // Emit only the body of the loop statement.
      if (S == LoopS) {
        if (const auto *For = dyn_cast<ForStmt>(S)) {
          S = For->getBody();
        } else {
          assert(isa<CXXForRangeStmt>(S) &&
                 "Expected canonical for loop or range-based for loop.");
          const auto *CXXFor = cast<CXXForRangeStmt>(S);
          EmitStmt(CXXFor->getLoopVarStmt());
          S = CXXFor->getBody();
        }
      }

      EmitStmt(S);
    };

    if (LoopExprs.PerfoSkip)
      EmitStmt(LoopExprs.PerfoSkip);
    Stmt *S = const_cast<Stmt *>(CS.getCapturedStmt());
    Stmt *LoopS = nullptr;
    OMPParallelForDirective *OMPFD = nullptr;
    if ((OMPFD = dyn_cast<OMPParallelForDirective>(S)))
      LoopS = OMPFD->getAssociatedStmt()->IgnoreContainers(true);
    else
      LoopS = S->IgnoreContainers();
    emitBody(emitBody, S, LoopS);

    // Emit "IV = IV + 1" and a back-edge to the condition block.
    EmitBlock(Continue.getBlock());
    if (LoopExprs.PerfoInc)
      EmitIgnoredExpr(LoopExprs.PerfoInc);
    auto *IncExpr = LoopExprs.Inc;
    EmitIgnoredExpr(IncExpr);
    BreakContinueStack.pop_back();
    EmitBranch(CondBlock);

    LoopStack.pop();
    // Emit the fall-through block.
    EmitBlock(LoopExit.getBlock());
  }

  FinishFunction(CD->getBodyRBrace());

  return F;
}

void CGApproxRuntime::CGApproxRuntimeEmitPerfoFn(
    CapturedStmt &CS, const ApproxLoopHelperExprs &LoopExprs,
    const ApproxPerfoClause &PC) {
  CodeGenFunction::CGCapturedStmtInfo CGSI(CS);
  CodeGenFunction CGF(CGM, true);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGSI);

  llvm::Function *Fn = CGF.GeneratePerfoCapturedStmtFunction(CS, LoopExprs, PC);
  approxRTParams[PerfoFn] =
      CGF.Builder.CreatePointerCast(Fn, CallbackFnTy->getPointerTo());
  return;
}

void CGApproxRuntime::CGApproxRuntimeExitRegion(CodeGenFunction &CGF) {
  Function *RTFn = nullptr;
  StringRef RTFnName("__approx_exec_call");
  RTFn = CGM.getModule().getFunction(RTFnName);

  assert(RTFnTy != nullptr);
  if (!RTFn)
    RTFn = Function::Create(RTFnTy, GlobalValue::ExternalLinkage, RTFnName,
                            CGM.getModule());

  llvm::FunctionCallee RTFnCallee({RTFnTy, RTFn});
  CGF.EmitRuntimeCall(RTFnCallee, ArrayRef<llvm::Value *>(approxRTParams));
}

void CGApproxRuntime::CGApproxRuntimeRegisterInputs(ApproxInClause &InClause) {
  for (auto *V : InClause.varlist()) {
    Inputs.push_back(std::make_pair(V, Input));
  }
}

void CGApproxRuntime::CGApproxRuntimeRegisterOutputs(
    ApproxOutClause &OutClause) {
  for (auto *V : OutClause.varlist()) {
    Outputs.push_back(std::make_pair(V, Output));
  }
}

void CGApproxRuntime::CGApproxRuntimeRegisterInputsOutputs(
    ApproxInOutClause &InOutClause) {
  for (auto *V : InOutClause.varlist()) {
    Inputs.push_back(std::make_pair(V, InputOuput));
    Outputs.push_back(std::make_pair(V, InputOuput));
  }
}

llvm::Constant* CGApproxRuntime::getOrCreateName(StringRef Name, CodeGenFunction& CGF){
  llvm::Constant *&NameStr = NameToConstant[Name]; 
  if ( !NameStr ){
    Constant *Init = ConstantDataArray::getString(CGM.getLLVMContext(), Name, false);
    NameStr = CGF.Builder.CreateGlobalString(Name);
  }
  return NameStr;
}

bool isApproxDecl(Expr *E) {
  if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (DRE->getDecl()->hasAttr<ApproxTensorDeclAttr>()) {
      return true;
    }
  }
  return false;
}

std::pair<llvm::Value *, llvm::Value *>
CGApproxRuntime::CGApproxRuntimeEmitData(
    CodeGenFunction &CGF,
    llvm::SmallVector<std::pair<Expr *, Directionality>, 16> &Data,
    const char *arrayName) {
  int numVars = Data.size();
  ASTContext &C = CGM.getContext();
  QualType VarInfoArrayTy;
  llvm::Value *NumOfElements =
      llvm::ConstantInt::get(CGM.Int32Ty, numVars, false);

  VarInfoArrayTy = C.getConstantArrayType(VarInfoTy, llvm::APInt(64, numVars),
                                          nullptr, ArrayType::Normal, 0);

  Address VarInfoArray = CGF.CreateMemTemp(VarInfoArrayTy, arrayName);
  VarInfoArray = CGF.Builder.CreateConstArrayGEP(VarInfoArray, 0);

  const auto *VarInfoRecord = VarInfoTy->getAsRecordDecl();
  unsigned Pos = 0;
  enum VarInfoFieldID { PTR, VAR_NAME, NUM_ELEM, SZ_ELEM, DATA_TYPE, DIR, IS_TENSOR};

  for (auto P : Data) {
    llvm::Value *Addr;
    llvm::Value *NumElements;
    llvm::Value *TypeOfElement;
    llvm::Value *SizeOfElement;
    Expr *E = P.first;
    Directionality Dir = P.second;


    if(isApproxDecl(E)) {
        Address addr = CGF.GetAddressOfTensor(E);
        LValue Base = CGF.MakeAddrLValue(
            CGF.Builder.CreateConstGEP(VarInfoArray, Pos), VarInfoTy);
        auto *FieldT = *std::next(VarInfoRecord->field_begin(), PTR);
        LValue BaseAddrLVal = CGF.EmitLValueForField(Base, FieldT);
        CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(addr.getPointer(), CGF.IntPtrTy),
                              BaseAddrLVal);

        LValue typeLVal = CGF.EmitLValueForField(
            Base, *std::next(VarInfoRecord->field_begin(), IS_TENSOR));
        CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int8Ty, 1, false),
                              typeLVal);
        continue;
    } else if(ApproxCompoundExpr *ACE = dyn_cast<ApproxCompoundExpr>(E)) {
      assert(ACE->getExpressions().size() == 1 && "Expected only one expression");

      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ACE->getExpressions()[0]);
      ApproxDeclareTensorDecl *ADTD = dyn_cast<ApproxDeclareTensorDecl>(DRE->getDecl());
      ADTD->setDirectionality(ApproxDeclareTensorDecl::Direction::TENSOR_TO_MEM);

      auto LVals = CGF.EmitApproxCompoundExpr(*cast<ApproxCompoundExpr>(E));
      assert(LVals.size() == 1 && "Expected only one LValue");

      Address addr = LVals[0].getAddress(CGF);
      LValue Base = CGF.MakeAddrLValue(
          CGF.Builder.CreateConstGEP(VarInfoArray, Pos), VarInfoTy);
      auto *FieldT = *std::next(VarInfoRecord->field_begin(), PTR);
      LValue BaseAddrLVal = CGF.EmitLValueForField(Base, FieldT);
      CGF.EmitStoreOfScalar(
          CGF.Builder.CreatePtrToInt(addr.getPointer(), CGF.IntPtrTy),
          BaseAddrLVal);

      LValue typeLVal = CGF.EmitLValueForField(
          Base, *std::next(VarInfoRecord->field_begin(), IS_TENSOR));
      CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int8Ty, 1, false),
                            typeLVal);
      continue;
    }
    
    std::tie(Addr, NumElements, SizeOfElement, TypeOfElement) =
        getPointerAndSize(CGF, E);
    // Store Addr
    LValue Base = CGF.MakeAddrLValue(
        CGF.Builder.CreateConstGEP(VarInfoArray, Pos), VarInfoTy);
    auto *FieldT = *std::next(VarInfoRecord->field_begin(), PTR);
    LValue BaseAddrLVal = CGF.EmitLValueForField(Base, FieldT);
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(Addr, CGF.IntPtrTy),
                          BaseAddrLVal);
    Base = CGF.MakeAddrLValue(
        CGF.Builder.CreateConstGEP(VarInfoArray, Pos), VarInfoTy);

    // Store VAR_NAME
    std::string ExprName = "";
    PrintingPolicy Policy(C.getLangOpts());
    llvm::raw_string_ostream OS(ExprName);
    E->printPretty(OS,nullptr,Policy);
    OS.flush();
    Value *nameAddr =  CGF.Builder.CreateGlobalStringPtr(ExprName.c_str());
    LValue nameLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), VAR_NAME));
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(nameAddr, CGF.IntPtrTy),
                          nameLVal);

    // Store NUM_ELEMENTS
    LValue nElemLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), NUM_ELEM));
    CGF.EmitStoreOfScalar(NumElements, nElemLVal);

    // Store SZ_ELEM
    LValue sElemLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), SZ_ELEM));
    CGF.EmitStoreOfScalar(SizeOfElement, sElemLVal);

    // Store DATA_TYPE
    LValue typeLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), DATA_TYPE));
    CGF.EmitStoreOfScalar(TypeOfElement, typeLVal);

    typeLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), IS_TENSOR));
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int8Ty, 0, false), typeLVal);

    // Store Dir
    Value *direction = llvm::ConstantInt::get(CGM.Int8Ty, Dir, false);
    LValue DirLVal = CGF.EmitLValueForField(
        Base, *std::next(VarInfoRecord->field_begin(), DIR));
    CGF.EmitStoreOfScalar(direction, DirLVal);
    Pos++;
  }
  return std::make_pair(NumOfElements,
                        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                            VarInfoArray.getPointer(), CGF.VoidPtrTy));
}

std::tuple<llvm::Value *, llvm::Value *, llvm::Value*>
getSliceExprValues(CodeGenFunction &CGF, Expr *Slice) {
  ApproxSliceExpr *SliceExpr = dyn_cast_or_null<ApproxSliceExpr>(Slice);
  assert(SliceExpr && "Expected a slice expression");

  Expr *Start = SliceExpr->getStart();
  Expr *Stop = SliceExpr->getStop();
  Expr *Step = SliceExpr->getStep();

  llvm::Value *StartVal = nullptr;
  llvm::Value *StopVal = nullptr; 
  llvm::Value *StepVal = nullptr;

  if (Start)
    StartVal = CGF.EmitScalarExpr(Start);
  if (Stop) {
    StopVal = CGF.EmitScalarExpr(Stop);
  }
  if (Step)
    StepVal = CGF.EmitScalarExpr(Step);

  return std::make_tuple(StartVal, StopVal, StepVal);
}


static int8_t getArrPointeeApproxType(CodeGenFunction &CGF, QualType BaseTy) {
    QualType ResultExprTy = BaseTy;
    int8_t TyKind = -1;
    // Drill down to find the scalar type we point to.
    do {
      if (auto *AT = CGF.getContext().getAsArrayType(ResultExprTy))
        ResultExprTy = AT->getElementType();
      else
        ResultExprTy = ResultExprTy->getPointeeType();
    } while (ResultExprTy->isPointerType() || ResultExprTy->isReferenceType() ||
             ResultExprTy->isArrayType());

    if (const BuiltinType *T = ResultExprTy->getAs<BuiltinType>()) {
      TyKind = convertToApproxType(T);
    }

  return TyKind;
}

Address CGApproxRuntime::CGApproxRuntimeEmitApproxArrayInfo(CodeGenFunction &CGF,
Expr *AAIE) {
  static int numArraysCreated = 0;
  ASTContext &C = CGM.getContext();
  Twine name = Twine("array.info_") + Twine(numArraysCreated);

  ApproxArraySliceExpr *E = dyn_cast_or_null<ApproxArraySliceExpr>(AAIE);
  Address ArrayInfo = CGF.CreateMemTemp(SurrogateInfo.NDArraySliceTy, name);
  const auto *ArrayInfoRecord = SurrogateInfo.NDArraySliceTy->getAsRecordDecl();


  LValue ArrayInfoStart = CGF.MakeAddrLValue(ArrayInfo, SurrogateInfo.NDArraySliceTy);

  auto Bases = E->getIndirections();
  Expr *ArrayBase = nullptr;

  QualType VoidPtrArrayTy =
      C.getConstantArrayType(C.getIntPtrType(), llvm::APInt(64, Bases.size()),
                             nullptr, ArrayType::Normal, 0);
  QualType Int8PtrArrayTy = C.getConstantArrayType(
      C.getIntTypeForBitwidth(8, false), llvm::APInt(64, Bases.size()), nullptr,
      ArrayType::Normal, 0);

  Address BaseArray = CGF.CreateMemTemp(VoidPtrArrayTy, "base.array");

  Address TypeArray = CGF.CreateMemTemp(Int8PtrArrayTy, "type.array");

  LValue FieldAddr;

  for(int i = 0; i < Bases.size(); i++) {
    ArrayBase = Bases[i];
    QualType BaseTy;

    llvm::Value *BasePtr = nullptr;
    auto BaseDestAddr = CGF.Builder.CreateConstArrayGEP(BaseArray, i);
    auto TypeDestAddr = CGF.Builder.CreateConstArrayGEP(TypeArray, i);


    // Base may be a null pointer if we are emitting array info for a functor decl
    if(ArrayBase) {
      BaseTy = E->getBaseOriginalType(ArrayBase);
      BasePtr = CGF.Builder.CreatePtrToInt(CGF.EmitScalarExpr(ArrayBase), CGF.IntPtrTy);
    } else {
      BaseTy = C.getIntTypeForBitwidth(32, false);
      BasePtr = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
    }

    LValue BaseDestLV = CGF.MakeAddrLValue(BaseDestAddr, CGF.getContext().getIntPtrType());
    CGF.EmitStoreOfScalar(BasePtr, BaseDestLV);
  
    int8_t TyKind = -1;
    if(ArrayBase) {
      TyKind = getArrPointeeApproxType(CGF, BaseTy);
    }

    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int8Ty, TyKind, false),
                          TypeDestAddr, false, C.getPointerType(C.getIntTypeForBitwidth(8, false)));

    }

    FieldAddr = CGF.EmitLValueForField(
        ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 0));
    CGF.EmitStoreOfScalar(BaseArray.getPointer(), FieldAddr);

    FieldAddr = CGF.EmitLValueForField(
        ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 1));
    CGF.EmitStoreOfScalar(TypeArray.getPointer(),
                          FieldAddr);
                        
    FieldAddr = CGF.EmitLValueForField(
      ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 2));
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int32Ty, Bases.size(), false),
                          FieldAddr);

    FieldAddr = CGF.EmitLValueForField(
        ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 3));

    auto ArraySlices = E->getSlices();
    auto NumDims = ArraySlices.size();
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int32Ty, NumDims, false),
                          FieldAddr);

    auto ExtraDims = CountAIVRExpandedShapes(&CGF, ArraySlices);
    Address SlicesStruct = CGApproxRuntimeEmitSlices(CGF, ArraySlices, ExtraDims);
    FieldAddr = CGF.EmitLValueForField(
        ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 4));
    CGF.EmitStoreOfScalar(
        CGF.Builder.CreatePtrToInt(SlicesStruct.getPointer(), CGF.IntPtrTy),
        FieldAddr);

    Address ShapesStruct =
        CGApproxRuntimeEmitShapeWithAIVRExpansion(CGF, ArraySlices);
    FieldAddr = CGF.EmitLValueForField(
        ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 5));
    CGF.EmitStoreOfScalar(
        CGF.Builder.CreatePtrToInt(ShapesStruct.getPointer(), CGF.IntPtrTy),
        FieldAddr);

    // we have a second shape struct that we store shapes with AIVR substituted
    // in
    Address ShapesStructSecondary =
        CGApproxRuntimeEmitShapeWithAIVRExpansion(CGF, ArraySlices);
    FieldAddr = CGF.EmitLValueForField(
        ArrayInfoStart, *std::next(ArrayInfoRecord->field_begin(), 6));
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(
                              ShapesStructSecondary.getPointer(), CGF.IntPtrTy),
                          FieldAddr);

    numArraysCreated++;

    return ArrayInfo;
}

void CGApproxRuntime::CGApproxRuntimeEmitSymbolicVarInits(CodeGenFunction &CGF) {
  auto VarMap = SurrogateInfo.SymbolVars;

  for(auto &SymbolInfo : VarMap) {
    auto Name = SymbolInfo.first;
    auto &Info = SymbolInfo.second;

    Value *LBVal = nullptr;

    assert(Info.Addr.has_value() && "Symbol should already be declared");
    if(Info.isFromRHS) {
      assert(Info.Range && "Symbol from RHS doesn't have a corresponding range?");
      ApproxSliceExpr *Slice = dyn_cast_or_null<ApproxSliceExpr>(Info.Range);
      assert(Slice && "Expected a slice expression");
      Expr *LB = Slice->getStart();
      LBVal = CGF.EmitScalarExpr(LB);
    }
    else {
      assert(!Info.Range && "Symbol from LHS shouldn't have a range?");
      int Repr = Info.Symbol->getShapeRepresentation();
      LBVal = llvm::ConstantInt::get(CGF.Int32Ty, Repr, true);
    }

    CGF.EmitStoreOfScalar(LBVal, Info.Addr.value(), false, CGF.getContext().getIntTypeForBitwidth(32, true));
  }

}

// void CGApproxRuntime::CGApproxRuntimeEmitSymbolicVarRange(CodeGenFunction &CGF) {

// }

// void CGApproxRuntine::CGApproxRuntimeEmitShapeForRange(CodeGenFunction &CGF) {

// }

// void CGApproxRuntime::CGApproxRuntimeEmitRHSTensorShapes(CodeGenFunction &CGF) {

// }


Address CGApproxRuntime::CGApproxRuntimeEmitSlices(CodeGenFunction &CGF,
                                                llvm::ArrayRef<Expr*> Slices,
                                                size_t ExtraDims) {
  static int numSliceArraysCreated = 0;
  ASTContext &C = CGM.getContext();
  auto numSlices = Slices.size();
  QualType SliceInfoArrayTy;

  SliceInfoArrayTy = C.getConstantArrayType(SurrogateInfo.SliceInfoTy, llvm::APInt(64, numSlices+ExtraDims),
                                          nullptr, ArrayType::Normal, 0);
  Twine ArrayName = Twine("slice.info_") + Twine(numSliceArraysCreated);
  Address SliceInfoArray = CGF.CreateMemTemp(SliceInfoArrayTy, ArrayName);
                
  for (size_t i = 0; i < numSlices; i++) {
    Address CurrentSlice = CGF.Builder.CreateConstArrayGEP(SliceInfoArray, i);
    CGApproxRuntimeEmitSlice(CGF, Slices[i], CurrentSlice);
  }

  numSliceArraysCreated++;

  return SliceInfoArray;
}

Address CGApproxRuntime::CGApproxRuntimeAllocateShape(CodeGenFunction &CGF, int ndim) {
  ASTContext &C = CGM.getContext();
  auto numSlices = ndim;
  QualType SliceTy;
  QualType Int64Ty = CGF.getContext().getIntTypeForBitwidth(64, true);
  QualType Int32Ty = CGF.getContext().getIntTypeForBitwidth(64, true);

  SliceTy = C.getConstantArrayType(Int64Ty, llvm::APInt(32, numSlices),
                                          nullptr, ArrayType::Normal, 0);
  Twine ArrayName = "slice.shape_";
  Address SliceShapeArray = CGF.CreateMemTemp(SliceTy, ArrayName);

  auto TensorShape = CGF.CreateMemTemp(SurrogateInfo.TensorShapeTy, "tensor.shape");
  auto TensorShapeAddr = CGF.MakeAddrLValue(TensorShape, SurrogateInfo.TensorShapeTy);
  auto TensorShapeDecl = SurrogateInfo.TensorShapeTy->getAsRecordDecl();

  LValue NumDimsLValue = CGF.EmitLValueForField(TensorShapeAddr, *TensorShapeDecl->field_begin());
  CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int32Ty, numSlices, false), NumDimsLValue);

  NumDimsLValue = CGF.EmitLValueForField(TensorShapeAddr, *std::next(TensorShapeDecl->field_begin(), 1));
  CGF.EmitStoreOfScalar(SliceShapeArray.getPointer(), NumDimsLValue);

  return TensorShape;
}


Address CGApproxRuntime::CGApproxRuntimeEmitShapeWithAIVRExpansion(CodeGenFunction &CGF,
                                                      llvm::ArrayRef<Expr*> Slices) {
  auto numSlices = Slices.size();

  // later, we will need to 'unpack' slices that look like
  // [i] to [i,1] and slices that look like [i*3:i*3+3] to [i,3]
  // we'll allocate the extra space we need. Note: we'll need an extra dimension
  // for /each/ slice with an AIVR, e.g., we need 2 extra slots for [i,j]
  auto allocNumSlices = numSlices + CountAIVRExpandedShapes(&CGF, Slices);

  Address AllocatedShapeStruct = CGApproxRuntimeAllocateShape(CGF, allocNumSlices);
  return CGApproxRuntimeEmitShape(CGF, AllocatedShapeStruct, Slices);
}

Address CGApproxRuntime::CGApproxRuntimeEmitShape(CodeGenFunction &CGF,
llvm::ArrayRef<Expr*> Slices) {
  auto numSlices = Slices.size();
  Address AllocatedShapeStruct = CGApproxRuntimeAllocateShape(CGF, numSlices);
  return CGApproxRuntimeEmitShape(CGF, AllocatedShapeStruct, Slices);
}

Address CGApproxRuntime::CGApproxRuntimeEmitShape(CodeGenFunction& CGF, Address Dest,
llvm::ArrayRef<Expr*> Slices) {

  ASTContext &C = CGM.getContext();
  QualType Int32Ty = C.getIntTypeForBitwidth(32, true);
  QualType Int64Ty = C.getIntTypeForBitwidth(64, true);
  auto numSlices = Slices.size();
  auto SliceNDimsAddr = CGF.Builder.CreateStructGEP(Dest, 0);
  auto SliceBaseAddr = CGF.Builder.CreateStructGEP(Dest, 1);

  CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int32Ty, numSlices, true), SliceNDimsAddr, false, Int32Ty);

  llvm::Value *SliceShapeArray = CGF.Builder.CreateLoad(SliceBaseAddr, false);
  
  Address SliceBase = Address(SliceShapeArray, CGF.Int64Ty, clang::CharUnits::fromQuantity(64));
            
  for (size_t i = 0; i < numSlices; i++) {
    Address CurrentSlice = CGF.Builder.CreateConstGEP(SliceBase, i);
    CGApproxRuntimeEmitSliceSize(CGF, Slices[i], CurrentSlice);
  }

  return Dest;
}

void CGApproxRuntime::CGApproxRuntimeEmitSliceSize(CodeGenFunction& CGF, llvm::Value *Start, llvm::Value *Stop, llvm::Value *Step, Address Dest) {
  // auto DestBB = CGF.createBasicBlock("slice.size.dest");
  // auto IncBB = CGF.createBasicBlock("slice.size.inc");
  QualType Int64Ty = CGF.getContext().getIntTypeForBitwidth(64, true);
  ASTContext &C = CGM.getContext();
  llvm::Value *StartValue = Start, *StopValue = Stop, *StepValue = Step;
  llvm::Value *Size = nullptr;

  // start - stop / step
  llvm::Value *StopMinusStart = CGF.Builder.CreateNSWSub(StopValue, StartValue);
  llvm::Value *StartMinusStopDivStep = CGF.Builder.CreateSDiv(StopMinusStart, StepValue);

  Address SizeAddr = CGF.CreateMemTemp(Int64Ty, "slice.size");
  CGF.EmitStoreOfScalar(StopMinusStart, SizeAddr, false, Int64Ty);

  // CGF.EmitStoreOfScalar(StartMinusStopDivStep, SizeAddr, false, Int64Ty);

  // if(StartMinusStopDivStep % Step == 0); ++Size;
  // llvm::Value *StartMinusStopDivStepModStep = CGF.Builder.CreateSRem(StartMinusStopDivStep, StepValue);
  // llvm::Value *IsModZero = CGF.Builder.CreateICmpEQ(StartMinusStopDivStepModStep, llvm::ConstantInt::get(CGF.Builder.getInt64Ty(), 0));
  // CGF.Builder.CreateCondBr(IsModZero, DestBB, IncBB);
  // CGF.EmitBlock(IncBB);
  // llvm::Value *StartMinusStopDivStepPlusOne = CGF.Builder.CreateAdd(StartMinusStopDivStep, llvm::ConstantInt::get(CGF.Builder.getInt64Ty(), 1));
  // CGF.EmitStoreOfScalar(StartMinusStopDivStepPlusOne, SizeAddr, false, Int64Ty);
  // CGF.Builder.CreateBr(DestBB);


  // CGF.EmitBlock(DestBB);
  Size = CGF.Builder.CreateLoad(SizeAddr);

  CGF.EmitStoreOfScalar(Size, Dest, false, Int64Ty);
}
void CGApproxRuntime::CGApproxRuntimeEmitSliceSize(CodeGenFunction &CGF, Expr *Slice, Address Dest) {
  ApproxSliceExpr *SliceExpr = dyn_cast_or_null<ApproxSliceExpr>(Slice);
  assert(SliceExpr && "Expected a slice expression");

  llvm::Value *StartValue, *StopValue, *StepValue;
  llvm::Value *Size = nullptr;

  std::tie(StartValue, StopValue, StepValue) = getSliceExprValues(CGF, SliceExpr);
  CGApproxRuntimeEmitSliceSize(CGF, StartValue, StopValue, StepValue, Dest);
}

void CGApproxRuntime::CGApproxRuntimeEmitSlice(CodeGenFunction &CGF, Expr *Slice, Address SliceMemory) {
  ApproxSliceExpr *SliceExpr = dyn_cast_or_null<ApproxSliceExpr>(Slice);
  assert(SliceExpr && "Expected a slice expression");

  ASTContext &C = CGM.getContext();
  llvm::Value *StartVal, *StopVal, *StepVal;
  std::tie(StartVal, StopVal, StepVal) = getSliceExprValues(CGF, SliceExpr);

  // Twine InfoInstanceName = Twine("slice.info_") + Twine(numSlices);
  // Address SliceMemory  = CGF.CreateMemTemp(SurrogateInfo.SliceInfoTy, InfoInstanceName);
  const auto *SliceInfoRecord = SurrogateInfo.SliceInfoTy->getAsRecordDecl();
  LValue SliceStart = CGF.MakeAddrLValue(SliceMemory, SurrogateInfo.SliceInfoTy);

  LValue FieldAddr = CGF.EmitLValueForField(
      SliceStart, *std::next(SliceInfoRecord->field_begin(), 0));
    CGF.EmitStoreOfScalar(StartVal, FieldAddr);
  

  FieldAddr = CGF.EmitLValueForField(
      SliceStart, *std::next(SliceInfoRecord->field_begin(), 1));
    CGF.EmitStoreOfScalar(StopVal, FieldAddr);

  FieldAddr = CGF.EmitLValueForField(
      SliceStart, *std::next(SliceInfoRecord->field_begin(), 2));
    CGF.EmitStoreOfScalar(StepVal, FieldAddr);

  FieldAddr = CGF.EmitLValueForField(
      SliceStart, *std::next(SliceInfoRecord->field_begin(), 3));
  int AIVREChildKind = static_cast<int>(SliceExpr->getAIVREChildKind());
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), AIVREChildKind), FieldAddr);

  // this next part is a little non-standard. Our slices may contain symbolic variables, and how and whether 
  // it contains symbolic variables affects the shape representation of the slice. We need to store information
  // that may affect the shape representation of the slice in the slice info struct. 
  auto SymbolVars = ApproxDeclareTensorFunctorDecl::getDeclaredSymbolicVarsFromExpression(SliceExpr->getStart());
  assert(SymbolVars.size() <= 1 && "Expected at most one symbolic variable in a slice");
  int SymbolVarRepr = 0;

  if(SymbolVars.size() == 1) {
    ApproxIndexVarRefExpr *IndexVarRef = dyn_cast_or_null<ApproxIndexVarRefExpr>(SymbolVars[0]);
    assert(IndexVarRef && "Expected an index variable reference");
    SymbolVarRepr = IndexVarRef->getShapeRepresentation();
  }

  FieldAddr = CGF.EmitLValueForField(
      SliceStart, *std::next(SliceInfoRecord->field_begin(), 4));
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.Builder.getInt64Ty(), SymbolVarRepr), FieldAddr);
  

}


void CGApproxRuntime::CGApproxRuntimeEmitDataValues(CodeGenFunction &CGF) {
  /// No Dependencies so exit.
  static int input_arrays = 0;
  static int output_arrays = 0;
  char name[100];
  if (!requiresData)
    return;

  llvm::Value *NumOfElements, *ArrayAddress;
  if (requiresInputs && Inputs.size() > 0) {
    sprintf(name, ".dep.approx_inputs.arr.addr_%d", input_arrays++);
    std::tie(NumOfElements, ArrayAddress) =
        CGApproxRuntimeEmitData(CGF, Inputs, name);
    approxRTParams[DataDescIn] = ArrayAddress;
    approxRTParams[DataSizeIn] = NumOfElements;
  }

  // All approximation techniques require the output
  sprintf(name, ".dep.approx_outputs.arr.addr_%d", output_arrays++);
  std::tie(NumOfElements, ArrayAddress) =
      CGApproxRuntimeEmitData(CGF, Outputs, name);
  approxRTParams[DataDescOut] = ArrayAddress;
  approxRTParams[DataSizeOut] = NumOfElements;
}

void CGApproxRuntime::CGApproxRuntimeEmitLabelInit(
    CodeGenFunction &CGF, ApproxLabelClause &LabelClause) {
  ASTContext &C = CGM.getContext();
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  llvm::PointerType *CharPtrTy =
      llvm::PointerType::getUnqual(Types.ConvertType(C.CharTy));

  LValue label;
  llvm::Value *Addr;
  if (StringLiteral *LiteralExpr = dyn_cast_or_null<StringLiteral>(LabelClause.getLabel())) {
      label =
          CGF.EmitStringLiteralLValue(cast<StringLiteral>(LabelClause.getLabel()));
    Addr = label.getPointer(CGF);
  }else{
    Addr = CGF.EmitLValue(LabelClause.getLabel()).getPointer(CGF);
  }
  Addr = CGF.Builder.CreatePointerCast(Addr, CharPtrTy);
  approxRTParams[Label] = Addr;
}

void CGApproxRuntime::CGApproxRuntimeEmitMLInit(
    CodeGenFunction &CGF, ApproxMLClause &MLClause) {
  requiresData = true;
  requiresInputs = true;
  if (MLClause.getMLType() == approx::ML_ONLINETRAIN) {
    approxRTParams[MLDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 1);
  } else if (MLClause.getMLType() == approx::ML_OFFLINETRAIN) {
    approxRTParams[MLDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 2);
  } else if (MLClause.getMLType() == approx::ML_INFER) {
    approxRTParams[MLDescr] =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 3);
  }
}

void CGApproxRuntime::CGApproxRuntimeEmitDeclInit(
    CodeGenFunction &CGF, ApproxDeclClause &DeclClause) {
      // raise an error that says we hav enot implemented this yet
    }

void CGApproxRuntime::mapSymbolicVarsToRanges(
        SymbolVarInfoMap &InfoMap, llvm::ArrayRef<Expr *> FunctorSlice,
        llvm::ArrayRef<Expr *> TensorSlice) {

  size_t idx = 0;
  for (auto *FS : FunctorSlice) {
    auto FunctorSymbols =
        ApproxDeclareTensorFunctorDecl::getDeclaredSymbolicVarsFromExpression(
            FS);
    std::unordered_set<std::string> SymbolNames;
    for (auto *Symbol : FunctorSymbols) {
        ApproxIndexVarRefExpr *First =
            dyn_cast_or_null<ApproxIndexVarRefExpr>(Symbol);
        SymbolNames.insert(std::string(First->getDeclName()));
    }
    assert(SymbolNames.size() == 1 &&
           "Expected exactly one symbolic var in functor slice");
    Expr *FunctorSymbol = FunctorSymbols[0];
    std::string Name = *SymbolNames.begin();
    // assign the range member of InfoMap at 'Name' to TensorSlice[idx]
    auto it = InfoMap.find(Name);
    if (it != InfoMap.end()) {
        it->second.Range = (TensorSlice[idx]);
        it->second.isFromRHS = true;
    }
    ++idx;
  }
}

Address CGApproxRuntime::EmitDeclarationOfSymbolVar(CodeGenFunction &CGF, ApproxIndexVarRefExpr *Symbol) {
  assert(Symbol->hasDecl() && "Expected a symbol with a declaration");
  VarDecl *SymbolDecl = *Symbol->getDecl();
  QualType SymbolTy = SymbolDecl->getType();
  llvm::StringRef SymbolName = SymbolDecl->getName();
  CGF.EmitDecl(*SymbolDecl);
  Address SymbolAddr = CGF.GetAddrOfLocalVar(SymbolDecl);
  // Address SymbolAddr = CGF.CreateMemTemp(SymbolTy, SymbolName);

  return SymbolAddr;
}

void CGApproxRuntime::EmitDeclarationOfSymbolVars(CodeGenFunction &CGF, llvm::ArrayRef<Expr*> Symbols) {
  // For each slice element in TensorSlice, emit a declaration of the symbolic
  // variables in the slice element
  // For example, if TensorSlice is [i, j, k], then emit declarations for i, j, k
  // Note that we want the same symbolic var in a slice to share the same declaration. For instance,
  // we want to create one declaration for a slice like: [i*6*i*6+6]
  for(Expr *E : Symbols) {
    ApproxIndexVarRefExpr *First = dyn_cast_or_null<ApproxIndexVarRefExpr>(E);
    assert(First && "Expected an index var ref expr");
    auto SymbolDeclaredName = std::string(First->getDeclName());
    if(SurrogateInfo.SymbolVars[SymbolDeclaredName].Addr.has_value()) {
      // we have already emitted a declaration for this symbol
      continue;
    }


    Address Location = EmitDeclarationOfSymbolVar(CGF, First);
    SurrogateInfo.SymbolVars[SymbolDeclaredName].Addr = Location;
  }
}

    void CGApproxRuntime::emitApproxDeclareTensorFunctor(
        CodeGenFunction *CGF, const ApproxDeclareTensorFunctorDecl *D) {
  llvm::dbgs() << "Emitting tensor functor declaration for functor "
               << D->getName() << "\n";
  }


  void CGApproxRuntime::initializeAndDeclareSymbolVars(ApproxDeclareTensorFunctorDecl *Decl, llvm::ArrayRef<Expr*> Vars) {
    for(auto *Var : Vars) {
      Decl->DeclareSymbolicVar(Var);
      ApproxIndexVarRefExpr *First = dyn_cast_or_null<ApproxIndexVarRefExpr>(Var);
      auto SymbolName = std::string(First->getDecl().value()->getName());
      SurrogateInfo.SymbolVars[SymbolName] = SymbolVarInfo(First);
    }

  }

  void addSymbolDeclarationsToNDSlice(ApproxDeclareTensorFunctorDecl *Decl, llvm::ArrayRef<Expr*> NDSlice) {
      for (auto *Expr : NDSlice) {
        ApproxSliceExpr *Slice = dyn_cast<ApproxSliceExpr>(Expr);
        auto DeclaredSymbVars =
            Decl->getDeclaredSymbolicVarsFromExpression(Slice);
        // not all will have symbolic vars, e.g., [i, 0:6]
        if(DeclaredSymbVars.size() == 0) {
          continue;
        }
        auto SymbVars = Decl->getSymbolicVarsFromExpression(Slice);

        auto *AIVRE =
            dyn_cast_or_null<ApproxIndexVarRefExpr>(DeclaredSymbVars[0]);
        auto *Decl = AIVRE->getDecl().value();
        auto VarName = std::string(AIVRE->getName());

        for (auto *Symbol : SymbVars) {
          ApproxIndexVarRefExpr *ThisSymbol =
              dyn_cast_or_null<ApproxIndexVarRefExpr>(Symbol);
          if (!ThisSymbol->hasDecl()) {
            assert(std::string(ThisSymbol->getName()) == VarName &&
                   "Expected the same symbol in a slice to have the same name");
            ThisSymbol->setDecl(Decl);
          }
        }
      }

  }

  void addSymbolDeclarationsToSlices(ApproxDeclareTensorFunctorDecl *Decl) {
    auto RHS = Decl->getRHSSlices();
    for (auto &NDSlice : RHS) {
      addSymbolDeclarationsToNDSlice(Decl, NDSlice);
    }

    auto LHS = Decl->getLHSSlice();
    addSymbolDeclarationsToNDSlice(Decl, LHS);
  }


Address CGApproxRuntime::CGApproxRuntimeCreateVoidPtrArray(CodeGenFunction &CGF, llvm::ArrayRef<Address> Vars) {
  size_t num_vars = Vars.size();
  // we want to create an array of type "void *" that holds all addresses in Vars
  // we will then return the address of this array

  QualType VoidPtrArrayType = CGF.getContext().getConstantArrayType(
      CGF.getContext().VoidPtrTy, llvm::APInt(64, num_vars), nullptr,
      ArrayType::Normal, 0);
  Address Array = CGF.CreateMemTemp(VoidPtrArrayType, "allocated_ptrs");

  for(size_t i = 0; i < num_vars; i++) {
    Address ElementPtr = CGF.Builder.CreateConstArrayGEP(Array, i);
    CGF.Builder.CreateStore(Vars[i].getPointer(), ElementPtr);
  }

  return Array;
}

void CGApproxRuntime::CGApproxRuntimeEmitSliceConversion(
    CodeGenFunction &CGF, size_t NumVals, Address TensorCollection,
    Address FunctorCollection) {
      Function *Fn = nullptr;
      StringRef FnName("__approx_runtime_slice_conversion");
      Fn = CGM.getModule().getFunction(FnName);
      if(!Fn) {
        Fn = Function::Create(SurrogateInfo.ConvertSliceInfoFnTy,
                          llvm::Function::ExternalLinkage, FnName,
                          CGM.getModule());
      }

      auto *NumValsArg = llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), llvm::APInt(32, NumVals));
      llvm::FunctionCallee FnCallee({SurrogateInfo.ConvertSliceInfoFnTy, Fn});
      llvm::Value *TensorCollectionValue = CGF.Builder.CreatePointerCast(TensorCollection.getPointer(), CGF.VoidPtrTy);
      llvm::Value *FunctorCollectionValue = CGF.Builder.CreatePointerCast(FunctorCollection.getPointer(), CGF.VoidPtrTy);
      CGF.EmitRuntimeCall(FnCallee, {NumValsArg, TensorCollectionValue, FunctorCollectionValue});
    }

void CGApproxRuntime::CGApproxRuntimeEmitHigherOrderShapeConversion(
  CodeGenFunction &CGF, size_t NumVals, Address TensorCollection,
  Address FunctorCollection) {
    Function *Fn = nullptr;
    StringRef FnName("__approx_runtime_convert_to_higher_order_shapes");
    Fn = CGM.getModule().getFunction(FnName);
    if(!Fn) {
      Fn = Function::Create(SurrogateInfo.ConvertToHigherOrderShapeFnTy,
                        llvm::Function::ExternalLinkage, FnName,
                        CGM.getModule());
    }

  auto *NumValsArg = llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), llvm::APInt(32, NumVals));
  llvm::FunctionCallee FnCallee({SurrogateInfo.ConvertToHigherOrderShapeFnTy, Fn});
  llvm::Value *TensorCollectionValue = CGF.Builder.CreatePointerCast(TensorCollection.getPointer(), CGF.VoidPtrTy);
  llvm::Value *FunctorCollectionValue = CGF.Builder.CreatePointerCast(FunctorCollection.getPointer(), CGF.VoidPtrTy);
  CGF.EmitRuntimeCall(FnCallee, {NumValsArg, TensorCollectionValue, FunctorCollectionValue});

}

namespace {
  class TensorCleanupTy final : public EHScopeStack::Cleanup {
    public:
      static const int TensorCleanupFinArgs = 1;

    private:
      llvm::FunctionCallee RTLFn;
      llvm::Value *Args[TensorCleanupFinArgs];

    public:
      TensorCleanupTy(llvm::FunctionCallee RTLFN, ArrayRef<llvm::Value *> CallArgs) : RTLFn{RTLFN} {
        assert(CallArgs.size() == TensorCleanupFinArgs && "Expected 1 argument for tensor cleanup");
        std::copy(CallArgs.begin(), CallArgs.end(), std::begin(Args));
      }
    
      void Emit(CodeGenFunction &CGF, Flags) override {
        CGF.EmitRuntimeCall(RTLFn, Args);
      }
};
} //namespace

void CGApproxRuntime::emitApproxDeclareTensor(
    CodeGenFunction *CGF, const ApproxDeclareTensorDecl *D) {

  bool isToMem = D->getDirectionality() ==
                 ApproxDeclareTensorDecl::Direction::TENSOR_TO_MEM;
  std::unique_ptr<TensorMemConversionDispatcher> Dispatcher;
  if (isToMem) {
    Dispatcher = std::make_unique<TensorToMemDispatcher>();
    llvm::dbgs() << "Emitting approx declare tensor for output tensor "
                 << D->getName() << "\n";
  } else {
    Dispatcher = std::make_unique<MemToTensorDispatcher>();
    llvm::dbgs() << "Emitting approx declare tensor for input tensor "
                 << D->getName() << "\n";
  }

  emitApproxDeclareTensorImpl(CGF, D, *Dispatcher);
}

void CGApproxRuntime::emitApproxDeclareTensorImpl(CodeGenFunction *CGF, const ApproxDeclareTensorDecl *D, 
TensorMemConversionDispatcher& Dispatcher) {

  auto *TensorFunctor =
      dyn_cast<ApproxDeclareTensorFunctorDecl>(D->getFunctor());
  auto IndexRefExprsRHS = TensorFunctor->getSymbolicVarsUniqueToEachSlice();
  auto IndexRefExprsLHS = TensorFunctor->getSymbolicVarsUniqueToEachLHSSlice();
  initializeAndDeclareSymbolVars(TensorFunctor, IndexRefExprsRHS);
  initializeAndDeclareSymbolVars(TensorFunctor, IndexRefExprsLHS);
  addSymbolDeclarationsToSlices(TensorFunctor);
  EmitDeclarationOfSymbolVars(*CGF, IndexRefExprsRHS);
  EmitDeclarationOfSymbolVars(*CGF, IndexRefExprsLHS);

  auto RHS = TensorFunctor->getRHSSlices();
  auto LHS = TensorFunctor->getLHSSlice();

  auto ArrSlices = D->getArraySlices();
  assert(RHS.size() == ArrSlices.size() &&
         "Expected same number of RHS slices and array slices");
  for (size_t i = 0; i < RHS.size(); i++) {
    ApproxArraySliceExpr *Slice = dyn_cast<ApproxArraySliceExpr>(ArrSlices[i]);
    assert(Slice && "Expected an array slice expression");
    auto SliceVals = Slice->getSlices();
    mapSymbolicVarsToRanges(SurrogateInfo.SymbolVars, RHS[i], SliceVals);
  }
  CGApproxRuntimeEmitSymbolicVarInits(*CGF);

  std::vector<Address> TensorDeclAddresses;
  std::vector<Address> FunctorDeclRHSAddresses;

  for (auto *E : D->getArraySlices()) {
    auto *ArraySlices = dyn_cast<ApproxArraySliceExpr>(E);
    auto Addr = CGApproxRuntimeEmitApproxArrayInfo(*CGF, ArraySlices);
    TensorDeclAddresses.push_back(Addr);
  }

  for (auto *E : TensorFunctor->getRHSSliceExprs()) {
    auto *Slice = dyn_cast<ApproxArraySliceExpr>(E);
    auto Addr = CGApproxRuntimeEmitApproxArrayInfo(*CGF, Slice);
    FunctorDeclRHSAddresses.push_back(Addr);
  }

  assert(TensorDeclAddresses.size() == FunctorDeclRHSAddresses.size() &&
         "Expected same number of tensor decl addresses and functor decl "
         "addresses");

  auto TensorCollectionAddr =
      CGApproxRuntimeCreateVoidPtrArray(*CGF, TensorDeclAddresses);
  auto FunctorCollectionAddr =
      CGApproxRuntimeCreateVoidPtrArray(*CGF, FunctorDeclRHSAddresses);

  CGApproxRuntimeEmitSliceConversion(*CGF, TensorDeclAddresses.size(),
                                     TensorCollectionAddr,
                                     FunctorCollectionAddr);
  CGApproxRuntimeEmitHigherOrderShapeConversion(*CGF, TensorDeclAddresses.size(),
                                     TensorCollectionAddr,
                                     FunctorCollectionAddr);

  auto LHSSliceAddress = CGApproxRuntimeEmitSlices(*CGF, LHS);
  auto LHSShapeAddress = CGApproxRuntimeEmitShape(*CGF, LHS);

  CGApproxRuntimeSubstituteAIVRInShapes(*CGF, LHS.size(), LHSSliceAddress, LHSShapeAddress);

  Address InternalRepr = CGApproxRuntimeEmitInternalReprConversion(
      *CGF, LHS.size(), LHSSliceAddress, LHSShapeAddress,
      FunctorDeclRHSAddresses.size(), FunctorCollectionAddr, Dispatcher);
  llvm::Value *InternalReprValue = CGF->Builder.CreatePointerCast(InternalRepr.getPointer(), CGF->VoidPtrTy);

  CGF->AddDeclaredTensorLocalVar((VarDecl*) D, InternalRepr);

  llvm::FunctionCallee CleanFNCall = getTensorCleanupFn(CGM);
  llvm::SmallVector<llvm::Value *, 1> CallArgs;
  CGF->EHStack.pushCleanup<TensorCleanupTy>(NormalAndEHCleanup, CleanFNCall, llvm::makeArrayRef(InternalReprValue));
}



llvm::FunctionCallee CGApproxRuntime::getTensorCleanupFn(CodeGenModule &CGM) {
  Function *Fn = nullptr;
  StringRef FnName("__approx_runtime_tensor_cleanup");
  Fn = CGM.getModule().getFunction(FnName);
  if(!Fn) {
    Fn = Function::Create(SurrogateInfo.TensorCleanupFnTy,
                      llvm::Function::ExternalLinkage, FnName,
                      CGM.getModule());
  }

  llvm::FunctionCallee FnCallee({SurrogateInfo.TensorCleanupFnTy, Fn});
  return FnCallee;
}
  

  void CGApproxRuntime::CGApproxRuntimeSubstituteAIVRInShapes(CodeGenFunction& GF, int ndim, Address Slices, Address Shapes) {
    Function *Fn = nullptr;
    StringRef FnName("__approx_runtime_substitute_aivr_in_shapes");
    Fn = CGM.getModule().getFunction(FnName);
    if(!Fn) {
      Fn = Function::Create(SurrogateInfo.SubstituteAIVRInShapesFnTy,
                        llvm::Function::ExternalLinkage, FnName,
                        CGM.getModule());
    }

    llvm::FunctionCallee FnCallee({SurrogateInfo.SubstituteAIVRInShapesFnTy, Fn});
    llvm::Value *NumDimsArg = llvm::ConstantInt::get(GF.Builder.getInt32Ty(), llvm::APInt(32, ndim));
    llvm::Value *SlicesValue = GF.Builder.CreatePointerCast(Slices.getPointer(), GF.VoidPtrTy);
    llvm::Value *ShapesValue = GF.Builder.CreatePointerCast(Shapes.getPointer(), GF.VoidPtrTy);
    GF.EmitRuntimeCall(FnCallee, {NumDimsArg, SlicesValue, ShapesValue});
  }
  Address CGApproxRuntime::CGApproxRuntimeEmitInternalReprConversion(CodeGenFunction &CGF, int nargsLHS, Address LHSSlices, Address LHSShapes,
      int nargsRHS, Address RHSAddress, TensorMemConversionDispatcher &Dispatcher) {

    llvm::FunctionCallee FnCallee = Dispatcher.getInternalReprConversionFn(CGM, *this);

    llvm::Value *NumArgsLHSArg = llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), llvm::APInt(32, nargsLHS));
    llvm::Value *SlicesLHSArg = CGF.Builder.CreatePointerCast(LHSSlices.getPointer(), CGF.VoidPtrTy);
    llvm::Value *ShapesLHSArg = CGF.Builder.CreatePointerCast(LHSShapes.getPointer(), CGF.VoidPtrTy);

    llvm::Value *NumArgsRHSArg = llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), llvm::APInt(32, nargsRHS));
    llvm::Value *RHSArg = CGF.Builder.CreatePointerCast(RHSAddress.getPointer(), CGF.VoidPtrTy);
    auto *RetVal = CGF.EmitRuntimeCall(FnCallee, {NumArgsLHSArg, SlicesLHSArg, ShapesLHSArg, NumArgsRHSArg, RHSArg});
    return Address(RetVal, CGF.VoidPtrTy, CGF.getPointerAlign());
  }


  Address CGApproxRuntime::CGApproxRuntimeAllocInternalReprMetadata(CodeGenFunction& CGF, int numArgs) {
    ASTContext &C = CGM.getContext();
    int numArgsAfterConversion = numArgs > 1 ? numArgs + 1 : 1;

    Twine ReprName = "internal_repr_";
    auto *ReprDecl = SurrogateInfo.InternalReprMetadataTy->getAsRecordDecl();
    auto InternalRepr = CGF.CreateMemTemp(SurrogateInfo.InternalReprMetadataTy, ReprName);
    auto InternalReprAddr = CGF.MakeAddrLValue(InternalRepr, SurrogateInfo.InternalReprMetadataTy);

    LValue Base = CGF.EmitLValueForField(InternalReprAddr, *ReprDecl->field_begin());
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), llvm::APInt(32, numArgs, true)), Base);

    Base = CGF.EmitLValueForField(InternalReprAddr, *std::next(ReprDecl->field_begin(), 1));
    Base = CGF.EmitLValueForField(InternalReprAddr, *std::next(ReprDecl->field_begin(), 2));

    llvm::Value *NullPtr = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
    CGF.EmitStoreOfScalar(NullPtr, Base);

    return InternalRepr;
  }

  llvm::FunctionCallee MemToTensorDispatcher::getInternalReprConversionFn(CodeGenModule &CGM, CGApproxRuntime &Runtime) {
    Function *Fn = nullptr;
    llvm::FunctionType *FnTy = nullptr;
    StringRef FnName;

      FnName = "__approx_runtime_convert_internal_mem_to_tensor";
      Fn = CGM.getModule().getFunction(FnName);
      FnTy = Runtime.SurrogateInfo.ConversionMemToTensorFnTy;

    Fn = CGM.getModule().getFunction(FnName);
    if(!Fn) {
      Fn = Function::Create(FnTy,
                        llvm::Function::ExternalLinkage, FnName,
                        CGM.getModule());
    }

    return llvm::FunctionCallee({FnTy, Fn});
  }

  llvm::FunctionCallee TensorToMemDispatcher::getInternalReprConversionFn(CodeGenModule &CGM, CGApproxRuntime &Runtime) {
    Function *Fn = nullptr;
    llvm::FunctionType *FnTy = nullptr;
    StringRef FnName;

    FnName = "__approx_runtime_convert_internal_tensor_to_mem";
    Fn = CGM.getModule().getFunction(FnName);
    FnTy = Runtime.SurrogateInfo.ConversionTensorToMemFnTy;

    Fn = CGM.getModule().getFunction(FnName);
    if(!Fn) {
      Fn = Function::Create(FnTy,
                        llvm::Function::ExternalLinkage, FnName,
                        CGM.getModule());
    }

    return llvm::FunctionCallee({FnTy, Fn});
  }
