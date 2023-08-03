//===----- CGApproxRuntime.h - Interface to Approx Runtimes -----*- C++ -*-===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGAPPROXRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGAPPROXRUNTIME_H

#include "CGValue.h"
#include "clang/AST/ApproxClause.h"
#include "clang/AST/DeclApprox.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/StmtApprox.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace CodeGen {
class CodeGenModule;

enum ApproxType : int8_t {
#define APPROX_TYPE(Id, type, name) Id,
#include "clang/Basic/approxTypes.def"
  INVALID
};

enum ApproxRTArgsIndex : uint {
  AccurateFn = 0,
  PerfoFn,
  CapDataPtr,
  Cond,
  Label,
  PerfoDesc,
  MemoDescr,
  PetruDescr,
  MLDescr,
  DataDescIn,
  DataSizeIn,
  DataDescOut,
  DataSizeOut,
  ARG_END
};

enum Directionality : int { Input = 1, Output = 2, InputOuput = 4 };

const unsigned ARG_START = AccurateFn;

class CGApproxRuntime {
private:
  CodeGenModule &CGM;
  /// PerfoInfoTy is a struct containing infor about the perforation.
  ///  typedef struct approx_perfo_info_t{
  ///    int type;
  ///    int region;
  ///    int step;
  ///    float rate;
  /// } approx_perfo_info_t;
  QualType PerfoInfoTy;

  /// VarInfoTy is a struct containing info about the in/out/inout variables
  /// of this region.
  ///    typedef struct approx_var_info_t{
  ///        void* ptr;         // Ptr to data
  ///        void* var_name;    // Name of the variable
  ///        size_t num_elem;   // Number of elements
  ///        size_t sz_elem;    // Size of elements in bytes
  ///        int8_t data_type; // Type of data float/double/int etc.
  ///        uint8_t dir;       // Direction of data: in/out/inout
  ///    } approx_var_info_t;
  QualType VarInfoTy;
  llvm::Value *approxRTParams[ARG_END];
  llvm::SmallVector<std::pair<Expr *, Directionality>, 16> Inputs;
  llvm::SmallVector<std::pair<Expr *, Directionality>, 16> Outputs;
  // Function type of callback functions.
  llvm::FunctionType *CallbackFnTy;
  // Function type of the runtime interface call.
  llvm::FunctionType *RTFnTy;
  int approxRegions;
  SourceLocation StartLoc;
  SourceLocation EndLoc;
  bool requiresData;
  bool requiresInputs;
  llvm::StringMap<llvm::Constant*> NameToConstant;

private:
  void CGApproxRuntimeEmitPerfoFn(CapturedStmt &CS, const ApproxLoopHelperExprs &LoopExprs, const ApproxPerfoClause &PC);
  std::pair<llvm::Value *, llvm::Value *> CGApproxRuntimeEmitData(CodeGenFunction &CGF, llvm::SmallVector<std::pair<Expr *, Directionality>, 16> &Data, const char *arrayName);

public:
  CGApproxRuntime(CodeGenModule &CGM);
  void CGApproxRuntimeEnterRegion(CodeGenFunction &CGF, CapturedStmt &CS);
  void CGApproxRuntimeEmitPerfoInit(CodeGenFunction &CGF, CapturedStmt &CS,
                                    ApproxPerfoClause &PerfoClause, const ApproxLoopHelperExprs &LoopExprs);
  void CGApproxRuntimeEmitMemoInit(CodeGenFunction &CGF,
                                   ApproxMemoClause &MemoClause);
  void CGApproxRuntimeEmitPetrubateInit(CodeGenFunction &CGF,
                                   ApproxPetrubateClause &PetrubateClause);
  void CGApproxRuntimeEmitMLInit( CodeGenFunction &CGF, 
                                  ApproxMLClause &MLClause);
  void CGApproxRuntimeEmitDeclInit(CodeGenFunction &CGF,
                                   ApproxDeclClause &DeclClause);
  void CGApproxRuntimeEmitIfInit(CodeGenFunction &CGF,
                                 ApproxIfClause &IfClause);
  void CGApproxRuntimeEmitLabelInit(CodeGenFunction &CGF, ApproxLabelClause &LabelCluse);
  void CGApproxRuntimeExitRegion(CodeGenFunction &CGF);
  void CGApproxRuntimeRegisterInputs(ApproxInClause &InClause);
  void CGApproxRuntimeRegisterOutputs(ApproxOutClause &OutClause);
  void CGApproxRuntimeRegisterInputsOutputs(ApproxInOutClause &InOutClause);
  void CGApproxRuntimeEmitDataValues(CodeGenFunction &CG);
  llvm::Constant* getOrCreateName(StringRef Name, CodeGenFunction &CGF);

  private:

class SymbolVarInfo {
  public:
  ApproxIndexVarRefExpr *Symbol = nullptr;
  std::optional<Address> Addr;
  Expr *Range = nullptr;
  bool isFromRHS = false;

  SymbolVarInfo(ApproxIndexVarRefExpr *Symbol, Address Addr, Expr *Range, bool isFromRHS = false) : Symbol(Symbol), Addr(Addr), Range(Range), isFromRHS(isFromRHS) {}
  SymbolVarInfo(ApproxIndexVarRefExpr *Symbol, Address Addr, bool isFromRHS = false) : Symbol(Symbol), Addr(Addr), isFromRHS(isFromRHS) {}
  SymbolVarInfo(ApproxIndexVarRefExpr *Symbol, bool isFromRHS = false) : Symbol(Symbol), isFromRHS(isFromRHS) {}
  SymbolVarInfo() {}

  void setAddress(Address A) {
    Addr = A;
  }

};

// Maps a symbolic in a tensor functor decl to the its range as given
// in the tensor decl
using SymbolVarInfoMap = std::unordered_map<std::string, SymbolVarInfo>;

  struct MLSurrogateInfo {
  // SliceInfoTy is a struct containing information about the slice for one
  // dimension. typedef struct slice_info_ty 
  // { 
  //   int start; 
  //   int stop; 
  //   int step;
  //} slice_info_t
  QualType SliceInfoTy;

  // NDArraySliceTy is a struct containing information about an ND
  // array slice.
  // It looks like:
  // typedef struct ndarray_slice_ty {

  // void* base;
  // int8_t type;
  // int ndim;
  // slice_info_t slices[ndim];
  // } ndarray_slice_t;
  QualType NDArraySliceTy;


  //typedef struct tensor_shape {
  //	int ndim;
  //	int *shapes;
  //} tensor_shape_t;
  QualType TensorShapeTy;

  SymbolVarInfoMap SymbolVars;

  // Function type of the function that takes
  // the tensor decl array slices and the
  // tensor functor decl slices and converts the
  // local i'th slice to global slices for the array
  llvm::FunctionType *ConvertSliceInfoFnTy;
  };

  MLSurrogateInfo SurrogateInfo;

    void mapSymbolicVarsToRanges(
        SymbolVarInfoMap &InfoMap,
        llvm::ArrayRef<Expr *> FunctorSlice,
        llvm::ArrayRef<Expr *> TensorSlice);
  void initializeAndDeclareSymbolVars(ApproxDeclareTensorFunctorDecl *Decl, llvm::ArrayRef<Expr*> Vars);
  Address EmitDeclarationOfSymbolVar(CodeGenFunction &CGF, ApproxIndexVarRefExpr *Symbol);
  Address CGApproxRuntimeEmitApproxArrayInfo(CodeGenFunction &CGF, Expr *AAIE);
  Address CGApproxRuntimeEmitSlices(CodeGenFunction &CGF, llvm::ArrayRef<Expr*> Slices);
  Address CGApproxRuntimeAllocateShape(CodeGenFunction &CGF, int ndim);
  Address CGApproxRuntimeEmitShape(CodeGenFunction &CGF, llvm::ArrayRef<Expr*> Slices);
  Address CGApproxRuntimeEmitShape(CodeGenFunction& CGF, Address Dest, llvm::ArrayRef<Expr*> Slices);
  void CGApproxRuntimeEmitSliceSize(CodeGenFunction &CGF, Expr *Slice, Address Dest);
  Address CGApproxRuntimeEmitSizeOfSliceElement(CodeGenFunction &CGF, std::unordered_map<Expr*,Expr*> &RangeMap, Expr *Slice);
  void CGApproxRuntimeEmitSymbolicVarInits(CodeGenFunction &CGF);
  void EmitDeclarationOfSymbolVars(CodeGenFunction &CGF, llvm::ArrayRef<Expr*> Symbols);
  void CGApproxRuntimeEmitSlice(CodeGenFunction &CFG, Expr *Slice, Address SliceMemory);
  llvm::Value *CGApproxRuntimeCreateVoidPtrArray(CodeGenFunction &CGF, llvm::ArrayRef<Address> Vars);
  void CGApproxRuntimeEmitSliceConversion(CodeGenFunction &CGF, size_t NumVals, llvm::Value *TensorCollection, llvm::Value *FunctorCollection);
  public:

  void emitApproxDeclareTensorFunctor(CodeGenFunction *CGF, const ApproxDeclareTensorFunctorDecl *D);
  void emitApproxDeclareTensor(CodeGenFunction *CGF, const ApproxDeclareTensorDecl *D);

};

} // namespace CodeGen
} // namespace clang

#endif
