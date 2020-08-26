//===----- CGApproxRuntime.h - Interface to OpenMP Runtimes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGAPPROXRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGAPPROXRUNTIME_H

#include "CGValue.h"
#include "clang/AST/ApproxClause.h"
#include "clang/AST/DeclApprox.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace CodeGen {
class CodeGenModule;

enum ApproxRTArgsIndex : uint {
  AccurateFn = 0,
  PerfoFn,
  CapDataPtr,
  Cond,
  PerfoDesc,
  DataDesc,
  DataSize,
  LAST
};

enum Directionality : int {
  Input = 1,
  Output = 2,
  InputOuput = 4
};

const unsigned ARG_START = AccurateFn;
const unsigned ARG_END = LAST;

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
  ///  typedef struct approx_var_info_t{
  ///    int_ptr_t ptr; /// pointer pointing to data
  ///    int direction; /// in:0, out:1, inout:2
  ///    int data_type; /// unique id per type
  ///    long num_elements; /// number of elements in the vector
  ///    int opaque; /// place holder for scalar in/out/inout values
  /// } approx_var_info_t;
  QualType VarInfoTy;
  bool hasPerfo;
  llvm::SmallVector<llvm::Value *, ARG_END> approxRTParams;
  llvm::SmallVector<llvm::Type *, ARG_END> approxRTTypes;
  llvm::SmallVector<std::pair<Expr *, Directionality>, 16> Data;
  int approxRegions;
  SourceLocation StartLoc;
  SourceLocation EndLoc;

public:
  CGApproxRuntime(CodeGenModule &CGM);
  void CGApproxRuntimeEnterRegion(CodeGenFunction &CGF, CapturedStmt &CS);
  void CGApproxRuntimeEmitPerfoInit(CodeGenFunction &CGF,
                                    ApproxPerfoClause &PerfoClause);
  void CGApproxRuntimeEmitIfInit(CodeGenFunction &CGF,
                                 ApproxIfClause &IfClause);
  void CGApproxRuntimeEmitPerfoFn(CapturedStmt &CS);
  void CGApproxRuntimeExitRegion(CodeGenFunction &CGF);
  void CGApproxRuntimeRegisterInputs(ApproxInClause &InClause);
  void CGApproxRuntimeRegisterOutputs(ApproxOutClause &OutClause);
  void CGApproxRuntimeRegisterInputsOutputs(ApproxInOutClause &InOutClause);
  void CGApproxRuntimeEmitDataValues(CodeGenFunction &CG);
};

} // namespace CodeGen
} // namespace clang

#endif
