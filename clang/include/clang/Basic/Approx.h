//===--- Approx.h - Approx enums ---------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some Approx-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_APPROX_H
#define LLVM_CLANG_BASIC_APPROX_H

#include "clang/Basic/SourceLocation.h"

#include <optional>

namespace clang {
namespace approx {

enum ApproxRuntimeKind {
  /// The runtime that deploys approximations as instructed by the source code file
  APPROX_DEPLOY = 1,
  /// The runtime that profiles approximate regions. No approximation takes place
  APPROX_PROFILE_TIME,
  ///The runtime that profiles the data values pipelined through the annotated regions.
  APPROX_PROFILE_DATA,
  /// An unknown option.
  APPROX_Unknown
};

enum ClauseKind : uint {
  CK_PERFO = 0,
  CK_MEMO,
  CK_DT,
  CK_NN,
  CK_USER,
  CK_IF,
  CK_IN,
  CK_OUT,
  CK_INOUT,
  CK_LABEL,
  CK_PETRUBATE,
  CK_ML,
  CK_END
};

const unsigned CK_START = CK_PERFO;

enum DeclKind : uint {
  DK_TF = 0,
  DK_T,
  DK_END
};

const unsigned DK_START = DK_TF;


enum PerfoType : uint {
  PT_SMALL = 0,
  PT_LARGE,
  PT_RAND,
  PT_SINIT,
  PT_SFINAL,
  PT_END
};
const unsigned PT_START = PT_SMALL;

enum DeclType : uint {
  DT_TENSOR_fUNCTOR = 0,
  DT_TENSOR,
  DT_END
};
const unsigned DT_START = DT_TENSOR_fUNCTOR;


enum MemoType : uint {
  MT_IN = 0,
  MT_OUT,
  MT_END
};

const unsigned MT_START = MT_IN;

enum PetrubateType: uint {
  PETRUBATE_IN = 0,
  PETRUBATE_OUT,
  PETRUBATE_INOUT,
  PETRUBATE_END
};

enum MLType: uint {
  ML_ONLINETRAIN = 0,
  ML_OFFLINETRAIN,
  ML_INFER,
  ML_END
};

const unsigned ML_START = ML_ONLINETRAIN;

const unsigned PETRUBATE_START = PETRUBATE_IN;

struct ApproxVarListLocTy {
  SourceLocation StartLoc;
  SourceLocation LParenLoc;
  SourceLocation EndLoc;
  ApproxVarListLocTy() = default;
  ApproxVarListLocTy(SourceLocation StartLoc, SourceLocation LParenLoc,
                     SourceLocation EndLoc)
      : StartLoc(StartLoc), LParenLoc(LParenLoc), EndLoc(EndLoc) {}
};

struct ApproxSliceLocTy {
  std::optional<SourceLocation> StartLoc;
  std::optional<SourceLocation> StopLoc;
  std::optional<SourceLocation> StepLoc;

  ApproxSliceLocTy() = default;
  ApproxSliceLocTy(std::optional<SourceLocation> StartLoc,
                   std::optional<SourceLocation> StopLoc,
                   std::optional<SourceLocation> StepLoc)
      : StartLoc(StartLoc), StopLoc(StopLoc), StepLoc(StepLoc) {}
};

} // namespace approx
} // namespace clang

#endif // LLVM_CLANG_BASIC_APPROX_H
