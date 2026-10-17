// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::HCU {
// Drop only the implicit lgkmcnt(0) wait at this publication barrier: late
// lifetime analysis proves a later rendezvous drains LDS reads before slot
// reuse. The async-copy wait and workgroup barrier are never removed.
inline constexpr llvm::StringLiteral kDropPublicationLgkmcnt =
    "hcu.block_pingpong.drop_lgkmcnt";
inline constexpr llvm::StringLiteral kBufferToLdsPublication =
    "hcu.block_pingpong.publish";

namespace planned {
// Maximum supported global prefetch distance, not a container capacity or
// LLVM async-mark history limit.
inline constexpr unsigned kMaxGlobalPrefetchDistance = 8;

// Validate resolved defaults as well as explicit options before changing IR.
inline bool hasValidPrefetchDistances(unsigned global, unsigned lds) {
  return global >= 2 && global <= kMaxGlobalPrefetchDistance && lds >= 1 &&
         lds < global;
}

} // namespace planned
} // namespace mlir::triton::HCU
