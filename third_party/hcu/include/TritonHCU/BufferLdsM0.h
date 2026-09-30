// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "TritonHCU/BufferLdsConfig.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::HCU {
// LLVM's current async-LDS intrinsic accepts M0 through its AS3 operand.
// Keep that ABI adaptation at emission: callers carry a genuine LDS address
// and a separate wave-uniform wrap distance. A future integer-M0 intrinsic
// can replace this adapter without changing the shared layout or scheduling.
inline Value encodeBufferLdsM0(RewriterBase &rewriter, Location loc,
                               Value ldsAddress, Value wrapBytes,
                               llvm::StringRef arch) {
  auto profile =
      getBufferLdsTargetProfile(std::string_view(arch.data(), arch.size()));
  assert(profile && "wrap emission requires a validated target profile");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value field;
  switch (profile->wrapEncoding) {
  case BufferLdsWrapEncoding::BmzCoarse4Dword:
    field = b.udiv(wrapBytes, b.i32_val(16));
    break;
  case BufferLdsWrapEncoding::SplitCoarse4DwordFineDword: {
    Value dwords = b.udiv(wrapBytes, b.i32_val(4));
    field = b.or_(b.lshr(dwords, b.i32_val(2)),
                  b.shl(b.and_(dwords, b.i32_val(3)), b.i32_val(3)));
    break;
  }
  case BufferLdsWrapEncoding::LinearDword:
    field = b.udiv(wrapBytes, b.i32_val(4));
    break;
  }
  Value m0 = b.or_(b.ptrtoint(rewriter.getI32Type(), ldsAddress),
                   b.shl(field, b.i32_val(profile->wrapFieldOffset)));
  return b.inttoptr(ldsAddress.getType(), m0);
}

// Apply the per-producer-wave rotation owned by the shared encoding. Wave w
// rotates its LDS transfer by
//
//   (w % wrapCount) * wrapStepBytes.
//
// All supported configurations use a power-of-two wrapCount, so modulo lowers to a
// mask. The wave id is required because M0 is wave-uniform while each producer
// wave may use a different rotation. The target-specific bit packing remains
// in encodeBufferLdsM0().
inline Value
encodeBufferLdsWaveM0(RewriterBase &rewriter, Location loc, Value ldsAddress,
                      Value producerWaveId,
                      gpu::HCUBufferLdsSharedEncodingAttr sharedEncoding,
                      llvm::StringRef arch) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value wrapBytes = b.mul(
      b.and_(producerWaveId, b.i32_val(sharedEncoding.getWrapCount() - 1)),
      b.i32_val(sharedEncoding.getWrapStepBytes()));
  return encodeBufferLdsM0(rewriter, loc, ldsAddress, wrapBytes, arch);
}
} // namespace mlir::triton::HCU
