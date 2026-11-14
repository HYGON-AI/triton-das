// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_PACKED_CONVERSION_H
#define TRITON_HCU_PACKED_CONVERSION_H

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/TargetParser/TargetParser.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::HCU {
inline bool supportsPackedFp32Downcast(llvm::AMDGPU::GPUKind gpuKind) {
  return llvm::is_contained(
      {llvm::AMDGPU::GPUKind::GK_GFX938, llvm::AMDGPU::GPUKind::GK_GFX92A,
       llvm::AMDGPU::GPUKind::GK_GFX946, llvm::AMDGPU::GPUKind::GK_GFX948},
      gpuKind);
}

// Convert the two values directly through typed ROCDL operations.
inline SmallVector<Value>
convertFp32PairToPacked(Location loc, ConversionPatternRewriter &rewriter,
                        Type outElemTy, Value lo, Value hi) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert((outElemTy.isBF16() || outElemTy.isF16()) &&
         "unsupported packed conversion destination");
  Type packedTy = vec_ty(outElemTy, 2);
  auto modifier = rewriter.getI32IntegerAttr(0);
  auto clamp = rewriter.getBoolAttr(false);
  Value packedVal;
  if (outElemTy.isF16())
    packedVal = ROCDL::HCUCvtPkF16F32Op::create(
        rewriter, loc, packedTy, modifier, lo, modifier, hi, clamp,
        rewriter.getI32IntegerAttr(0));
  else
    packedVal = ROCDL::HCUCvtPkBf16F32Op::create(
        rewriter, loc, packedTy, modifier, lo, modifier, hi, clamp);
  return {b.extract_element(outElemTy, packedVal, b.i32_val(0)),
          b.extract_element(outElemTy, packedVal, b.i32_val(1))};
}

} // namespace mlir::triton::HCU
#endif
