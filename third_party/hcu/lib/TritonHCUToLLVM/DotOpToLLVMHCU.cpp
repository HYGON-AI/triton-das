// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

namespace mlir::triton::HCU {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::HCU

namespace {

// HCU-specific DotOp conversion that uses HCU::convertMFMA (with MMAC support).
struct HCUDotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto D = op.getResult();
    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (!isa<mlir::triton::gpu::AMDMfmaEncodingAttr>(dEncoding))
      return failure();
    return mlir::triton::HCU::convertMFMA(op, adaptor, getTypeConverter(),
                                          rewriter);
  }
};

struct HCUScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto D = op.getResult();
    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (!isa<mlir::triton::gpu::AMDMfmaEncodingAttr>(dEncoding))
      return failure();
    return mlir::triton::HCU::convertScaledMFMA(op, adaptor, getTypeConverter(),
                                                rewriter);
  }
};
} // namespace

namespace mlir::triton::HCU {
void populateHCUDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<HCUDotOpConversion, HCUScaledDotOpConversion>(typeConverter,
                                                             benefit);
}
} // namespace mlir::triton::HCU
