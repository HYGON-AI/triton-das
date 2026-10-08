// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "BufferLdsAddress.h"
#include "TritonHCU/BufferLdsEncoding.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::HCU;

namespace {

struct BufferLdsLocalLoadLowering : ConvertOpToLLVMPattern<LocalLoadOp> {
  BufferLdsLocalLoadLowering(LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto sharedEncoding =
        dyn_cast<HCUBufferLdsSharedEncodingAttr>(srcTy.getEncoding());
    // Matrix reads have their own coordinate-to-physical mapping in DsReadM.
    // Unmarked local loads and all ordinary local stores use common lowering.
    if (!sharedEncoding ||
        canUseBufferLdsMatrixRead(srcTy, dyn_cast<DotOperandEncodingAttr>(
                                             op.getType().getEncoding())))
      return failure();

    auto arch = targetInfo.getArch();
    auto profile =
        getBufferLdsTargetProfile(std::string_view(arch.data(), arch.size()));
    auto copyConfig = getBufferLdsConfig(sharedEncoding);
    if (!profile || !isBufferLdsConfigEncodable(copyConfig, *profile))
      return rewriter.notifyMatchFailure(
          op, "buffer-to-LDS layout is not encodable on this target");
    auto dstTy = op.getType();
    auto cvt = toLinearLayout(dstTy).invertAndCompose(toLinearLayout(srcTy));
    auto *ctx = op.getContext();
    auto dim = [&](StringRef name) { return StringAttr::get(ctx, name); };
    if (!cvt.isTrivialOver({dim("block")}))
      return failure();
    cvt = cvt.sublayout({dim("register"), dim("lane"), dim("warp")},
                        {dim("offset")});

    auto loc = op.getLoc();
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         elemTy, rewriter);
    TritonLLVMOpBuilder b(loc, rewriter);
    auto physicalOffset = [&](Value offset) {
      return emitBufferLdsWrappedByteOffset(b, copyConfig, offset);
    };
    unsigned elementBytes = srcTy.getElementTypeBitWidth() / 8;
    unsigned maxReadBytes = copyConfig.copyBytesPerLane;
    if (auto dot = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding())) {
      auto mma = dyn_cast<AMDMfmaEncodingAttr>(dot.getParent());
      // MmacLayout only changes accumulator/C ownership. A/B ownership is
      // unchanged, so the same contiguous fragment width applies here.
      if (mma && sharedEncoding.getContiguousDim() == 1 - dot.getOpIdx())
        maxReadBytes = legacyContiguousReadBytes(elementBytes, dot.getKWidth(),
                                                 copyConfig.copyBytesPerLane);
    }
    // Match the common local-load broadcast elimination and vector packing,
    // using the existing shared-load callback rather than a target-wide hook.
    auto removeBroadcast = actionRemoveBroadcastedRegs(cvt);
    auto outVals = lowerLdStShared(
        loc, ctx, removeBroadcast.apply(cvt), {}, elemTy, smemObj.getBase(),
        physicalOffset, smemObj.getShmemOffset(loc, rewriter, srcTy),
        smemObj.getMaskSpanOffsets(srcTy), rewriter, targetInfo,
        /*maybeMaxVecElems=*/maxReadBytes / elementBytes, op);
    if (!removeBroadcast.isIdentity())
      outVals = broadcastAs(outVals, cvt);
    rewriter.replaceOp(
        op, packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy));
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::HCU {
void populateBufferLdsLocalLoadPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const AMD::TargetInfo &targetInfo,
                                        PatternBenefit benefit) {
  patterns.add<BufferLdsLocalLoadLowering>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::HCU
