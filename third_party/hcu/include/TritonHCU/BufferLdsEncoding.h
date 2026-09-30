// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_BUFFER_LDS_ENCODING_H
#define TRITON_HCU_BUFFER_LDS_ENCODING_H

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/BufferLdsConfig.h"
#include "TritonHCU/LdsReadModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::HCU {

inline bool isBufferLdsFloatType(Type type) {
  return type.isF16() || type.isBF16() ||
         isa<Float8E4M3FNType, Float8E5M2Type>(type);
}

// Physical layout is owned by the memdesc encoding, never an operation
// attribute.
inline gpu::SharedLinearEncodingAttr
bufferLdsLinearComponent(MLIRContext *ctx, const BufferLdsConfig &copyConfig,
                         unsigned elementBytes, unsigned contiguousDim = 1) {
  std::vector<std::vector<int32_t>> bases;
  for (unsigned bit = elementBytes; bit < copyConfig.tileBytes(); bit *= 2) {
    unsigned logical = copyConfig.logicalByte(bit) / elementBytes;
    bases.push_back({int32_t(logical / (copyConfig.rowBytes / elementBytes)),
                     int32_t(logical % (copyConfig.rowBytes / elementBytes))});
    if (contiguousDim == 0)
      std::swap(bases.back()[0], bases.back()[1]);
  }
  return gpu::SharedLinearEncodingAttr::get(
      ctx,
      LinearLayout({{StringAttr::get(ctx, "offset"), std::move(bases)},
                    {StringAttr::get(ctx, "block"), {}}},
                   standardOutDimNames(ctx, 2)),
      16);
}

inline gpu::HCUBufferLdsSharedEncodingAttr
bufferLdsSharedEncoding(MLIRContext *ctx, const BufferLdsConfig &copyConfig,
                        unsigned elementBytes, unsigned contiguousDim = 1) {
  return gpu::HCUBufferLdsSharedEncodingAttr::get(
      ctx,
      bufferLdsLinearComponent(ctx, copyConfig, elementBytes, contiguousDim),
      elementBytes, contiguousDim, copyConfig.rows, copyConfig.rowBytes,
      copyConfig.waves, copyConfig.copyBytesPerLane, copyConfig.rowsPerChunk,
      copyConfig.wrapCount, copyConfig.wrapStepBytes,
      copyConfig.wavesPerRowGroup, copyConfig.transferMajor);
}

inline gpu::LinearEncodingAttr
bufferLdsProducerEncoding(MLIRContext *ctx, const BufferLdsConfig &copyConfig,
                          unsigned elementBytes, unsigned contiguousDim = 1) {
  // Use the shared encoding's logical bases, partitioned by register/lane/warp.
  // Their composition recovers physical copy offsets in generic lowerLdSt;
  // interleaving needs neither a separate address path nor a block exemption.
  std::vector<std::vector<int32_t>> registers, lanes, waves;
  for (unsigned bit = elementBytes; bit < copyConfig.tileBytes(); bit *= 2) {
    unsigned logical = copyConfig.logicalByte(bit) / elementBytes;
    // Each wave may issue multiple transfers. Their high offset bits belong
    // to registers, not to fictitious additional waves.
    auto &basis =
        bit < copyConfig.copyBytesPerLane         ? registers
        : bit < copyConfig.bytesPerWaveTransfer() ? lanes
        : copyConfig.transferMajor
            ? bit < copyConfig.bytesPerWaveTransfer() * copyConfig.waves
                  ? waves
                  : registers
        : bit < copyConfig.totalBytesPerWave() ? registers
                                               : waves;
    basis.push_back({int32_t(logical / (copyConfig.rowBytes / elementBytes)),
                     int32_t(logical % (copyConfig.rowBytes / elementBytes))});
    if (contiguousDim == 0)
      std::swap(basis.back()[0], basis.back()[1]);
  }
  return gpu::LinearEncodingAttr::get(
      ctx,
      LinearLayout({{StringAttr::get(ctx, "register"), std::move(registers)},
                    {StringAttr::get(ctx, "lane"), std::move(lanes)},
                    {StringAttr::get(ctx, "warp"), std::move(waves)},
                    {StringAttr::get(ctx, "block"), {}}},
                   standardOutDimNames(ctx, 2)));
}

inline BufferLdsConfig
getBufferLdsConfig(gpu::HCUBufferLdsSharedEncodingAttr enc) {
  return {enc.getRows(),
          enc.getRowBytes(),
          enc.getWaves(),
          enc.getCopyBytesPerLane(),
          64,
          enc.getRowsPerChunk(),
          enc.getWrapCount(),
          enc.getWrapStepBytes(),
          0,
          0,
          0,
          enc.getWavesPerRowGroup(),
          enc.getTransferMajor()};
}

inline gpu::LinearEncodingAttr
getBufferLdsProducerEncoding(gpu::HCUBufferLdsSharedEncodingAttr enc) {
  return bufferLdsProducerEncoding(enc.getContext(), getBufferLdsConfig(enc),
                                   enc.getElementBytes(),
                                   enc.getContiguousDim());
}

inline bool matchesBufferLdsProducer(RankedTensorType srcTy,
                                     gpu::MemDescType dstTy) {
  auto enc = dyn_cast<gpu::HCUBufferLdsSharedEncodingAttr>(dstTy.getEncoding());
  auto pointer = dyn_cast<triton::PointerType>(srcTy.getElementType());
  return enc && srcTy.getRank() == 2 && dstTy.getRank() == 2 && pointer &&
         pointer.getPointeeType() == dstTy.getElementType() &&
         srcTy.getShape() == dstTy.getShape() &&
         srcTy.getEncoding() == getBufferLdsProducerEncoding(enc);
}

inline FailureOr<unsigned>
getValidatedBufferLdsCopyWidth(Operation *op, RankedTensorType srcTy,
                               gpu::MemDescType dstTy, unsigned availableVec,
                               bool hasOther, std::string_view arch) {
  if (hasOther) {
    op->emitError("wrapped buffer-to-LDS requires zero-fill masking");
    return failure();
  }
  if (!matchesBufferLdsProducer(srcTy, dstTy)) {
    op->emitError("wrapped buffer-to-LDS producer has incompatible type, shape "
                  "or layout");
    return failure();
  }
  auto enc = cast<gpu::HCUBufferLdsSharedEncodingAttr>(dstTy.getEncoding());
  auto profile = getBufferLdsTargetProfile(arch);
  if (!profile ||
      !isBufferLdsConfigEncodable(getBufferLdsConfig(enc), *profile)) {
    op->emitError("buffer-to-LDS is not encodable on this target");
    return failure();
  }
  unsigned copyVec = enc.getCopyBytesPerLane() / enc.getElementBytes();
  if (availableVec < copyVec) {
    op->emitError("buffer-to-LDS vector/mask granularity does not match the "
                  "configuration's copy bytes per lane");
    return failure();
  }
  return copyVec;
}

inline bool canUseBufferLdsMatrixRead(gpu::MemDescType srcTy,
                                      gpu::DotOperandEncodingAttr consumer) {
  auto enc = dyn_cast<gpu::HCUBufferLdsSharedEncodingAttr>(srcTy.getEncoding());
  if (!enc || !consumer || srcTy.getRank() != 2)
    return false;
  auto mma = dyn_cast<gpu::AMDMfmaEncodingAttr>(consumer.getParent());
  unsigned nonKDim = consumer.getOpIdx();
  return mma && enc.getContiguousDim() == nonKDim &&
         selectLegacyMatrixRead(enc.getElementBytes() * 8, 16,
                                consumer.getKWidth())
             .has_value() &&
         srcTy.getShape()[nonKDim] % 32 == 0 &&
         mma.getTilesPerWarp()[nonKDim] == 2 &&
         mma.getTilesPerWarp()[1 - nonKDim] == 1;
}

// Pure query: no contract reconstruction, mutation, or validation side effects.
inline bool hasBufferLdsEncoding(ModuleOp module) {
  auto isBufferLdsMemDesc = [](Type type) {
    auto memdesc = dyn_cast<gpu::MemDescType>(type);
    return memdesc &&
           isa<gpu::HCUBufferLdsSharedEncodingAttr>(memdesc.getEncoding());
  };
  return module
      .walk([&](Operation *op) -> WalkResult {
        return llvm::any_of(op->getOperandTypes(), isBufferLdsMemDesc) ||
                       llvm::any_of(op->getResultTypes(), isBufferLdsMemDesc)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

// MLIR adapter implemented in Transforms/BufferLdsEncoding.cpp.
gpu::HCUBufferLdsSharedEncodingAttr selectBufferLdsOperandEncoding(
    Value value, RankedTensorType type, gpu::DotOperandEncodingAttr consumer,
    std::optional<gpu::DotOperandEncodingAttr> compatible, Type instructionType,
    unsigned waves, llvm::StringRef arch);
inline Value materializeBufferLdsOperand(
    PatternRewriter &rewriter, Location loc, Value value,
    gpu::DotOperandEncodingAttr consumer,
    gpu::HCUBufferLdsSharedEncodingAttr sharedEncoding) {
  auto type = cast<RankedTensorType>(value.getType());
  auto sharedTy = gpu::MemDescType::get(
      type.getShape(), type.getElementType(), sharedEncoding,
      gpu::SharedMemorySpaceAttr::get(value.getContext()),
      /*mutableMemory=*/true);
  auto alloc = gpu::LocalAllocOp::create(rewriter, loc, sharedTy, value);
  auto dotTy = type.cloneWithEncoding(consumer);
  return gpu::LocalLoadOp::create(rewriter, loc, dotTy, alloc);
}
} // namespace mlir::triton::HCU
#endif
