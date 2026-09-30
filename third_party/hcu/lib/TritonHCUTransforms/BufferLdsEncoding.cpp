// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "TritonHCU/BufferLdsEncoding.h"
#include "TritonHCU/BufferLdsConfigSelection.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"

namespace mlir::triton::HCU {
static gpu::HCUBufferLdsSharedEncodingAttr buildBufferLdsOperandEncoding(
    RankedTensorType type, gpu::DotOperandEncodingAttr consumer, unsigned waves,
    unsigned copyBytes = 0, unsigned contiguousDim = 1,
    unsigned maxCopyBytes = 16, const LdsGeometry &lds = {},
    const BufferLdsWrapCapabilities &wrap = {}, bool matrixRead = true,
    std::string_view arch = "gfx936") {
  if (type.getRank() != 2 || !isBufferLdsFloatType(type.getElementType()) ||
      !consumer)
    return {};
  auto mfma = dyn_cast<gpu::AMDMfmaEncodingAttr>(consumer.getParent());
  // MmacLayout describes the accumulator/C distribution. The A/B dot operand
  // layout is derived from instruction shape, opIdx, kWidth and warp tiling,
  // so BufferLds selection must not reject otherwise identical operands when
  // a target chooses a different C layout.
  if (!mfma || mfma.getIsTransposed())
    return {};
  unsigned bits = type.getElementType().getIntOrFloatBitWidth();
  if (bits % 8 || !bits)
    return {};
  auto instr =
      mfma.getInstrShapeForOperand(consumer.getKWidth(), consumer.getOpIdx());
  if (instr.size() != 2)
    return {};
  if (contiguousDim > 1)
    return {};
  unsigned kDim = consumer.getOpIdx() == 0 ? 1 : 0;
  bool kContiguous = contiguousDim == kDim;
  auto copyConfig = selectMmacBufferLdsConfig(
      type.getShape()[1 - contiguousDim], type.getShape()[contiguousDim],
      bits / 8, waves, copyBytes, kContiguous, instr[1 - kDim],
      consumer.getKWidth(), lds,
      matrixRead && mfma.getTilesPerWarp()[1 - kDim] == 2 &&
          mfma.getTilesPerWarp()[kDim] == 1,
      maxCopyBytes, wrap, arch);
  if (!copyConfig)
    return {};
  auto *ctx = type.getContext();
  return bufferLdsSharedEncoding(ctx, *copyConfig, bits / 8, contiguousDim);
}

gpu::HCUBufferLdsSharedEncodingAttr selectBufferLdsOperandEncoding(
    Value value, RankedTensorType type, gpu::DotOperandEncodingAttr consumer,
    std::optional<gpu::DotOperandEncodingAttr> compatible, Type instructionType,
    unsigned waves, llvm::StringRef arch) {
  // gfx936 emulates FP8 dot with FP16 MMAC. Copy the stored bytes and keep
  // the existing FP8 -> FP16 conversion after the shared-memory read.
  bool fp8Emulation =
      isa<Float8E4M3FNType, Float8E5M2Type>(type.getElementType()) &&
      instructionType.isF16();
  if (compatible || (type.getElementType() != instructionType && !fp8Emulation))
    return {};
  while (auto convert = value.getDefiningOp<gpu::ConvertLayoutOp>())
    value = convert.getSrc();
  auto load = value.getDefiningOp<triton::LoadOp>();
  if (!load || load.getIsVolatile() || load.getOther() || !load->hasOneUse())
    return {};
  auto loadType = dyn_cast<RankedTensorType>(value.getType());
  if (!loadType || loadType.getRank() != 2 ||
      !isa<gpu::DistributedEncodingTrait>(loadType.getEncoding()))
    return {};
  auto order = gpu::getOrder(loadType);
  if (order.size() != 2)
    return {};
  // Order is a preference, not a pointer-contiguity proof.
  triton::AMD::ModuleAxisInfoAnalysis axis(load->getParentOfType<ModuleOp>());
  unsigned vec = axis.getAlignment(load.getPtr());
  if (load.getMask())
    vec = std::min(vec, axis.getMaskAlignment(load.getMask()));
  unsigned maxBytes = std::min(16u, vec * (type.getElementTypeBitWidth() / 8));
  auto profile =
      getBufferLdsTargetProfile(std::string_view(arch.data(), arch.size()));
  if (!profile)
    return {};
  return buildBufferLdsOperandEncoding(type, consumer, waves, 0, order[0],
                                       maxBytes, profile->lds, profile->wrap,
                                       /*matrixRead=*/true,
                                       std::string_view(arch.data(), arch.size()));
}
} // namespace mlir::triton::HCU
