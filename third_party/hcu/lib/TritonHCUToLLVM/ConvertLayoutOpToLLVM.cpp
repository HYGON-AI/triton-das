// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Closed-form warp-shuffle lowering for the HCU LEGACY MMAC accumulator ->
// dot-A conversion of a chained dot. Registered with a higher benefit than the
// common ConvertLayoutOp lowering so it is tried first; every other conversion
// falls through to the common pattern.

#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// `HalfQuad` is the 16-bit packed-register variant of `Quad`.
enum class HCULegacyQuadTransposeKind {
  None,
  Quad,
  HalfQuad,
};

// Selects the closed-form lowering, or None if the conversion does not have
// the exact bit structure the shuffle sequences below implement.
HCULegacyQuadTransposeKind getQuadTransposeKind(RankedTensorType srcTy,
                                                RankedTensorType dstTy) {
  if (!isHCULegacyMmacToDotA(srcTy, dstTy))
    return HCULegacyQuadTransposeKind::None;

  auto *ctx = srcTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");

  // The lowering works on the unpacked registers directly, so it does not
  // handle register broadcasting or padding.
  auto srcLayout = toLinearLayout(srcTy);
  auto dstLayout = toLinearLayout(dstTy);
  if (!actionRemoveBroadcastedRegs(srcLayout).isIdentity() ||
      !actionRemoveBroadcastedRegs(dstLayout).isIdentity() ||
      srcLayout.getInDimSize(kRegister) != dstLayout.getInDimSize(kRegister))
    return HCULegacyQuadTransposeKind::None;

  // `Quad` moves whole (unpacked) registers; `HalfQuad` moves 16-bit halves
  // of packed registers.
  int bitwidth = getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
  auto conversion = getWarpLayoutConvertDecomposition(srcTy, dstTy, bitwidth);
  if (conversion.mixedTranspositions.size() != 2 ||
      (conversion.nPack != 0 && !(bitwidth == 16 && conversion.nPack == 1)))
    return HCULegacyQuadTransposeKind::None;

  if (!squareSublayoutIsIdentity(conversion.pReg, kRegister) ||
      !squareSublayoutIsIdentity(conversion.pLane, kLane) ||
      conversion.pReg.getInDimSize(kRegister) % 4 != 0)
    return HCULegacyQuadTransposeKind::None;

  int lane4Reg = -1;
  int lane5Reg = -1;
  int laneBits = 0;
  int regBits = 0;
  for (const auto &t : conversion.mixedTranspositions) {
    int regBit = t.transposition.first;
    int laneBit = t.transposition.second;
    if ((laneBit != 4 && laneBit != 5) || regBit < conversion.nPack)
      return HCULegacyQuadTransposeKind::None;
    if (laneBit == 4)
      lane4Reg = regBit;
    else
      lane5Reg = regBit;
    laneBits |= 1 << laneBit;
    regBits |= 1 << regBit;
  }

  if (laneBits != ((1 << 4) | (1 << 5)) ||
      regBits != ((1 << conversion.nPack) | (1 << (conversion.nPack + 1))))
    return HCULegacyQuadTransposeKind::None;

  if (conversion.nPack == 0)
    return lane4Reg == 0 && lane5Reg == 1 ? HCULegacyQuadTransposeKind::Quad
                                          : HCULegacyQuadTransposeKind::None;

  return lane4Reg == conversion.nPack + 1 && lane5Reg == conversion.nPack
             ? HCULegacyQuadTransposeKind::HalfQuad
             : HCULegacyQuadTransposeKind::None;
}

struct HCULegacyQuadTransposeConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  HCULegacyQuadTransposeConversion(LLVMTypeConverter &typeConverter,
                                   const AMD::TargetInfo &targetInfo,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto kind = getQuadTransposeKind(srcTy, dstTy);
    if (kind == HCULegacyQuadTransposeKind::None)
      return failure();

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    bool isHalf = kind == HCULegacyQuadTransposeKind::HalfQuad;

    // The classifier guarantees an identity register permutation and no
    // register broadcasting, so the unpacked values are used as-is.
    SmallVector<Value> vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (isHalf) {
      SmallVector<Value> packed;
      packed.reserve(vals.size() / 2);
      for (size_t i = 0; i < vals.size(); i += 2)
        packed.push_back(b.bitcast(
            packLLVector(loc, ArrayRef<Value>(vals).slice(i, 2), rewriter),
            i32_ty));
      vals = std::move(packed);
    }

    Value laneId = getLaneId(rewriter, loc);
    Value l4Off = getLaneBitIsOff(b, laneId, /*laneBit=*/4);
    Value l5Off = getLaneBitIsOff(b, laneId, /*laneBit=*/5);
    for (size_t base = 0; base < vals.size(); base += 4) {
      if (isHalf) {
        // In packed space the low register bit is the 16-bit halfword:
        // swap packed register bit 0 with l5, then move the halfword across
        // l4 with one v_perm per register.
        swapRegisterGroupWithLaneBit(loc, rewriter, vals, base,
                                     /*registerMask=*/1, /*laneBit=*/5, l5Off);
        repackHalvesWithLaneBit(loc, rewriter, vals, base, /*laneBit=*/4,
                                l4Off);
      } else {
        // final[lane a][reg b] = initial[lane b][reg a] for a, b in 0..3,
        // where a is the (l5, l4) lane pair inside a 16-lane quad.
        swapRegisterGroupWithLaneBit(loc, rewriter, vals, base,
                                     /*registerMask=*/1, /*laneBit=*/4, l4Off);
        swapRegisterGroupWithLaneBit(loc, rewriter, vals, base,
                                     /*registerMask=*/2, /*laneBit=*/5, l5Off);
      }
    }

    if (isHalf) {
      SmallVector<Value> unpacked;
      unpacked.reserve(vals.size() * 2);
      for (Value v : vals) {
        auto halves =
            unpackLLVector(loc, b.bitcast(v, vec_ty(elemTy, 2)), rewriter);
        unpacked.append(halves.begin(), halves.end());
      }
      vals = std::move(unpacked);
    }

    Value result =
        packLLElements(loc, getTypeConverter(), vals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Value getLaneBitIsOff(TritonLLVMOpBuilder &b, Value laneId,
                               int laneBit) {
    return b.icmp_eq(b.and_(laneId, b.i32_val(1 << laneBit)), b.i32_val(0));
  }

  // Swaps register `i` with register `i ^ registerMask` of the partner lane
  // across `laneBit`, within the 4-register group starting at `base`.
  void swapRegisterGroupWithLaneBit(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    SmallVectorImpl<Value> &vals, size_t base,
                                    int registerMask, int laneBit,
                                    Value laneBitIsOff) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value, 4> shuffled;
    for (int i = 0; i < 4; ++i)
      shuffled.push_back(
          targetInfo.shuffleXor(rewriter, loc, vals[base + i], 1 << laneBit));
    for (int i = 0; i < 4; ++i) {
      int partner = i ^ registerMask;
      vals[base + i] =
          i < partner
              ? b.select(laneBitIsOff, vals[base + i], shuffled[partner])
              : b.select(laneBitIsOff, shuffled[partner], vals[base + i]);
    }
  }

  // Moves the 16-bit halfword across `laneBit`: out.half(0) comes from the
  // lane-bit-0 side and out.half(1) from the lane-bit-1 side, each taking the
  // half selected by our own lane bit (self = bytes 0-3, partner = bytes 4-7).
  void repackHalvesWithLaneBit(Location loc,
                               ConversionPatternRewriter &rewriter,
                               SmallVectorImpl<Value> &vals, size_t base,
                               int laneBit, Value laneBitIsOff) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value, 4> shuffled;
    for (int i = 0; i < 4; ++i)
      shuffled.push_back(
          targetInfo.shuffleXor(rewriter, loc, vals[base + i], 1 << laneBit));
    Value sel = b.select(laneBitIsOff, b.i32_val(0x5410), b.i32_val(0x3276));
    for (int i = 0; i < 4; ++i)
      vals[base + i] =
          targetInfo.permute(rewriter, loc, vals[base + i], shuffled[i], sel);
  }

  const AMD::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::HCU {
void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const AMD::TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit) {
  patterns.add<HCULegacyQuadTransposeConversion>(typeConverter, targetInfo,
                                                 benefit);
}
} // namespace mlir::triton::HCU
