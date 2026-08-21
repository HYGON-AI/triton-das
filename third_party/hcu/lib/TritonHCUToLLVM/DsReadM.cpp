// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonHCU/DsReadMLayout.h"
#include "TritonHCU/MlsGroup.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::HCU;

namespace {

enum class DsReadMKind { B16, B8 };

struct DsReadMPlan {
  DsReadMKind kind;
  int64_t bk;
  int64_t bn;
  int64_t ldsBk;
  int64_t ldsBn;
  int64_t maxPhase;
  bool panelMajor;
  unsigned warpsN;
  unsigned nPanels;
  unsigned kTiles; // b16: BK/16; b8: BK/32
};

std::optional<DsReadMLayoutSpec>
getDsReadMLayoutSpecForTypes(Type srcElemTy, Type dstElemTy) {
  if (!srcElemTy.isIntOrFloat() || !dstElemTy.isIntOrFloat())
    return std::nullopt;

  unsigned srcBits = srcElemTy.getIntOrFloatBitWidth();
  unsigned dstBits = dstElemTy.getIntOrFloatBitWidth();
  if (srcBits != dstBits)
    return std::nullopt;
  return mlir::triton::HCU::getDsReadMLayoutSpec(srcBits);
}

bool hasExpectedSharedLayout(SwizzledSharedEncodingAttr sharedEnc,
                             const DsReadMLayoutSpec &spec, int64_t bn) {
  if (!sharedEnc || sharedEnc.getOrder().size() != 2 ||
      sharedEnc.getOrder()[0] != 1 || sharedEnc.getOrder()[1] != 0)
    return false;

  return sharedEnc.getVec() == spec.vec &&
         sharedEnc.getPerPhase() == spec.perPhase &&
         sharedEnc.getMaxPhase() == getDsReadMFlatMaxPhase(spec, bn);
}

struct DsReadMPanelViewPlan {
  int64_t ldsBk;
  int64_t ldsBn;
};

// Prove from the memdesc type that a local_load view is a translation within
// one panel-major address space. allocShape identifies the complete physical
// panel allocation, while shape identifies the selected logical view. Keeping
// the full N dimension and shrinking only K by whole ds_read_m tiles preserves
// every N32 panel; the affine K origin is already folded into the converted
// shared-memory base. Slicing N would change the panel address formula.
std::optional<DsReadMPanelViewPlan>
buildDsReadMPanelViewPlan(MemDescType viewTy,
                          const DsReadMLayoutSpec &spec) {
  auto panelEncoding =
      dyn_cast<SharedLinearEncodingAttr>(viewTy.getEncoding());
  ArrayRef<int64_t> viewShape = viewTy.getShape();
  ArrayRef<int64_t> allocShape = viewTy.getAllocShape();
  if (!panelEncoding || viewShape.size() != 2 || allocShape.size() < 2)
    return std::nullopt;

  ArrayRef<int64_t> panelShape = allocShape.take_back(2);
  if (!isDsReadMPanelEncoding(panelEncoding, panelShape[0], panelShape[1],
                              spec.elementBitWidth) ||
      viewShape[1] != panelShape[1] || viewShape[0] > panelShape[0] ||
      viewShape[0] % spec.kPerTile != 0)
    return std::nullopt;

  return DsReadMPanelViewPlan{panelShape[0], 32};
}

std::optional<DsReadMPlan> canUseDsReadM(LocalLoadOp op) {
  // Default-on: unset => enabled; ENABLE=0/false/off disables.
  // (getBoolEnv treats unset as false, so do not use it for this flag.)
  if (!isDsReadMEnabled())
    return std::nullopt;

  auto srcTy = cast<MemDescType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2 || !dotEnc ||
      dotEnc.getOpIdx() != 1)
    return std::nullopt;

  auto spec = getDsReadMLayoutSpecForTypes(srcTy.getElementType(),
                                          dstTy.getElementType());
  if (!spec)
    return std::nullopt;

  // b16: ds_read_m32x16 covers K=16; b8: ds_read_m32x32 covers K=32.
  int64_t bk = dstTy.getShape()[0], bn = dstTy.getShape()[1];
  auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding());
  std::optional<DsReadMPanelViewPlan> panelViewPlan =
      buildDsReadMPanelViewPlan(srcTy, *spec);
  bool panelMajor = panelViewPlan.has_value();
  if (!isDsReadMTileShape(bk, bn, *spec) ||
      (!panelMajor && !hasExpectedSharedLayout(sharedEnc, *spec, bn)))
    return std::nullopt;

  // Generic affine shared views remain unsupported.  The only affine panel
  // view accepted here was fully proved by buildDsReadMPanelViewPlan above.
  if (LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(srcTy) &&
      !panelMajor)
    return std::nullopt;

  auto mfmaEnc = cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
  auto instrShape = mfmaEnc.getInstrShape();
  auto warpsPerCTA = mfmaEnc.getWarpsPerCTA();
  auto tilesPerWarp = mfmaEnc.getTilesPerWarp();
  // 16x16xK MFMA, tilesPerWarp={1,2}; each N-warp owns >=1 full 32-col panel.
  // AccelerateHCUMatmul redistributes warpsPerCTA so warpsN <= BN/32.
  unsigned warpsN = warpsPerCTA.size() == 2 ? warpsPerCTA[1] : 0;
  if (instrShape.size() != 3 || instrShape[0] != 16 || instrShape[1] != 16 ||
      instrShape[2] != spec->kPerTile || dotEnc.getKWidth() != spec->kWidth ||
      tilesPerWarp.size() != 2 || tilesPerWarp[0] != 1 ||
      tilesPerWarp[1] != 2 || warpsN == 0 || bn % (warpsN * 32))
    return std::nullopt;

  return DsReadMPlan{spec->elementBitWidth == 8 ? DsReadMKind::B8
                                                : DsReadMKind::B16,
                     bk,
                     bn,
                     panelMajor ? panelViewPlan->ldsBk : bk,
                     panelMajor ? panelViewPlan->ldsBn : bn,
                     panelMajor ? spec->panelMaxPhase
                                : getDsReadMFlatMaxPhase(*spec, bn),
                     panelMajor,
                     warpsN,
                     static_cast<unsigned>(bn / (warpsN * 32)),
                     static_cast<unsigned>(bk / spec->kPerTile)};
}

Value createDsReadM32x16B16(Location loc, RewriterBase &rewriter,
                            Type resultType, Value addr, Value offset) {
  auto call = LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                              "llvm.hcu.ds.read.m32x16.b16",
                                              {resultType}, {addr, offset});
  return call.getResult(0);
}

Value createDsReadM32x32B8(Location loc, RewriterBase &rewriter,
                           Type resultType, Value addr, Value offset) {
  auto call = LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                              "llvm.hcu.ds.read.m32x32.b8",
                                              {resultType}, {addr, offset});
  return call.getResult(0);
}

SmallVector<Value> unpackI32Regs(Location loc, Type elemTy,
                                 ArrayRef<Value> regs, TritonLLVMOpBuilder &b) {
  SmallVector<Value> elems;
  unsigned elemsPerReg = 32 / elemTy.getIntOrFloatBitWidth();
  Type vecTy = vec_ty(elemTy, elemsPerReg);
  for (Value reg : regs) {
    Value regVec = b.bitcast(reg, vecTy);
    for (unsigned i = 0; i < elemsPerReg; ++i)
      elems.push_back(b.extract_element(elemTy, regVec, b.i32_val(i)));
  }
  return elems;
}

// Emit ds_read_m32x16_b16 for each (nPanel, kTile).
//
// Both paths return the same logical N32 x K16 fragment. Panel-major LDS uses
// one physical [BK,32] panel per N-warp, while the flat path addresses the
// complete [BK,BN] tile. Their address formulas are documented at each path.
SmallVector<SmallVector<Value>>
emitB16Reads(Location loc, RewriterBase &rewriter, TritonLLVMOpBuilder &b,
             Type smemPtrTy, Type elemTy, Value smemBase, Value laneId,
             Value warpId, const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> elemsByKTile;
  Type vecTy = vec_ty(elemTy, 8);

  Value colInQuad = b.and_(laneId, b.i32_val(3)); // c = lane & 3
  if (plan.panelMajor) {
    // Panel-major physical addressing:
    //   k_addr = lane >> 2
    //   c       = lane & 3
    //   phase   = (k_addr >> 1) & 1
    //   vaddr_elem = warpN * BK * 32 + k_addr * 32 + ((c ^ phase) * 8)
    //
    // The VGPR address selects this warp's first physical [BK,32] panel.
    // kTile and nPanel select later K16 tiles and N32 panel groups through the
    // instruction immediate, avoiding repeated per-instruction VGPR address
    // arithmetic:
    //   offset_bytes =
    //       (kTile * 16 * 32 + nPanel * warpsN * BK * 32) * sizeof(b16)
    Value kAddr = b.lshr(laneId, b.i32_val(2));
    Value phase = b.and_(b.lshr(kAddr, b.i32_val(1)), b.i32_val(1));
    Value nGroup = b.xor_(colInQuad, phase);
    Value vaddrElem =
        b.add(b.add(b.mul(kAddr, b.i32_val(plan.ldsBn)),
                    b.mul(nGroup, b.i32_val(8))),
              b.mul(warpId, b.i32_val(plan.ldsBk * plan.ldsBn)));
    Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                              LLVM::GEPNoWrapFlags::inbounds);

    for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
      for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
        int64_t offsetBytes =
            (static_cast<int64_t>(kTile) * 16 * plan.ldsBn +
             static_cast<int64_t>(nPanel) * plan.warpsN * plan.ldsBk *
                 plan.ldsBn) *
            2;
        Value dsValue = createDsReadM32x16B16(
            loc, rewriter, vecTy, loadAddress, b.i32_val(offsetBytes));
        SmallVector<Value> elems;
        for (int i = 0; i < 8; ++i)
          elems.push_back(b.extract_element(elemTy, dsValue, b.i32_val(i)));
        elemsByKTile.push_back(elems);
      }
    }
    return elemsByKTile;
  }

  // Flat [BK,BN] addressing:
  //   k_addr = (lane >> 2) + kTile * 16
  //   c       = lane & 3
  //   panel   = nPanel * warpsN + warpN
  //   n0      = panel * 32 + c * 8
  //
  // For shared {vec=8, perPhase=2, maxPhase=2, order=[1,0]}:
  //   phase         = (k_addr >> 1) & 1
  //   swizzled_n    = ((n0 / 8) ^ phase) * 8
  //   vaddr_elem    = k_addr * BN + swizzled_n
  //
  // The full K/N displacement is carried by the VGPR address, so the
  // instruction immediate remains zero.
  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      Value kAddr =
          b.add(b.lshr(laneId, b.i32_val(2)), b.i32_val(kTile * 16));
      Value phase = b.and_(b.lshr(kAddr, b.i32_val(1)), b.i32_val(1));
      Value nPanelBase = b.mul(b.i32_val(nPanel * plan.warpsN), b.i32_val(4));
      Value nGroup = b.xor_(
          b.add(b.add(nPanelBase, b.mul(warpId, b.i32_val(4))), colInQuad),
          phase);
      Value vaddrElem =
          b.add(b.mul(kAddr, b.i32_val(plan.bn)), b.mul(nGroup, b.i32_val(8)));

      Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                                LLVM::GEPNoWrapFlags::inbounds);
      Value dsValue = createDsReadM32x16B16(loc, rewriter, vecTy, loadAddress,
                                            b.i32_val(0));

      SmallVector<Value> elems;
      for (int i = 0; i < 8; ++i)
        elems.push_back(b.extract_element(elemTy, dsValue, b.i32_val(i)));
      elemsByKTile.push_back(elems);
    }
  }

  return elemsByKTile;
}

// Emit ds_read_m32x32_b8 for each (nPanel, kTile).
//
// Both paths return the same logical N32 x K32 fragment. Panel-major LDS uses
// one physical [BK,32] panel per N-warp, while the flat path addresses the
// complete [BK,BN] tile. Hardware returns 4xi32, which form two 8-element
// MMAC-B K fragments. Address formulas are documented at each path.
SmallVector<SmallVector<Value>>
emitB8Reads(Location loc, RewriterBase &rewriter, TritonLLVMOpBuilder &b,
            Type smemPtrTy, Type elemTy, Value smemBase, Value laneId,
            Value warpId, const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> elemsByPanel;
  Value colInPair = b.and_(laneId, b.i32_val(1)); // c = lane & 1
  Type vecTy = vec_ty(rewriter.getI32Type(), 4);
  if (plan.panelMajor) {
    // Panel-major physical addressing:
    //   k_addr = lane >> 1
    //   c       = lane & 1
    //   phase   = k_addr & 1
    //   panel   = nPanel * warpsN + warpN
    //   physical_k = k_addr ^ (panel & 3)
    //   vaddr_elem = warpN * BK * 32 + physical_k * 32
    //              + ((c ^ phase) * 16)
    //
    // maxPhase=2 keeps the N XOR within the two 16-byte groups of this warp's
    // physical N32 panel. The panel-dependent K skew matches the store layout
    // and rotates adjacent panels onto disjoint banks. kTile and nPanel select
    // later K32 tiles and N32 panel groups through the byte-unit immediate:
    //   offset_bytes = kTile * 32 * 32 + nPanel * warpsN * BK * 32
    Value kAddr = b.lshr(laneId, b.i32_val(1));
    Value phase = b.and_(kAddr, b.i32_val(1));
    Value nGroup = b.xor_(colInPair, phase);

    for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
      // The panel layout XORs panel-id bits [1:0] into the physical K row to
      // make N-contiguous vector stores bank-disjoint across four panels.
      // Undo that skew in the matrix-read address.  nPanel selects a later
      // group of warpsN panels through the immediate below, so include it when
      // forming the full logical panel id.
      Value panelId = b.add(warpId, b.i32_val(nPanel * plan.warpsN));
      Value panelK = b.and_(panelId, b.i32_val(3));
      Value physicalK = b.xor_(kAddr, panelK);
      Value vaddrElem =
          b.add(b.add(b.mul(physicalK, b.i32_val(plan.ldsBn)),
                      b.mul(nGroup, b.i32_val(16))),
                b.mul(warpId, b.i32_val(plan.ldsBk * plan.ldsBn)));
      Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                                LLVM::GEPNoWrapFlags::inbounds);
      for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
        int64_t offsetBytes =
            static_cast<int64_t>(kTile) * 32 * plan.ldsBn +
            static_cast<int64_t>(nPanel) * plan.warpsN * plan.ldsBk *
                plan.ldsBn;
        Value dsValue = createDsReadM32x32B8(
            loc, rewriter, vecTy, loadAddress, b.i32_val(offsetBytes));

        SmallVector<Value> regs;
        for (int i = 0; i < 4; ++i)
          regs.push_back(
              b.extract_element(rewriter.getI32Type(), dsValue, b.i32_val(i)));
        ArrayRef<Value> regRef(regs);
        elemsByPanel.push_back(
            unpackI32Regs(loc, elemTy, regRef.take_front(2), b));
        elemsByPanel.push_back(
            unpackI32Regs(loc, elemTy, regRef.drop_front(2), b));
      }
    }
    return elemsByPanel;
  }

  // Flat [BK,BN] addressing:
  //   k_addr = (lane >> 1) + kTile * 32
  //   c       = lane & 1
  //   panel   = nPanel * warpsN + warpN
  //   n0      = panel * 32 + c * 16
  //
  // For shared {vec=16, perPhase=1, maxPhase=min(4, BN/16), order=[1,0]}:
  //   phase         = k_addr % maxPhase
  //   swizzled_n    = ((n0 / 16) ^ phase) * 16
  //   vaddr_elem    = k_addr * BN + swizzled_n
  //
  // The full K/N displacement is carried by the VGPR address, so the
  // instruction immediate remains zero.
  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      Value kAddr =
          b.add(b.lshr(laneId, b.i32_val(1)), b.i32_val(kTile * 32));
      Value phase = b.urem(kAddr, b.i32_val(plan.maxPhase));
      Value nPanelBase = b.mul(b.i32_val(nPanel * plan.warpsN), b.i32_val(2));
      Value nGroup = b.xor_(
          b.add(b.add(nPanelBase, b.mul(warpId, b.i32_val(2))), colInPair),
          phase);
      Value vaddrElem = b.add(b.mul(kAddr, b.i32_val(plan.bn)),
                              b.mul(nGroup, b.i32_val(16)));

      Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                                LLVM::GEPNoWrapFlags::inbounds);
      Value dsValue = createDsReadM32x32B8(loc, rewriter, vecTy, loadAddress,
                                           b.i32_val(0));

      SmallVector<Value> regs;
      for (int i = 0; i < 4; ++i)
        regs.push_back(
            b.extract_element(rewriter.getI32Type(), dsValue, b.i32_val(i)));

      // Two 8-element MMAC-B K fragments: regs[0:2], regs[2:4].
      ArrayRef<Value> regRef(regs);
      elemsByPanel.push_back(
          unpackI32Regs(loc, elemTy, regRef.take_front(2), b));
      elemsByPanel.push_back(
          unpackI32Regs(loc, elemTy, regRef.drop_front(2), b));
    }
  }
  return elemsByPanel;
}

SmallVector<Value>
packDsReadMFragments(const SmallVector<SmallVector<Value>> &elemsByKTile,
                     const DsReadMPlan &plan) {
  SmallVector<Value> outVals;
  outVals.reserve(static_cast<size_t>(plan.bk) * plan.nPanels);

  if (plan.kind == DsReadMKind::B16) {
    // Triton's B dot operand struct is ordered by N-rep then K-rep.
    for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel)
      for (int nRep = 0; nRep < 2; ++nRep)
        for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile)
          for (int k = 0; k < 4; ++k)
            outVals.push_back(
                elemsByKTile[nPanel * plan.kTiles + kTile][nRep * 4 + k]);
    return outVals;
  }

  // B8: emitB8Reads stores (nPanel, kTile) → [nRep0, nRep1] (kTile-major).
  // Triton/MMAC B layout is nRep-major then kTile (for n / for k). Reindex;
  // do not flatten emit order — that mismatches when kTiles > 1.
  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel)
    for (int nRep = 0; nRep < 2; ++nRep)
      for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
        const SmallVector<Value> &frag =
            elemsByKTile[nPanel * plan.kTiles * 2 + kTile * 2 + nRep];
        for (int k = 0; k < 8; ++k)
          outVals.push_back(frag[k]);
      }
  return outVals;
}

struct HCUDsReadMConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto plan = canUseDsReadM(op);
    if (!plan)
      return failure();

    Location loc = op.getLoc();
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto typeConverter = getTypeConverter();
    Type elemTy = typeConverter->convertType(srcTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         elemTy, rewriter);
    // canUseDsReadM accepts an affine view only after proving that it is a
    // panel-preserving K-tile translation. Fold that translation into the LDS
    // base; otherwise K0 and K1 subslices would issue from the same address.
    Value smemBase =
        LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(srcTy)
            ? smemObj.getShmemAffineBase(loc, rewriter, srcTy)
            : smemObj.getBase();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto mfmaEnc = cast<AMDMfmaEncodingAttr>(
        cast<DotOperandEncodingAttr>(dstTy.getEncoding()).getParent());
    SmallVector<Value> warpIds =
        delinearize(rewriter, loc, warpId, mfmaEnc.getWarpsPerCTA(),
                    triton::gpu::getMatrixOrder(/*rank=*/2, /*rowMajor=*/true));
    Value warpN = warpIds[1];
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    SmallVector<SmallVector<Value>> elemsByKTile =
        plan->kind == DsReadMKind::B16
            ? emitB16Reads(loc, rewriter, b, smemPtrTy, elemTy,
                           smemBase, laneId, warpN, *plan)
            : emitB8Reads(loc, rewriter, b, smemPtrTy, elemTy,
                          smemBase, laneId, warpN, *plan);

    SmallVector<Value> outVals = packDsReadMFragments(elemsByKTile, *plan);
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace mlir::triton::HCU {
void populateDsReadMToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit) {
  patterns.add<HCUDsReadMConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::HCU
