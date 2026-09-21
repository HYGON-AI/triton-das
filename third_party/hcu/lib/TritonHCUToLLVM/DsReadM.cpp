// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "PatternTritonGPUOpToLLVM.h"
#include "BufferLdsAddress.h"
#include "TritonHCU/DsReadMLayout.h"
#include "TritonHCU/LdsReadAccessPatterns.h"
#include "TritonHCU/MlsGroup.h"
#include "TritonHCU/BufferLdsEncoding.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
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

enum class DsReadMKind { B32, B16, B8 };

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
  unsigned kTiles; // b32: BK/8; b16: BK/16; b8: BK/32
};

DsReadMKind dsReadMKind(const DsReadMLayoutSpec &spec) {
  switch (spec.elementBitWidth) {
  case 32:
    return DsReadMKind::B32;
  case 8:
    return DsReadMKind::B8;
  default:
    return DsReadMKind::B16;
  }
}

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
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2 || !dotEnc)
    return std::nullopt;
  auto sharedEncoding =
      dyn_cast<HCUBufferLdsSharedEncodingAttr>(srcTy.getEncoding());
  // Existing flat/panel layouts remain B-only. BufferLds encodings use
  // the common (non-K, K) legacy fragment for either operand.
  if (dotEnc.getOpIdx() != 1 && !sharedEncoding)
    return std::nullopt;
  unsigned nonKDim = dotEnc.getOpIdx(), kDim = 1 - nonKDim;

  auto spec = getDsReadMLayoutSpecForTypes(srcTy.getElementType(),
                                          dstTy.getElementType());
  if (!spec)
    return std::nullopt;

  // b32: ds_read_m32x8 covers K=8; b16: ds_read_m32x16 covers K=16;
  // b8: ds_read_m32x32 covers K=32.
  int64_t bk = dstTy.getShape()[kDim], bn = dstTy.getShape()[nonKDim];
  auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding());
  std::optional<DsReadMPanelViewPlan> panelViewPlan =
      buildDsReadMPanelViewPlan(srcTy, *spec);
  bool panelMajor = panelViewPlan.has_value();
  bool interleavedBufferLds = false;
  if (sharedEncoding) {
    // BufferLds matrix reads address the full [BK, BN] tile from the memdesc
    // base: it carries memdesc_index strides but not subslice offsets, so
    // only full-tile views are addressable here.
    if (srcTy.getAllocShape().take_back(srcTy.getRank()) != srcTy.getShape())
      return std::nullopt;
    if (!canUseBufferLdsMatrixRead(srcTy, dotEnc))
      return std::nullopt;
    interleavedBufferLds = true;
    panelMajor = false;
  }
  if (!isDsReadMTileShape(bk, bn, *spec) ||
      (!panelMajor && !interleavedBufferLds &&
       !hasExpectedSharedLayout(sharedEnc, *spec, bn)))
    return std::nullopt;

  // Generic affine shared views remain unsupported.  The only affine panel
  // view accepted here was fully proved by buildDsReadMPanelViewPlan above.
  if (LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(srcTy) &&
      !panelMajor && !interleavedBufferLds)
    return std::nullopt;

  auto mfmaEnc = cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
  auto instrShape = mfmaEnc.getInstrShape();
  auto warpsPerCTA = mfmaEnc.getWarpsPerCTA();
  auto tilesPerWarp = mfmaEnc.getTilesPerWarp();
  // 16x16xK MFMA, tilesPerWarp={1,2}; each N-warp owns >=1 full 32-col panel.
  // AccelerateHCUMatmul redistributes warpsPerCTA so warpsN <= BN/32.
  unsigned warpsN = warpsPerCTA.size() == 2 ? warpsPerCTA[nonKDim] : 0;
  if (instrShape.size() != 3 || instrShape[0] != 16 || instrShape[1] != 16 ||
      instrShape[2] != spec->kPerTile || dotEnc.getKWidth() != spec->kWidth ||
      tilesPerWarp.size() != 2 || tilesPerWarp[kDim] != 1 ||
      tilesPerWarp[nonKDim] != 2 || warpsN == 0 || bn % (warpsN * 32))
    return std::nullopt;

  return DsReadMPlan{dsReadMKind(*spec), bk, bn,
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

Value createDsReadM32x8B32(Location loc, RewriterBase &rewriter,
                           Type resultType, Value addr, Value offset) {
  auto call = LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                              "llvm.hcu.ds.read.m32x8.b32",
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

// Emit ds_read_m32x8.b32 for each (N32 panel, K8 tile) of the N-major flat
// [BK, BN] tile (lds[k * BN + n]). Lane t owns K row t/8 and N group t%8
// (4 consecutive f32); each N32 panel holds 8 N groups, so
//   vaddr_elem = (t / 8 + kTile * 8) * BN
//              + ((nPanel * warpsN + warpN) * 8 + t % 8) * 4
// b32 flat has maxPhase == 1, so the kTile/nPanel displacements are
// compile-time constants: the VGPR address is computed once and both go in
// the byte immediate (same as the panel path).
SmallVector<SmallVector<Value>>
emitB32Reads(Location loc, RewriterBase &rewriter, TritonLLVMOpBuilder &b,
             Type smemPtrTy, Type elemTy, Value smemBase, Value laneId,
             Value warpId, const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> elemsByKTile;
  elemsByKTile.reserve(static_cast<size_t>(plan.nPanels) * plan.kTiles);
  Type i32Ty = rewriter.getI32Type();
  Type vecTy = vec_ty(i32Ty, 4);

  //   vaddr_elem = (t / 8) * BN + warpN * 32 + (t % 8) * 4
  Value vaddrElem =
      b.add(b.mul(b.lshr(laneId, b.i32_val(3)), b.i32_val(plan.bn)),
            b.add(b.mul(warpId, b.i32_val(32)),
                  b.mul(b.and_(laneId, b.i32_val(7)), b.i32_val(4))));
  Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                            LLVM::GEPNoWrapFlags::inbounds);

  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      //   offset_bytes = (kTile * 8 * BN + nPanel * warpsN * 32) * sizeof(f32)
      int64_t offsetBytes = (static_cast<int64_t>(kTile) * 8 * plan.bn +
                             static_cast<int64_t>(nPanel) * plan.warpsN * 32) *
                            4;
      Value dsValue = createDsReadM32x8B32(loc, rewriter, vecTy, loadAddress,
                                           b.i32_val(offsetBytes));
      SmallVector<Value> elems;
      elems.reserve(4);
      for (int i = 0; i < 4; ++i)
        elems.push_back(b.bitcast(
            b.extract_element(i32Ty, dsValue, b.i32_val(i)), elemTy));
      elemsByKTile.push_back(std::move(elems));
    }
  }
  return elemsByKTile;
}

// BufferLds b16/b8 reads share a 16-byte source span per lane. The element
// width determines the source-lane grid and the returned fragment packing.
// The view must be a full [BK, BN] tile (see canUseDsReadM): smemBase is the
// only per-view translation, so no logical offsets are added here.
SmallVector<SmallVector<Value>>
emitBufferLdsMatrixReads(Location loc, RewriterBase &rewriter,
                 TritonLLVMOpBuilder &b, Type smemPtrTy, Type elemTy,
                 Value smemBase, Value laneId, Value warpNonK,
                 const BufferLdsConfig &copyConfig,
                 const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> fragments;
  bool b16 = plan.kind == DsReadMKind::B16;
  unsigned elementBytes = b16 ? 2 : 1;
  unsigned elementsPerLane = 16 / elementBytes;
  unsigned lanesPerRow = 32 / elementsPerLane;
  unsigned kPerTile = 64 / lanesPerRow;
  Type vecTy = b16 ? vec_ty(elemTy, 8) : vec_ty(rewriter.getI32Type(), 4);
  for (unsigned panel = 0; panel < plan.nPanels; ++panel)
    for (unsigned kt = 0; kt < plan.kTiles; ++kt) {
      Value row = b.add(b.udiv(laneId, b.i32_val(lanesPerRow)),
                        b.i32_val(kt * kPerTile));
      Value col = b.add(
          b.mul(b.add(warpNonK, b.i32_val(panel * plan.warpsN)), b.i32_val(32)),
          b.mul(b.urem(laneId, b.i32_val(lanesPerRow)),
                b.i32_val(elementsPerLane)));
      Value bytes =
          emitBufferLdsByteOffset(b, copyConfig, row, col, elementBytes);
      Value elementOffset = b.udiv(bytes, b.i32_val(elementBytes));
      Value addr = b.gep(smemPtrTy, elemTy, smemBase, elementOffset,
                         LLVM::GEPNoWrapFlags::inbounds);
      if (b16) {
        Value result =
            createDsReadM32x16B16(loc, rewriter, vecTy, addr, b.i32_val(0));
        SmallVector<Value> elems;
        for (unsigned i = 0; i < 8; ++i)
          elems.push_back(b.extract_element(elemTy, result, b.i32_val(i)));
        fragments.push_back(elems);
      } else {
        Value result =
            createDsReadM32x32B8(loc, rewriter, vecTy, addr, b.i32_val(0));
        SmallVector<Value> regs;
        for (unsigned i = 0; i < 4; ++i)
          regs.push_back(b.extract_element(rewriter.getI32Type(), result,
                                           b.i32_val(i)));
        ArrayRef<Value> regRef(regs);
        fragments.push_back(unpackI32Regs(loc, elemTy, regRef.take_front(2), b));
        fragments.push_back(unpackI32Regs(loc, elemTy, regRef.drop_front(2), b));
      }
    }
  return fragments;
}

// Emit ds_read_m32x16_b16 for each (nPanel, kTile).
//
// Both paths return the same logical N32 x K16 fragment. Panel-major LDS uses
// one physical [BK,32] panel per N-warp, while the flat path addresses the
// complete [BK,BN] tile. Address formulas are documented at each path.
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
  // Both displacements are compile-time constants (phase is kTile-
  // independent; nPanel enters n0/8 in multiples of 4, commuting with the
  // bit-0 XOR), so they go in the byte immediate, like the panel path.
  Value kAddr = b.lshr(laneId, b.i32_val(2));
  Value phase = b.and_(b.lshr(kAddr, b.i32_val(1)), b.i32_val(1));
  Value nGroup = b.xor_(b.add(b.mul(warpId, b.i32_val(4)), colInQuad), phase);
  Value vaddrElem =
      b.add(b.mul(kAddr, b.i32_val(plan.bn)), b.mul(nGroup, b.i32_val(8)));
  Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                            LLVM::GEPNoWrapFlags::inbounds);

  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      int64_t offsetBytes = (static_cast<int64_t>(kTile) * 16 * plan.bn +
                             static_cast<int64_t>(nPanel) * plan.warpsN * 32) *
                            2;
      Value dsValue = createDsReadM32x16B16(loc, rewriter, vecTy, loadAddress,
                                            b.i32_val(offsetBytes));

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
            Value warpNonK, const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> elemsByPanel;
  Value colInPair = b.and_(laneId, b.i32_val(1)); // c = lane & 1
  Type vecTy = vec_ty(rewriter.getI32Type(), 4);

  if (plan.panelMajor) {
    // Panel-major physical addressing:
    //   k_addr = lane >> 1
    //   c       = lane & 1
    //   phase   = k_addr & 1
    //   panel   = nPanel * warpsN + warpNonK
    //   physical_k = k_addr ^ (panel & 3)
    //   vaddr_elem = warpNonK * BK * 32 + physical_k * 32
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
      Value panelId = b.add(warpNonK, b.i32_val(nPanel * plan.warpsN));
      Value panelK = b.and_(panelId, b.i32_val(3));
      Value physicalK = b.xor_(kAddr, panelK);
      Value vaddrElem =
          b.add(b.add(b.mul(physicalK, b.i32_val(plan.ldsBn)),
                      b.mul(nGroup, b.i32_val(16))),
                b.mul(warpNonK, b.i32_val(plan.ldsBk * plan.ldsBn)));
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
  //   panel   = nPanel * warpsN + warpNonK
  //   n0      = panel * 32 + c * 16
  //
  // For shared {vec=16, perPhase=1, maxPhase=min(4, BN/16), order=[1,0]}:
  //   phase         = k_addr % maxPhase
  //   swizzled_n    = ((n0 / 16) ^ phase) * 16
  //   vaddr_elem    = k_addr * BN + swizzled_n
  //
  // maxPhase divides 32, so phase is kTile-independent and the kTile offset
  // goes in the byte immediate. nPanel cannot be hoisted: it enters n0/16 in
  // multiples of 2, which the phase XOR may flip, so each panel group keeps
  // its own VGPR address (same split as the b8 panel path).
  Value kAddr = b.lshr(laneId, b.i32_val(1));
  Value phase = b.urem(kAddr, b.i32_val(plan.maxPhase));
  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    Value nPanelBase = b.mul(b.i32_val(nPanel * plan.warpsN), b.i32_val(2));
    Value nGroup = b.xor_(
        b.add(b.add(nPanelBase, b.mul(warpNonK, b.i32_val(2))), colInPair),
        phase);
    Value vaddrElem = b.add(b.mul(kAddr, b.i32_val(plan.bn)),
                            b.mul(nGroup, b.i32_val(16)));
    Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, vaddrElem,
                              LLVM::GEPNoWrapFlags::inbounds);
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      Value dsValue = createDsReadM32x32B8(
          loc, rewriter, vecTy, loadAddress,
          b.i32_val(static_cast<int64_t>(kTile) * 32 * plan.bn));

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
  // Every kind reads bk/kPerTile tiles and each lane covers bk/2 elements
  // per panel (kPerTile*32/64), so the total is bk/2 * nPanels.
  SmallVector<Value> outVals;
  outVals.reserve(static_cast<size_t>(plan.bk) / 2 * plan.nPanels);

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

  if (plan.kind == DsReadMKind::B32) {
    // m32x8.b32 returns two K values per N repetition: each K8 tile is
    // [N-rep0 k0,k1, N-rep1 k0,k1]. The MMAC dot operand consumes registers
    // N-rep-major, then K-tile-major.
    for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
      for (int nRep = 0; nRep < 2; ++nRep) {
        for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
          const SmallVector<Value> &frag =
              elemsByKTile[nPanel * plan.kTiles + kTile];
          outVals.push_back(frag[nRep * 2]);
          outVals.push_back(frag[nRep * 2 + 1]);
        }
      }
    }
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
  HCUDsReadMConversion(LLVMTypeConverter &converter, StringRef arch,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), arch(arch.str()) {}
  std::string arch;

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto plan = canUseDsReadM(op);
    if (!plan)
      return failure();

    Location loc = op.getLoc();
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto sharedEncoding =
        dyn_cast<HCUBufferLdsSharedEncodingAttr>(srcTy.getEncoding());
    MatrixReadOpcode opcode = plan->kind == DsReadMKind::B16
                                  ? MatrixReadOpcode::M32x16B16
                                  : plan->kind == DsReadMKind::B32
                                        ? MatrixReadOpcode::M32x8B32
                                        : MatrixReadOpcode::M32x32B8;
    if (sharedEncoding && !matrixReadBankPhases(arch, opcode))
      return rewriter.notifyMatchFailure(op,
                                         "target has no BufferLds matrix-read profile");
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto typeConverter = getTypeConverter();
    Type elemTy = typeConverter->convertType(srcTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         elemTy, rewriter);
    // Panel-major views use a linear affine base; fold the K-tile shift into
    // it, or K0/K1 subslices would emit from the same address (canUseDsReadM
    // only admits such proved affine views).
    Value smemBase =
        !sharedEncoding &&
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
    // M for operand A, N for operand B.
    Value warpNonK = warpIds[cast<DotOperandEncodingAttr>(dstTy.getEncoding()).getOpIdx()];
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    SmallVector<SmallVector<Value>> elemsByKTile =
        sharedEncoding
            ? emitBufferLdsMatrixReads(loc, rewriter, b, smemPtrTy, elemTy,
                                       smemBase, laneId, warpNonK,
                                       getBufferLdsConfig(sharedEncoding),
                                       *plan)
            : plan->kind == DsReadMKind::B32
                  ? emitB32Reads(loc, rewriter, b, smemPtrTy, elemTy, smemBase,
                                 laneId, warpNonK, *plan)
                  : plan->kind == DsReadMKind::B16
                        ? emitB16Reads(loc, rewriter, b, smemPtrTy, elemTy,
                                       smemBase, laneId, warpNonK, *plan)
                        : emitB8Reads(loc, rewriter, b, smemPtrTy, elemTy,
                                      smemBase, laneId, warpNonK, *plan);

    SmallVector<Value> outVals = packDsReadMFragments(elemsByKTile, *plan);
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace mlir::triton::HCU {
void populateDsReadMToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns, StringRef arch,
                                   PatternBenefit benefit) {
  patterns.add<HCUDsReadMConversion>(typeConverter, arch, benefit);
}
} // namespace mlir::triton::HCU
