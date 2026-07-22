#include "PatternTritonGPUOpToLLVM.h"
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
  int64_t maxPhase;
  unsigned warpsN;
  unsigned nPanels;
  unsigned kTiles; // b16: BK/16; b8: BK/32
};

std::optional<unsigned> getDsReadMBitWidth(Type srcElemTy, Type dstElemTy) {
  if (!srcElemTy.isIntOrFloat() || !dstElemTy.isIntOrFloat())
    return std::nullopt;

  unsigned srcBits = srcElemTy.getIntOrFloatBitWidth();
  unsigned dstBits = dstElemTy.getIntOrFloatBitWidth();
  if (srcBits != dstBits || (srcBits != 8 && srcBits != 16))
    return std::nullopt;
  return srcBits;
}

// TODO: ds_read_m may not strictly require maxPhase to match this
int64_t getExpectedMaxPhase(unsigned elemBits, int64_t bn) {
  return elemBits == 8 ? std::min<int64_t>(4, bn / 16) : 2;
}

bool hasExpectedSharedLayout(SwizzledSharedEncodingAttr sharedEnc,
                             unsigned elemBits, int64_t bn) {
  if (!sharedEnc || sharedEnc.getOrder().size() != 2 ||
      sharedEnc.getOrder()[0] != 1 || sharedEnc.getOrder()[1] != 0)
    return false;

  if (elemBits == 16)
    return sharedEnc.getVec() == 8 && sharedEnc.getPerPhase() == 2 &&
           sharedEnc.getMaxPhase() == getExpectedMaxPhase(elemBits, bn);

  return sharedEnc.getVec() == 16 && sharedEnc.getPerPhase() == 1 &&
         sharedEnc.getMaxPhase() == getExpectedMaxPhase(elemBits, bn);
}

std::optional<DsReadMPlan> canUseDsReadM(LocalLoadOp op) {
  // Default-on: unset => enabled; ENABLE=0/false/off disables.
  // (getBoolEnv treats unset as false, so do not use it for this flag.)
  std::string enableEnv =
      triton::tools::getStrEnv("TRITON_HCU_ENABLE_DS_READ_M");
  if (!enableEnv.empty()) {
    if (auto v = triton::tools::isEnvValueBool(enableEnv); v && !*v)
      return std::nullopt;
  }

  auto srcTy = cast<MemDescType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2 || !dotEnc ||
      dotEnc.getOpIdx() != 1)
    return std::nullopt;

  auto bitWidth =
      getDsReadMBitWidth(srcTy.getElementType(), dstTy.getElementType());
  auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding());
  // Dense N-major B[BK, BN] only; affine/sliced LDS needs a separate vaddr path.
  if (!bitWidth ||
      LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(srcTy))
    return std::nullopt;

  // b16: ds_read_m32x16 covers K=16; b8: ds_read_m32x32 covers K=32.
  int64_t bk = dstTy.getShape()[0], bn = dstTy.getShape()[1];
  unsigned kPerTile = *bitWidth == 8 ? 32 : 16;
  unsigned kWidth = *bitWidth == 8 ? 8 : 4;
  if (bn < 32 || bn % 32 || bk < kPerTile || bk % kPerTile ||
      !hasExpectedSharedLayout(sharedEnc, *bitWidth, bn))
    return std::nullopt;

  auto mfmaEnc = cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
  auto instrShape = mfmaEnc.getInstrShape();
  auto warpsPerCTA = mfmaEnc.getWarpsPerCTA();
  auto tilesPerWarp = mfmaEnc.getTilesPerWarp();
  // 16x16xK MFMA, tilesPerWarp={1,2}; each N-warp owns >=1 full 32-col panel.
  // AccelerateHCUMatmul redistributes warpsPerCTA so warpsN <= BN/32.
  unsigned warpsN = warpsPerCTA.size() == 2 ? warpsPerCTA[1] : 0;
  if (instrShape.size() != 3 || instrShape[0] != 16 || instrShape[1] != 16 ||
      instrShape[2] != kPerTile || dotEnc.getKWidth() != kWidth ||
      tilesPerWarp.size() != 2 || tilesPerWarp[0] != 1 ||
      tilesPerWarp[1] != 2 || warpsN == 0 || bn % (warpsN * 32))
    return std::nullopt;

  return DsReadMPlan{*bitWidth == 8 ? DsReadMKind::B8 : DsReadMKind::B16,
                     bk,
                     bn,
                     getExpectedMaxPhase(*bitWidth, bn),
                     warpsN,
                     static_cast<unsigned>(bn / (warpsN * 32)),
                     static_cast<unsigned>(bk / kPerTile)};
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
// Contiguous N-major B[K,N] panel addressing:
//   k_addr = lane >> 2                         // 0..15, K row this lane addresses
//   c      = lane & 3                          // 4 lanes cover one K row
//   n0     = panel * 32 + c * 8                // logical N start of 8-half chunk
//   vaddr_half_contiguous = k_addr * N + n0
//
// Swizzled shared {vec=8, perPhase=2, maxPhase=2, order=[1,0]}:
//   phase(k)        = (k / perPhase) % maxPhase
//   swizzled_n(k,n) = ((n / vec) ^ phase(k)) * vec + (n % vec)
//   vaddr_half      = k * N + swizzled_n(k, n0)
//
// With vec=8, n0 is always 8-aligned, so this simplifies to:
//   phase = ((lane >> 2) / 2) % 2              // == (k_addr >> 1) & 1
//   vaddr_half =
//       (lane >> 2) * N + (((panel * 4 + (lane & 3)) ^ phase) * 8)
//
// Multi-warp / multi-K: panel = nPanel * warpsN + warpN; k_addr += kTile * 16.
SmallVector<SmallVector<Value>>
emitB16Reads(Location loc, RewriterBase &rewriter, TritonLLVMOpBuilder &b,
             Type smemPtrTy, Type elemTy, Value smemBase, Value laneId,
             Value warpId, const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> elemsByKTile;
  Type vecTy = vec_ty(elemTy, 8);

  Value colInQuad = b.and_(laneId, b.i32_val(3)); // c = lane & 3
  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      // k_addr = (lane >> 2) + kTile * 16
      Value kAddr =
          b.add(b.lshr(laneId, b.i32_val(2)), b.i32_val(kTile * 16));
      // phase = (k_addr / 2) % 2
      Value phase = b.and_(b.lshr(kAddr, b.i32_val(1)), b.i32_val(1));
      // nGroup = (panel * 4 + c) ^ phase, panel = nPanel * warpsN + warpN
      Value nPanelBase = b.mul(b.i32_val(nPanel * plan.warpsN), b.i32_val(4));
      Value nGroup = b.xor_(
          b.add(b.add(nPanelBase, b.mul(warpId, b.i32_val(4))), colInQuad),
          phase);
      // vaddr_half = k_addr * BN + nGroup * 8
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
// Contiguous N-major B[K,N] panel addressing (16 i8 = 128-bit per lane):
//   k_addr = lane >> 1                         // 0..31, K row this lane addresses
//   c      = lane & 1                          // 2 lanes cover one K row
//   n0     = panel * 32 + c * 16               // logical N start of 16-i8 chunk
//   vaddr_i8_contiguous = k_addr * N + n0
//
// Swizzled shared {vec=16, perPhase=1, maxPhase=min(4, BN/16), order=[1,0]}:
//   phase(k)        = (k / perPhase) % maxPhase   // == k % maxPhase
//   swizzled_n(k,n) = ((n / vec) ^ phase(k)) * vec + (n % vec)
//   vaddr_i8        = k * N + swizzled_n(k, n0)
//
// With vec=16, n0 is always 16-aligned, so this simplifies to:
//   phase = k_addr % maxPhase
//   vaddr_i8 =
//       (lane >> 1) * N + (((panel * 2 + (lane & 1)) ^ phase) * 16)
//
// Multi-warp / multi-K: panel = nPanel * warpsN + warpN; k_addr += kTile * 32.
// Hardware returns 4xi32 = two 8-element MMAC-B K fragments (regs[0:2], [2:4]).
SmallVector<SmallVector<Value>>
emitB8Reads(Location loc, RewriterBase &rewriter, TritonLLVMOpBuilder &b,
            Type smemPtrTy, Type elemTy, Value smemBase, Value laneId,
            Value warpId, const DsReadMPlan &plan) {
  SmallVector<SmallVector<Value>> elemsByPanel;
  Value colInPair = b.and_(laneId, b.i32_val(1)); // c = lane & 1
  Type vecTy = vec_ty(rewriter.getI32Type(), 4);
  for (unsigned nPanel = 0; nPanel < plan.nPanels; ++nPanel) {
    for (unsigned kTile = 0; kTile < plan.kTiles; ++kTile) {
      // k_addr = (lane >> 1) + kTile * 32
      Value kAddr =
          b.add(b.lshr(laneId, b.i32_val(1)), b.i32_val(kTile * 32));
      // phase = k_addr % maxPhase
      Value phase = b.urem(kAddr, b.i32_val(plan.maxPhase));
      // nGroup = (panel * 2 + c) ^ phase, panel = nPanel * warpsN + warpN
      Value nPanelBase = b.mul(b.i32_val(nPanel * plan.warpsN), b.i32_val(2));
      Value nGroup = b.xor_(
          b.add(b.add(nPanelBase, b.mul(warpId, b.i32_val(2))), colInPair),
          phase);
      // vaddr_i8 = k_addr * BN + nGroup * 16
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
                           smemObj.getBase(), laneId, warpN, *plan)
            : emitB8Reads(loc, rewriter, b, smemPtrTy, elemTy,
                          smemObj.getBase(), laneId, warpN, *plan);

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
