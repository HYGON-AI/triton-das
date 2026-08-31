#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonHCU/MlsGroup.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/Utility.h"
#include "TritonHCU/WdraSplitPlan.h"
using namespace ::mlir::triton::HCU;

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

SmallVector<unsigned> getMatrixLoadTensorOrder(triton::MatrixLoadOp matrixOp, int opIdx) {
  auto strides = matrixOp.getStrides();
  assert(strides.size() == 2);

  auto strideKIdx = opIdx == 0 ? 1 : 0;
  auto strideK = strides[strideKIdx];

  bool kMajor = false;
  if (auto constantOp = strideK.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      kMajor = attr.getValue().isOne();
    }
  }

  if (opIdx == 0) {
    return {kMajor, !kMajor};
  }
  else {
    return {!kMajor, kMajor};
  }
}

ttg::AMDMfmaEncodingAttr findSourceMfmaEncoding(Value value);
bool requiresBInterleave2(triton::MatrixStoreOp storeOp);
bool feedsInterleavedFp16MatrixStore(tt::DotOpInterface dot);

SmallVector<unsigned> getMatrixStoreTensorOrder(triton::MatrixStoreOp storeOp) {
  auto strides = storeOp.getStrides();
  assert(strides.size() == 2);
  if (auto constantOp = strides[1].getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue()))
      if (attr.getValue().isOne())
        return {1, 0};
  return {0, 1};
}

// Chooses a proper MLS instruction
FailureOr<MlsInsn> chooseMlsInstruction(tt::DotOpInterface dot, int opIdx,
                                        triton::MatrixLoadOp matrixOp, bool kMajor,
                                        int mlsVersion, int numWarps) {
  // Get MFMA encoding info.
  auto dType = cast<RankedTensorType>(dot.getOperation()->getResult(0).getType());
  auto mfmaEncoding = dyn_cast<ttg::AMDMfmaEncodingAttr>(dType.getEncoding());

  // Tensor-level shapes stay in logic space. Here we derive the elem-space
  // tile selection inputs needed by MLS / MMAC instructions.
  auto rank = matrixOp.getType().getRank();
  auto matrixElemType = matrixOp.getType().getElementType();
  unsigned bitwidth = matrixElemType.getIntOrFloatBitWidth();
  auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
  auto blockNonK = dType.getShape()[(rank - 2) + opIdx];

  // `matrix_load` and `dot` tensor shapes are logic shapes. Derive the expanded
  // elem-space dimensions used by MLS tile selection from those logic shapes.
  int64_t logicalK = blockK;
  int64_t logicalNonK = blockNonK;
  if (auto wdraPlan = HCU::getWdraSplitPlan(dot)) {
    // TwoPTwoC: partitioned operand is a half-tile MLS write — constrain tile
    // selection by the eventual per-consumer shape.
    // OnePTwoC: one full-tile MLS write; consumers memdesc_subslice — keep
    // full-tile logicalNonK so mlsTile matches the shared write.
    auto topo = HCU::getWdraTopologyAttr(dot);
    if (topo.producerSliced() &&
        wdraPlan->partitionedOperand == static_cast<unsigned>(opIdx)) {
      auto effectiveShape =
          HCU::getWdraEffectiveResultShape(dot, *wdraPlan);
      logicalNonK = effectiveShape[(rank - 2) + opIdx];
    }
  }

  MlsElemBitTyKind mlsElemBitTyKind = MlsElemBitTyKind::None;
  if (auto dotScaled = dyn_cast<tt::DotScaledOp>(dot.getOperation())) {
    auto dotScaledOp = cast<tt::DotScaledOp>(dot);
    auto scaledElemType = opIdx == 0 ? dotScaled.getAElemType() : dotScaled.getBElemType();
    if (scaledElemType == ScaleDotElemType::E2M1 && kMajor)
      logicalK *= 2;

    auto dotEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(
        (opIdx == 0 ? dotScaledOp.getA().getType() : dotScaledOp.getB().getType())
            .getEncoding());
    auto mlsScaledExt = dotEncoding.getMlsScaledExt();
    mlsElemBitTyKind = getMlsElemBitTyKind(mlsScaledExt);
    if (scaledElemType != ScaleDotElemType::E2M1) {
      bool isF4Padding = isMlsPaddedBit4ElemTyKind(mlsElemBitTyKind);
      mlsElemBitTyKind =
          scaleDotElemTypeToMlsElemBitTyKind(scaledElemType, isF4Padding);
    }
  }

  auto altKind = MlsInterleaveKind::InterleaveNone;
  // FP16 VGPR matrix_store requires the TT B operand: its N-major MLS view
  // can use Interleave2 so adjacent output values are packable as f32 VGPRs.
  if (opIdx == 1 && !kMajor && feedsInterleavedFp16MatrixStore(dot))
    altKind = MlsInterleaveKind::Interleave2;

  // Try candidate MLS elems tiles first (same strategy as
  // chooseMfmaInstructionWithMLS): pick the largest valid candidate.
  unsigned nonKTile = opIdx == 0 ? mfmaEncoding.getMfmaTile()[0]
                                 : mfmaEncoding.getMfmaTile()[1];
  unsigned kTile = 0;
  auto tileCandidates = MlsInsn::getMlsElemsTileCandidates(opIdx,
                                                           kMajor,
                                                           bitwidth,
                                                           mlsElemBitTyKind,
                                                           mlsVersion,
                                                           altKind);

  for (int i = tileCandidates.size() - 1; i >= 0; --i) {
    unsigned candNonK = tileCandidates[i].first;
    unsigned candK = tileCandidates[i].second;
    if (candNonK == nonKTile &&
        candNonK <= static_cast<unsigned>(logicalNonK) &&
        candK <= static_cast<unsigned>(logicalK) &&
        logicalNonK % candNonK == 0 && logicalK % candK == 0) {
      kTile = candK;

      if (candNonK * candK * numWarps <= static_cast<unsigned>(logicalNonK * logicalK))
        break;
    }
  }

  assert(kTile != 0 && "pre select mls tile failed, no candidate survived\n");

  // Select the MLS instruction.
  auto maybeMlsInsn = MlsInsn::selectOrGetMlsInsn(nonKTile, kTile, bitwidth,
                                                                      mlsElemBitTyKind,
                                                                      opIdx, kMajor, mlsVersion, altKind);
  if (failed(maybeMlsInsn))
    llvm::report_fatal_error("No match found in MLS database\n");

  return maybeMlsInsn;
}

SmallVector<unsigned>
warpsPerCTAMatrixLoad(ArrayRef<int64_t> shape, ArrayRef<unsigned> shapePerWarp, ArrayRef<unsigned> order, int numWarps) {
  unsigned rank = shape.size();
  SmallVector<unsigned> warpsPerCTA(rank);

  unsigned remainingWarps = numWarps;
  unsigned prevWarps = 1;

  // starting from the contiguous dimension
  for (unsigned d = 0; d < rank - 1; ++d) {
    unsigned i = order[d];
    unsigned maxWarpsInDim = std::max<unsigned>(1, static_cast<unsigned>(shape[i]) / shapePerWarp[i]);
    warpsPerCTA[i] = std::clamp<unsigned>(remainingWarps, 1, maxWarpsInDim);
    remainingWarps /= warpsPerCTA[i];
    prevWarps *= warpsPerCTA[i];
  }

  // Expand the last dimension to fill the remaining lanes and warps
  warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

  return warpsPerCTA;
}

/// Walk convert_layout / trunc / unary elementwise to find the GEMM MFMA-C
/// encoding that produced `value` (if any).
ttg::AMDMfmaEncodingAttr findSourceMfmaEncoding(Value value) {
  Value cur = value;
  for (int depth = 0; depth < 16 && cur; ++depth) {
    auto ty = dyn_cast<RankedTensorType>(cur.getType());
    if (!ty)
      break;
    if (auto mfma = dyn_cast<ttg::AMDMfmaEncodingAttr>(ty.getEncoding()))
      return mfma;
    Operation *def = cur.getDefiningOp();
    if (!def)
      break;
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(def)) {
      cur = cvt.getSrc();
      continue;
    }
    if (isa<arith::TruncFOp, arith::ExtFOp, tt::FpToFpOp>(def) &&
        def->getNumOperands() == 1) {
      cur = def->getOperand(0);
      continue;
    }
    if (def->hasTrait<OpTrait::Elementwise>() && def->getNumOperands() == 1) {
      cur = def->getOperand(0);
      continue;
    }
    break;
  }
  return {};
}

bool requiresBInterleave2(triton::MatrixStoreOp storeOp) {
  auto valueTy = dyn_cast<RankedTensorType>(storeOp.getValue().getType());
  if (!valueTy || !valueTy.getElementType().isF16())
    return false;

  // VGPR matrix-store packing is currently defined for output produced by
  // fp32 MMAC C layout 3 (LTS=1, LIT=0). Output strides can be runtime values,
  // so the decision must not depend on recognizing a constant unit stride.
  auto mfma = findSourceMfmaEncoding(storeOp.getValue());
  return mfma && mfma.getElementBitWidth() == 32 &&
         mfma.getMmacLayout() == ttg::MmacLayout::TRANSPOSE;
}

bool hasInterleavableBOperand(triton::MatrixStoreOp storeOp) {
  auto func = storeOp->getParentOfType<tt::FuncOp>();
  if (!func)
    return false;

  bool found = false;
  func.walk([&](triton::MatrixLoadOp loadOp) {
    if (found)
      return;
    if (auto attr = loadOp->getAttrOfType<triton::amdgpu::MlsEncodingAttr>(
            triton::amdgpu::MlsEncodingAttr::getMnemonic())) {
      if (attr.getOpIdx() == 1 && attr.getOrder()[0] == 1)
        found = true;
      return;
    }
    auto dotAndIdx = getDotOpIdxFromMatrixLoad(loadOp);
    if (succeeded(dotAndIdx) && dotAndIdx->second == 1) {
      auto order = getMatrixLoadTensorOrder(loadOp, /*opIdx=*/1);
      if (order[0] == 1)
        found = true;
    }
  });
  return found;
}

bool feedsInterleavedFp16MatrixStore(tt::DotOpInterface dot) {
  auto resultTy = dyn_cast<RankedTensorType>(dot->getResult(0).getType());
  if (!resultTy || !resultTy.getElementType().isF32())
    return false;
  auto mfma = dyn_cast<ttg::AMDMfmaEncodingAttr>(resultTy.getEncoding());
  if (!mfma || mfma.getMmacLayout() != ttg::MmacLayout::TRANSPOSE)
    return false;

  SetVector<Operation *> slice;
  getForwardSlice(dot->getResult(0), &slice);
  std::deque<Operation *> worklist(slice.begin(), slice.end());
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (auto store = dyn_cast<triton::MatrixStoreOp>(op)) {
      if (requiresBInterleave2(store))
        return true;
      continue;
    }
    auto yield = dyn_cast<scf::YieldOp>(op);
    if (!yield)
      continue;
    for (auto [index, operand] : llvm::enumerate(yield.getOperands())) {
      Operation *def = operand.getDefiningOp();
      if (def != dot.getOperation() && (!def || !slice.contains(def)))
        continue;
      Value loopResult = yield->getParentOp()->getResult(index);
      SetVector<Operation *> outerSlice;
      getForwardSlice(loopResult, &outerSlice);
      for (Operation *outerOp : outerSlice)
        if (slice.insert(outerOp))
          worklist.push_back(outerOp);
    }
  }

  // WASP places the producer load and consumer dot/store in sibling
  // warp-specialize partitions, which are not connected by SSA forward-slice
  // traversal.  The store phase runs first, so use its explicit encoding as
  // the cross-partition contract for layout-3 dots in the same function.
  auto func = dot->getParentOfType<tt::FuncOp>();
  if (!func)
    return false;
  bool hasInterleavedStore = false;
  func.walk([&](triton::MatrixStoreOp store) {
    auto attr = store->getAttrOfType<triton::amdgpu::MlsStoreEncodingAttr>(
        triton::amdgpu::MlsStoreEncodingAttr::getMnemonic());
    if ((attr && attr.getInterleaveKind() ==
                     static_cast<unsigned>(MlsInterleaveKind::Interleave2)) ||
        requiresBInterleave2(store))
      hasInterleavedStore = true;
  });
  return hasInterleavedStore;
}

Value convertAndCastTensor(OpBuilder &builder, Value value,
                           Attribute newEncoding) {
  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      builder.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  return convertedTensor;
}

struct MlsEncodingInsertion : public OpRewritePattern<triton::MatrixLoadOp> {
  int mlsVersion;
public:
  MlsEncodingInsertion(MLIRContext *context, int mlsVersion)
      : OpRewritePattern(context, 1), mlsVersion(mlsVersion) {}

  LogicalResult matchAndRewrite(triton::MatrixLoadOp matrixOp, PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = matrixOp.getType();
    if (!oldRetType.getEncoding() || !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto maybeDotOpIdxPair = getDotOpIdxFromMatrixLoad(matrixOp);
    if (succeeded(maybeDotOpIdxPair)) {
      auto dotOp = maybeDotOpIdxPair.value().first;
      auto opIdx = maybeDotOpIdxPair.value().second;

      // 1. choose the mls instruction for matrixOp
      auto dotRetType =
          cast<RankedTensorType>(dotOp.getOperation()->getResult(0).getType());
      assert(isa<ttg::AMDMfmaEncodingAttr>(dotRetType.getEncoding()));
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);
      auto mlsInsn = chooseMlsInstruction(dotOp, opIdx, matrixOp, kMajor, mlsVersion, numWarps);
      if (failed(mlsInsn))
        return failure();

      // 2. warps per tile calculate
      auto mlsTile = mlsInsn->getMlsTile();
      auto mlsTileGlobal = mlsInsn->getMlsTile(MlsTileKind::Global);
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto altKind = mlsInsn->getAlt2Kind();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTileGlobal, order, numWarps);

      // 3. create new matrix load op with new encoding and mls attr
      auto newEncoding = opIdx == 0
                             ? cast<RankedTensorType>(dotOp.getA().getType()).getEncoding()
                             : cast<RankedTensorType>(dotOp.getB().getType()).getEncoding();
      if (dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getKWidth() != mlsInsn->getDotLayoutKWidth()) {
        newEncoding = ttg::DotOperandEncodingAttr::get(matrixOp.getContext(),
                                                       opIdx,
                                                       dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getParent(),
                                                       mlsInsn->getDotLayoutKWidth());
      }

      auto oldResultType = cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType = RankedTensorType::get(oldResultType.getShape(),
                                                                   oldResultType.getElementType(),
                                                                   newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
                              matrixOp.getLoc(), newResultType,
                              matrixOp.getBase(), matrixOp.getShape(),
                              matrixOp.getStrides(), matrixOp.getTensorShape(), matrixOp.getIndices(),
                              matrixOp.getBoundaryCheck(), matrixOp.getCache(), matrixOp.getEvict(),
                              matrixOp.getIsVolatile());
      auto mlsEncoding = triton::amdgpu::MlsEncodingAttr::get(
                                                          matrixOp.getContext(),
                                                          opIdx,
                                                          mlsTile,
                                                          elemBitWidth,
                                                          mlsInsn->getElemBitTyKind(),
                                                          altKind, version,
                                                          order, warpsPerCTA);
      newMatrixOp->setAttr(triton::amdgpu::MlsEncodingAttr::getMnemonic(), mlsEncoding);

      // 4. replace the original op
      auto convertedTensor = convertAndCastTensor(rewriter, newMatrixOp.getResult(), matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    } else {// non-dot case, try to construct a mfma encoding for the matrix load
      // 1. choose the mls instruction for matrixOp
      unsigned opIdx = 0;
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;

      auto rank = matrixOp.getType().getRank();
      auto elemType = matrixOp.getType().getElementType();
      auto bitwidth = elemType.getIntOrFloatBitWidth();
      auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
      auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);

      unsigned kTile = 0;
      unsigned nonKTile = 0;
      if (mlsVersion == 1) {
        if (bitwidth == 16) {
          nonKTile = kMajor ? 16 : (blockNonK >= 64 ? 64 : 32);
          kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
          if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
            kTile = 32;
        } else if (bitwidth == 8) {
          nonKTile = kMajor ? 16 : (blockNonK >= 128 ? 128 : 64);
          kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
          if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
            kTile = 64;
        }
      } else if (mlsVersion >= 2) {
        if (bitwidth == 16) {
          nonKTile = kMajor ? 16 : 32;
          kTile = kMajor ? 32 : 16;
        } else if (bitwidth == 8) {
          nonKTile = kMajor ? 16 : 64;
          kTile = kMajor ? 64 : 32;
        }
      }

      auto altKind = MlsInterleaveKind::InterleaveNone;
      auto mlsInsn = MlsInsn::selectOrGetMlsInsn(nonKTile, kTile,
                                                 bitwidth, MlsElemBitTyKind::None,
                                                 opIdx, kMajor, mlsVersion, altKind);
      if (failed(mlsInsn))
        llvm::report_fatal_error("No match found in MLS database\n");

      assert(blockK % kTile == 0 && blockNonK % nonKTile == 0 && "M or N should be divisible by mDim or nDim");

      // 3. warps per tile calucate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. try to construct a mfma encoding for the matrix load
      constexpr unsigned mfmaVersion = 3;
      auto maybeMfmaInsn = MfmaIntrinsic::selectFor(matrixOp.getLoc(),
                                                    mfmaVersion,
                                                    16, 16,
                                                    blockK, elemType,
                                                    elemType,
                                                    false, false,
                                                    HCUISAFeature::MAMC_FP8);
      if (failed(maybeMfmaInsn))
        llvm::report_fatal_error("No match found in MFMA database\n");

      SmallVector<unsigned> tilesPerWarp = {1, 1};
      SmallVector<unsigned> mfmaTiles = {16, 16};
      SmallVector<unsigned> mfmaOrder = {1, 0};
      if (rank == 3) {
        tilesPerWarp.insert(tilesPerWarp.begin(), 1);
        mfmaOrder.insert(mfmaOrder.begin(), 2);
        mfmaTiles.insert(mfmaTiles.begin(), 1);
      }

      auto tileM = nonKTile;
      auto tileN = kTile;

      bool hasBatchDim = rank == 3;
      int mIndex = 0 + hasBatchDim;
      int nIndex = 1 + hasBatchDim;
      tilesPerWarp[mIndex] = tileM/maybeMfmaInsn->mDim;
      tilesPerWarp[nIndex] = tileN/maybeMfmaInsn->nDim;

      SmallVector<unsigned> warpsPerCTAMfma = warpsPerCTAMatrixLoad(shape, mfmaTiles, mfmaOrder, numWarps);
      unsigned mfmaElementBitWidth = elemType.isF64() ? 64 : 32;
      auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        matrixOp.getContext(),
        /*versionMajor*/ mfmaVersion,
        warpsPerCTAMfma,
        /*instrShape=*/SmallVector<unsigned>{maybeMfmaInsn->mDim, maybeMfmaInsn->nDim,
                                             maybeMfmaInsn->kDim},
        /*isTransposed=*/false,
        ttg::getCTALayout(matrixOp.getType().getEncoding()),
        tilesPerWarp,
        mfmaElementBitWidth,
        ttg::MmacLayout::INTERLEAVE_TRANSPOSE);
      auto dotAEncoding = ttg::DotOperandEncodingAttr::get(matrixOp.getContext(),
                                                          0, mfmaEnc,
                                                          maybeMfmaInsn->kBase);

      // 4. create new matrix load op with new encoding and mls attr
      auto newEncoding = dotAEncoding;

      auto oldResultType = cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType = RankedTensorType::get(oldResultType.getShape(),
                                                                   oldResultType.getElementType(),
                                                                   newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
                              matrixOp.getLoc(), newResultType,
                              matrixOp.getBase(), matrixOp.getShape(),
                              matrixOp.getStrides(), matrixOp.getTensorShape(), matrixOp.getIndices(),
                              matrixOp.getBoundaryCheck(), matrixOp.getCache(), matrixOp.getEvict(),
                              matrixOp.getIsVolatile());
      auto mlsEncoding = triton::amdgpu::MlsEncodingAttr::get(
                                                          matrixOp.getContext(), opIdx, mlsTile,
                                                          elemBitWidth, mlsInsn->getElemBitTyKind(),
                                                          static_cast<unsigned>(altKind), version,
                                                          order, warpsPerCTA);
      newMatrixOp->setAttr(triton::amdgpu::MlsEncodingAttr::getMnemonic(), mlsEncoding);

      // 5. replace the original op
      auto convertedTensor = convertAndCastTensor(rewriter,
                                                        newMatrixOp.getResult(),
                                                        matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    }

    return success();
  }
};

/// Select an fp16 VGPR-store encoding for MMAC C layout 3.
struct MlsStoreEncodingInsertion
    : public OpRewritePattern<triton::MatrixStoreOp> {
  int mlsVersion;

public:
  MlsStoreEncodingInsertion(MLIRContext *context, int mlsVersion)
      : OpRewritePattern(context, 1), mlsVersion(mlsVersion) {}

  LogicalResult matchAndRewrite(triton::MatrixStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (storeOp->hasAttr(
            triton::amdgpu::MlsStoreEncodingAttr::getMnemonic()))
      return failure();

    auto value = storeOp.getValue();
    auto valueType = cast<RankedTensorType>(value.getType());
    if (valueType.getRank() != 2 || !valueType.getElementType().isF16() ||
        !requiresBInterleave2(storeOp) || !hasInterleavableBOperand(storeOp))
      return failure();

    constexpr unsigned bitwidth = 16;
    constexpr auto altKind = MlsInterleaveKind::Interleave2;

    SmallVector<unsigned> order = getMatrixStoreTensorOrder(storeOp);
    bool transpose = order[0] == 1;
    if (!transpose)
      return failure();

    auto srcMfma = findSourceMfmaEncoding(value);
    assert(srcMfma && "requiresBInterleave2 must find an MFMA source");
    auto warpDims = srcMfma.getWarpsPerCTA();
    if (warpDims.size() < 2 ||
        valueType.getShape()[0] % warpDims[0] != 0 ||
        valueType.getShape()[1] % warpDims[1] != 0)
      return failure();

    // Instruction names are [contiguous, non-contiguous], while candidates
    // use logical tensor [M, N] tiles. Select against the per-warp C tile using
    // the same capability-query strategy as MLS load selection. Scanning the
    // M/N-sorted candidates backwards prefers 32x32 over 16x64 when both have
    // the same area; this also avoids the currently unsupported PMD VGPR-256
    // path whenever the 32x32 instruction can cover the warp tile.
    int64_t warpM = valueType.getShape()[0] / warpDims[0];
    int64_t warpN = valueType.getShape()[1] / warpDims[1];
    auto candidates = MlsInsn::getMatrixStoreTileCandidates(
        bitwidth, mlsVersion, altKind, transpose);
    std::optional<std::array<unsigned, 2>> selectedTile;
    for (auto tile : llvm::reverse(candidates)) {
      if (warpM % tile[0] == 0 && warpN % tile[1] == 0) {
        selectedTile = tile;
        break;
      }
    }
    if (!selectedTile)
      return failure();
    unsigned mTile = (*selectedTile)[0];
    unsigned nTile = (*selectedTile)[1];

    auto mlsInsn = MlsInsn::selectOrGetMatrixStoreInsn(
        mTile, nTile, bitwidth, mlsVersion, altKind, transpose);
    if (failed(mlsInsn))
      return failure();

    Value convertedValue = valueType.getEncoding() == Attribute(srcMfma)
                               ? value
                               : convertAndCastTensor(rewriter, value, srcMfma);
    SmallVector<unsigned> warpsPerCTA(srcMfma.getWarpsPerCTA().begin(),
                                      srcMfma.getWarpsPerCTA().end());

    auto mlsEncoding = triton::amdgpu::MlsStoreEncodingAttr::get(
        storeOp.getContext(), SmallVector<unsigned>{mTile, nTile}, bitwidth,
        static_cast<unsigned>(altKind), mlsInsn->getMlsVersion(), order,
        warpsPerCTA);

    rewriter.modifyOpInPlace(storeOp, [&]() {
      storeOp.getValueMutable().assign(convertedValue);
      storeOp->setAttr(triton::amdgpu::MlsStoreEncodingAttr::getMnemonic(),
                       mlsEncoding);
      if (auto func = storeOp->getParentOfType<triton::FuncOp>())
        func->setAttr("triton.hcu.has_matrix_store",
                      UnitAttr::get(storeOp.getContext()));
    });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUMLSENCODINGINSERTION
#include "TritonHCU/Passes.h.inc"

class TritonHCUMlsEncodingInsertionPass
    : public impl::TritonHCUMlsEncodingInsertionBase<
          TritonHCUMlsEncodingInsertionPass> {
public:
  using impl::TritonHCUMlsEncodingInsertionBase<
      TritonHCUMlsEncodingInsertionPass>::TritonHCUMlsEncodingInsertionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    auto arch = getAMDArch(mod);
    assert(arch.has_value() && "expected arch");
    auto features = mlir::triton::HCU::deduceHCUISAFeature(*arch);
    auto mlsVersion = getMlsVersionFromFeatures(features);
    mlir::RewritePatternSet patterns(context);
    patterns.add<MlsEncodingInsertion, MlsStoreEncodingInsertion>(context,
                                                                  mlsVersion);
    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }

    bool invalidFp16MatrixStore = false;
    mod.walk([&](triton::MatrixStoreOp storeOp) {
      if (storeOp->hasAttr(
              triton::amdgpu::MlsStoreEncodingAttr::getMnemonic()))
        return;
      auto valueType = dyn_cast<RankedTensorType>(storeOp.getValue().getType());
      if (!valueType || !valueType.getElementType().isF16() ||
          !requiresBInterleave2(storeOp) ||
          hasInterleavableBOperand(storeOp))
        return;
      storeOp.emitError()
          << "FP16 matrix_store from MMAC layout 3 requires TT operand "
             "layout so the B operand can use N-direction Interleave2; use "
             "tt.store for non-TT layouts";
      invalidFp16MatrixStore = true;
    });
    if (invalidFp16MatrixStore)
      signalPassFailure();
  }
};

} // namespace mlir
