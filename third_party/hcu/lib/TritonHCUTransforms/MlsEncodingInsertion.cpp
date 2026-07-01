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
    patterns.add<MlsEncodingInsertion>(context, mlsVersion);
    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
