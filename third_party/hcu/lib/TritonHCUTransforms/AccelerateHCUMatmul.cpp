#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonHCU/MlsGroup.h"
#include "TritonHCU/WdraSplitPlan.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonHCU/Passes.h"
#include "TritonHCU/Utility.h"
using namespace ::mlir::triton::HCU;
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using ::mlir::LLVM::AMD::isChainDotHead;
using ::mlir::LLVM::AMD::isChainDotTail;
using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
using mlir::triton::gpu::chooseScaledMfmaScaleLayout;
using mlir::triton::gpu::MmacLayout;
using ::mlir::triton::HCU::HCUISAFeature;
#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamd-accelerate-matmul"
namespace mlir {
namespace {
using triton::AMD::ISAFamily;
constexpr char AttrDecomposedDotScaledSource[] =
    "amdg.decomposed_dot_scaled_source";

bool isDsReadMEnabled() {
  // Default-on while keeping the ENABLE_* name: getBoolEnv() treats unset as
  // false, so parse explicitly here instead of changing the public helper.
  std::string s = triton::tools::getStrEnv("TRITON_HCU_ENABLE_DS_READ_M");
  if (s.empty())
    return true;
  if (auto v = triton::tools::isEnvValueBool(s))
    return *v;
  return true;
}

struct MatrixLoadInfo {
  triton::MatrixLoadOp matrixOp;
  bool isKMajor;
  unsigned dotMfmaKWidth; /* mls determines its dot layout kWidth */
  MlsElemBitTyKind mlsElemBitTyKind;
  bool isNonKPack;
  SmallVector<std::pair<unsigned, unsigned>> elemsTileCands;
  std::pair<unsigned, unsigned> choosedElemsTile; // final {nonKTile, kTile} for MLS
};
using DotOperandMatrixLoadInfo =
    std::pair<std::optional<MatrixLoadInfo>, std::optional<MatrixLoadInfo>>;
std::optional<MatrixLoadInfo> buildDotOperandMatrixLoadInfo(DotOpInterface dotOp, int opIdx,
                                                            tt::MatrixLoadOp matrixOp) {
  if (!matrixOp)
    return std::nullopt;
  auto getKMajorFromMatrixLoad = [](tt::MatrixLoadOp checkMatrixOp,
                                    int checkOpIdx) -> bool {
    auto strideKIdx = checkOpIdx == 0 ? 1 : 0;
    auto strideK = checkMatrixOp.getStrides()[strideKIdx];
    if (auto constantOp = strideK.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue()))
        return attr.getValue().isOne();
    }
    return false;
  };
  bool kMajor = getKMajorFromMatrixLoad(matrixOp, opIdx);
  if (auto dotScaledOp = dyn_cast<tt::DotScaledOp>(dotOp.getOperation())) {
    if (opIdx == 0 && dotScaledOp.getAElemType() == ScaleDotElemType::E2M1)
      assert(dotScaledOp.getLhsKPack() == kMajor &&
             "opIdx 0: kMajor mismatch for lhs kPack!");
    if (opIdx == 1 && dotScaledOp.getBElemType() == ScaleDotElemType::E2M1)
      assert(dotScaledOp.getRhsKPack() == kMajor &&
             "opIdx 1: kMajor mismatch for rhs kPack!");
  }
  unsigned elemBitWidth =
      matrixOp.getType().getElementType().getIntOrFloatBitWidth();
  unsigned dotMfmaKWidth =
      elemBitWidth == 16 ? 4 : (elemBitWidth == 8 ? 8 : 8);
  return MatrixLoadInfo{matrixOp,
                        kMajor,
                        dotMfmaKWidth,
                        MlsElemBitTyKind::None,
                        false,
                        {},
                        {0, 0}};
}
DotOperandMatrixLoadInfo getDotOperandMatrixLoadInfo(DotOpInterface dotOp,
                                                     HCUISAFeature features = HCUISAFeature::NONE) {
  auto modOp = dotOp->getParentOp();
  while (!isa<ModuleOp>(modOp)) {
    modOp = modOp->getParentOp();
  }
  ModuleOp mod = cast<ModuleOp>(modOp);
  std::array<tt::MatrixLoadOp, 2> matrixOps = {nullptr, nullptr};
  mod.walk([&](tt::MatrixLoadOp mOp) {
    auto maybeDotOpIdxPair = getDotOpIdxFromMatrixLoad(mOp);
    if (failed(maybeDotOpIdxPair) || maybeDotOpIdxPair.value().first != dotOp)
      return;
    int opIdx = maybeDotOpIdxPair.value().second;
    if (opIdx >= 0 && opIdx < 2 && !matrixOps[opIdx])
      matrixOps[opIdx] = mOp;
  });
  std::array<std::optional<MatrixLoadInfo>, 2> infos = {
      buildDotOperandMatrixLoadInfo(dotOp, 0, matrixOps[0]),
      buildDotOperandMatrixLoadInfo(dotOp, 1, matrixOps[1])};
  unsigned mlsVersion = getMlsVersionFromFeatures(features);
  if (auto dotScaledOp = dyn_cast<tt::DotScaledOp>(dotOp.getOperation())) {
    auto oldRetShape = dotScaledOp.getType().getShape();
    auto oldRetRank = dotScaledOp.getType().getRank();
    int64_t logicalM = oldRetShape[oldRetRank - 2];
    int64_t logicalN = oldRetShape[oldRetRank - 1];
    auto getOperandMlsElemBitTyKind =
        [&](int opIdx, bool useB4MixTyScale) -> MlsElemBitTyKind {
      auto scaledElemType =
          opIdx == 0 ? dotScaledOp.getAElemType() : dotScaledOp.getBElemType();
      if (scaledElemType == ScaleDotElemType::E2M1)
        return useB4MixTyScale ? MlsElemBitTyKind::B4Mix
                               : MlsElemBitTyKind::B4;
      return scaleDotElemTypeToMlsElemBitTyKind(scaledElemType,
                                                /*isF4Padding=*/false);
    };
    auto hasCompatTile = [&](const MatrixLoadInfo &info, int opIdx,
                             MlsElemBitTyKind elemBitTyKind) -> bool {
      auto cMatrixOp = info.matrixOp;
      auto cShape = cMatrixOp.getType().getShape();
      auto cRank = cMatrixOp.getType().getRank();
      int64_t logicalK = cShape[(cRank - 1) - opIdx];
      int64_t logicalNonK = opIdx == 0 ? logicalM : logicalN;
      auto cScaledElemType =
          opIdx == 0 ? dotScaledOp.getAElemType() : dotScaledOp.getBElemType();
      if (cScaledElemType == ScaleDotElemType::E2M1 && info.isKMajor)
        logicalK *= 2;
      unsigned kWidth = getMlsDataFlowInfo(elemBitTyKind).getMfmaKWidth(8);
      int64_t minMfmaInstKReq = 64 / 16 * kWidth;
      auto tileCands = MlsInsn::getMlsElemsTileCandidates(
          opIdx, info.isKMajor,
          cMatrixOp.getType().getElementType().getIntOrFloatBitWidth(),
          elemBitTyKind, mlsVersion);
      for (const auto &[candNonK, candK] : tileCands) {
        bool outputShapeMatched =
            logicalM > 0 && logicalN > 0 &&
            candNonK <= static_cast<unsigned>(logicalNonK) &&
            candK <= static_cast<unsigned>(logicalK) &&
            candK >= minMfmaInstKReq &&
            logicalNonK % candNonK == 0 && logicalK % candK == 0;
        if (outputShapeMatched)
          return true;
      }
      return false;
    };
    bool isAE2M1 = dotScaledOp.getAElemType() == ScaleDotElemType::E2M1;
    bool isBE2M1 = dotScaledOp.getBElemType() == ScaleDotElemType::E2M1;
    bool hasAnyE2M1 = isAE2M1 || isBE2M1;
    bool useB4 = false;
    bool useB4Mix = false;
    if ((features & HCUISAFeature::MLS_B4) && hasAnyE2M1 &&
        infos[0].has_value() && infos[1].has_value()) {
      bool useB4NonMixTyScale =
          isAE2M1 && isBE2M1 &&
          hasCompatTile(*infos[0], 0, MlsElemBitTyKind::B4) &&
          hasCompatTile(*infos[1], 1, MlsElemBitTyKind::B4);
      bool useB4MixedTyScale =
          hasCompatTile(*infos[0], 0, getOperandMlsElemBitTyKind(0, true)) &&
          hasCompatTile(*infos[1], 1, getOperandMlsElemBitTyKind(1, true));
      useB4 = useB4NonMixTyScale || useB4MixedTyScale;
      useB4Mix = !useB4NonMixTyScale && useB4MixedTyScale;
    }
    for (int opIdx = 0; opIdx < 2; ++opIdx) {
      if (!infos[opIdx].has_value())
        continue;
      auto scaledElemType =
          opIdx == 0 ? dotScaledOp.getAElemType() : dotScaledOp.getBElemType();
      bool operandUsesB4 =
          useB4 && scaledElemType == ScaleDotElemType::E2M1;
      bool isF4Padding = !operandUsesB4;
      infos[opIdx]->mlsElemBitTyKind = operandUsesB4
                                           ? (useB4Mix ? MlsElemBitTyKind::B4Mix
                                                       : MlsElemBitTyKind::B4)
                                           : scaleDotElemTypeToMlsElemBitTyKind(
                                                 scaledElemType, isF4Padding);
      infos[opIdx]->isNonKPack =
          !(opIdx == 0 ? dotScaledOp.getLhsKPack() : dotScaledOp.getRhsKPack());
    }
  }
  for (int opIdx = 0; opIdx < 2; ++opIdx) {
    if (!infos[opIdx].has_value())
      continue;
    auto &info = *infos[opIdx];
    unsigned elemBitWidth = info.matrixOp.getType().getElementType().getIntOrFloatBitWidth();
    info.elemsTileCands = MlsInsn::getMlsElemsTileCandidates(
        opIdx, info.isKMajor, elemBitWidth, info.mlsElemBitTyKind,
        mlsVersion);
  }
  return {infos[0], infos[1]};
}
/* HCU support: mmac v1 dont have chaindot or has special accel mode */
bool hasReduceInChainDots(Operation *dotOp) {
  if (!isa<mlir::triton::DotOpInterface>(dotOp))
    return false;
  auto filter = [dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  // step 1. collect chain dots
  SmallVector<Operation *, 4> chainDots{dotOp};
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices) {
    if (isa<mlir::triton::DotOpInterface>(op) && (op != dotOp))
      chainDots.push_back(op);
  }
  // step 2. check reduce exist in chain dots.
  for(auto dot : chainDots) {
    SetVector<mlir::Operation*> fwdSlices;
    getForwardSlice(dot, &fwdSlices);
    for (Operation *op : fwdSlices) {
      if (isa<tt::ReduceOp>(op))
        return true;
    }
  }
  return false;
}
bool hasTransInDefChain(tt::DotOpInterface dotOp, unsigned opIdx) {
  SmallVector<Value> worklist{dotOp->getOperand(opIdx)};
  DenseSet<Value> seen;
  Region *region = dotOp->getParentRegion();
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getParentRegion() != region)
      continue;
    if (isa<tt::TransOp>(defOp))
      return true;
    llvm::append_range(worklist, defOp->getOperands());
  }
  return false;
}
int getMfmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
    return 1;
  case ISAFamily::CDNA2:
    return 2;
  case ISAFamily::CDNA3:
    return 3;
  case ISAFamily::CDNA4:
    return 4;
  default:
    break;
  }
  return 0;
}
int getWmmaVersion(StringRef archGen) {
  if (archGen.contains("gfx11"))
    return 1;
  if (archGen.contains("gfx12"))
    return 2;
  return 0;
}
FailureOr<ScaleDotElemType> mlirTypeToScaledElemType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<ScaleDotElemType>>(type)
      .Case<Float8E4M3FNType>([](Type) { return ScaleDotElemType::E4M3; })
      .Case<Float8E5M2Type>([](Type) { return ScaleDotElemType::E5M2; })
      .Case<Float6E3M2FNType>([](Type) { return ScaleDotElemType::E3M2; })
      .Case<Float6E2M3FNType>([](Type) { return ScaleDotElemType::E2M3; })
      .Case<Float4E2M1FNType>([](Type) { return ScaleDotElemType::E2M1; })
      .Default([](Type) { return failure(); });
}
SmallVector<unsigned, 3>
warpsPerTile(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
             std::pair<int64_t, int64_t> shapePerWarp) {
  auto rank = shape.size();
  // Case 1: Early exit for batched matmul
  if (rank == 3)
    return {static_cast<unsigned>(numWarps), 1, 1};
  // ZTODO: Need check to remove ?
  // Case for HCUs
  // FIXME: This is primarily an optimization for HCUs(gfx926/928/936), as more mmac
  //        layouts are supported on new-generation HCUs, here we need to return the
  //        appropriate warps based on different layouts currently in use.
  if (hasReduceInChainDots(dotOp))
    return {static_cast<unsigned>(numWarps), 1};
  // Case 2: For FA-like pattern, i.e. result of 1st tl.dot is used as the opA
  // of the 2nd dot, we will set warpsPerCTA differently for 1st and 2nd dot
  auto ttDotOp = cast<tt::DotOpInterface>(dotOp);
  bool isHeadDot = isChainDotHead(ttDotOp);
  bool isTailDot = isChainDotTail(ttDotOp);
  // For the 1st dot in chain-dot, we always set warpsPerCTA={numWarps, 1}
  // because this eliminates
  // 1) inter-warp reduction in the softmax step.
  // 2) layout conversion from #mma to #dot_op of the second dot.
  if (isHeadDot)
    return {static_cast<unsigned>(numWarps), 1};
  // For the 2nd dot in chain-dot, we always distribute warp along dim0 first,
  // then dim1. Because
  // 1) This is how we distribute the warps for the 1st dot. Now the
  //    warpsPerCTA for the 1st dot become the warp layout of the dotOperand
  //    layout of the 2nd dot, which must match the warpsPerCTA of the 2nd dot.
  // 2) When shape[0] is small, as in decode kernels, we don't want to
  //    distribute more warps than shape[0] // mDim. If we do so, each warp
  //    needs to hold more elements in the final output, which increases
  //    register pressure, especially for large head dim (e.g. 512) attention
  //    kernels.
  if (isTailDot) {
    SmallVector<unsigned, 3> ret = {1, 1};
    ret[0] = static_cast<unsigned>(std::min(
        static_cast<int64_t>(numWarps),
        static_cast<int64_t>(llvm::divideCeil(shape[0], shapePerWarp.first))));
    ret[1] = numWarps / ret[0];
    return ret;
  }
  // Case 3: Regular cases
  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 3> ret = {1, 1};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp.first * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp.second / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp.first) {
        ret[0] *= 2;
      } else {
        ret[1] *= 2;
      }
    } else {
      ret[1] *= 2;
    }
  } while (true);
  if (ret[1] * shapePerWarp.second > tensorShape[1]) {
    return {ret[1], ret[0]};
  }
  return ret;
}
SmallVector<unsigned, 3>
warpsPerTileMFMA(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
                 std::pair<int64_t, int64_t> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}
SmallVector<unsigned, 3>
warpsPerTileWMMA(Operation *dotOp, ArrayRef<int64_t> shape, int numWarps,
                 std::pair<int64_t, int64_t> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}
std::optional<ttg::DotOperandEncodingAttr>
getCompatibleDotOpEncoding(int opIdx,
                           ttg::DotOperandEncodingAttr dotAEncoding,
                           ttg::DotOperandEncodingAttr dotBEncoding,
                           std::optional<MatrixLoadInfo> aMatInfo,
                           std::optional<MatrixLoadInfo> bMatInfo) {
  if (!aMatInfo.has_value() && !bMatInfo.has_value())
    return std::nullopt;
  auto kWidthA = dotAEncoding.getKWidth();
  auto kWidthB = dotBEncoding.getKWidth();
  if (kWidthA == kWidthB)
      return std::nullopt;
  auto mfmaLayoutA = dyn_cast<AMDMfmaEncodingAttr>(dotAEncoding.getParent());
  auto mfmaLayoutB = dyn_cast<AMDMfmaEncodingAttr>(dotBEncoding.getParent());
  assert(mfmaLayoutA == mfmaLayoutB && "dotA and dotB should have the same mfma layout");
  auto ctx = dotAEncoding.getParent().getContext();
  if (opIdx == 0) {
    if (kWidthA == 4 && kWidthB == 8) {
      return std::nullopt;
    } else if (kWidthA == 8 && kWidthB == 4) {
      return ttg::DotOperandEncodingAttr::get(ctx, 0, dotAEncoding.getParent(), 4);
    } else {
      assert(false && "Unsupported kWidth compatible case!");
    }
  } else {
    if (kWidthA == 4 && kWidthB == 8) {
      return ttg::DotOperandEncodingAttr::get(ctx, 1, dotBEncoding.getParent(), 4);
    } else if (kWidthA == 8 && kWidthB == 4) {
      return std::nullopt;
    } else {
      assert(false && "Unsupported kWidth compatible case!");
    }
  }
  return std::nullopt;
}
FailureOr<MfmaIntrinsic>
chooseMfmaInstructionWithMLS(tt::DotOpInterface dot,
                             int mfmaVersion, int enforcedNonKDim,
                             std::optional<MatrixLoadInfo> &aMatInfo,
                             std::optional<MatrixLoadInfo> &bMatInfo,
                             HCUISAFeature features,
                             const HCU::WdraSplitPlan *wdraPlan = nullptr) {
  (void)enforcedNonKDim;
  RankedTensorType aType = cast<RankedTensorType>(dot.getA().getType());
  RankedTensorType bType = cast<RankedTensorType>(dot.getB().getType());
  RankedTensorType cType =
      cast<RankedTensorType>(dot.getOperation()->getResult(0).getType());
  int64_t inputKDim = aType.getShape().back();
  Type aElemType = aType.getElementType();
  Type bElemType = bType.getElementType();
  bool withScale = false;
  bool bothOperandsAreE2M1 = false;
  auto resShape = cType.getShape();
  auto rank = resShape.size();
  auto M = resShape[rank - 2];
  auto N = resShape[rank - 1];
  // Before DataPartition clones the graph, select MLS/MFMA tiles using the
  // logical shape that one WDRA consumer will own. The result tensor remains
  // full-sized at this point; only candidate legality/coverage is per-slice.
  if (wdraPlan) {
    auto effectiveShape = HCU::getWdraEffectiveResultShape(dot, *wdraPlan);
    M = effectiveShape[effectiveShape.size() - 2];
    N = effectiveShape[effectiveShape.size() - 1];
  }
  if (auto dotScaled = dyn_cast<tt::DotScaledOp>(dot.getOperation())) {
    auto ctx = dotScaled.getContext();
    aElemType = scaleDotElemTypeToMLIRType(ctx, dotScaled.getAElemType());
    bElemType = scaleDotElemTypeToMLIRType(ctx, dotScaled.getBElemType());
    withScale = true;
    bothOperandsAreE2M1 =
        dotScaled.getAElemType() == ScaleDotElemType::E2M1 &&
        dotScaled.getBElemType() == ScaleDotElemType::E2M1;
    // Get the logical M/N/K dimension (considering fp4 packing).
    if (dotScaled.getAElemType() == ScaleDotElemType::E2M1 && dotScaled.getLhsKPack()) {
        inputKDim *= 2;
    }
  }
  int numWarps = ttg::lookupNumWarps(dot.getOperation());
  // Default values if MatrixLoadInfo is not available
  unsigned mDim = 16, mKDim = withScale ? 32 : 16;
  unsigned nDim = 16, nKDim = withScale ? 32 : 16;
  unsigned minMfmaInstKReq = std::max(mKDim, nKDim);
  SmallVector<std::pair<unsigned, unsigned>> aTileCands;
  SmallVector<std::pair<unsigned, unsigned>> bTileCands;
  if (aMatInfo.has_value() && !aMatInfo.value().elemsTileCands.empty())
    aTileCands = aMatInfo.value().elemsTileCands;
  else
    aTileCands.push_back({mDim, mKDim});
  if (bMatInfo.has_value() && !bMatInfo.value().elemsTileCands.empty())
    bTileCands = bMatInfo.value().elemsTileCands;
  else
    bTileCands.push_back({nDim, nKDim});
  bool foundValidTile = false;
  int64_t bestCover = -1; // maximize aM * bN * numWarps under M * N.
  for (const auto &[aM, aK] : aTileCands) {
    for (const auto &[bN, bK] : bTileCands) {
      if (aM > static_cast<unsigned>(M) || bN > static_cast<unsigned>(N))
        continue;
      unsigned maxKTile = aK > bK ? aK : bK;
      unsigned minKTile = aK < bK ? aK : bK;
      if (maxKTile > static_cast<unsigned>(inputKDim))
        continue;
      if (minKTile < minMfmaInstKReq)
        continue;
      if (M % aM != 0 || N % bN != 0 || inputKDim % aK != 0 ||
          inputKDim % bK != 0)
        continue;
      int64_t cover = static_cast<int64_t>(aM) * static_cast<int64_t>(bN) *
                      static_cast<int64_t>(numWarps);
      if (cover > M * N && foundValidTile)
        continue;
      if (!foundValidTile || cover > bestCover ||
          (cover == bestCover && (aM > mDim || (aM == mDim && bN > nDim)))) {
        foundValidTile = true;
        bestCover = cover;
        mDim = aM;
        mKDim = aK;
        nDim = bN;
        nKDim = bK;
      }
    }
  }
  if (!foundValidTile) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "[chooseMfmaInstructionWithMLS::"
       << (withScale ? "DotScaledOp" : "DotOp")
       << "] no valid combined tile found, "
       << "M=" << M << ", N=" << N << ", K=" << inputKDim
       << ", numWarps=" << numWarps
       << ", aElemType=" << aElemType << " (" << aElemType.getIntOrFloatBitWidth() << "-bit)"
       << ", bElemType=" << bElemType << " (" << bElemType.getIntOrFloatBitWidth() << "-bit)"
       << ", aCands=[";
    for (size_t i = 0; i < aTileCands.size(); ++i) {
      os << "{nonKDim=" << aTileCands[i].first
         << ", KDim=" << aTileCands[i].second << "}";
      if (i + 1 < aTileCands.size())
        os << ", ";
    }
    os << "], bCands=[";
    for (size_t i = 0; i < bTileCands.size(); ++i) {
      os << "{nonKDim=" << bTileCands[i].first
         << ", KDim=" << bTileCands[i].second << "}";
      if (i + 1 < bTileCands.size())
        os << ", ";
    }
    os << "]";
    return emitError(dot.getLoc(), os.str());
  }
  FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic = failure();
  auto arch = getAMDArch(dot.getOperation()->getParentOfType<ModuleOp>());
  assert(arch.has_value() && "expected arch");
  if (withScale) {
    auto kWidth = 8;
    if (bothOperandsAreE2M1) {
      bool bothMatrixLoadsUseB4 =
          aMatInfo.has_value() && bMatInfo.has_value() &&
          aMatInfo.value().mlsElemBitTyKind == MlsElemBitTyKind::B4 &&
          bMatInfo.value().mlsElemBitTyKind == MlsElemBitTyKind::B4;
      kWidth = getMlsDataFlowInfo(bothMatrixLoadsUseB4 ? MlsElemBitTyKind::B4
                                                       : MlsElemBitTyKind::B4Mix)
                   .getMfmaKWidth(8);
    } else {
      kWidth = 8;
    }
    inputKDim = std::min<unsigned>(inputKDim, 64/16 * kWidth);
  }
  unsigned selectedKTile = std::numeric_limits<unsigned>::max();
  if (aMatInfo.has_value())
    selectedKTile = std::min(selectedKTile, mKDim);
  if (bMatInfo.has_value())
    selectedKTile = std::min(selectedKTile, nKDim);
  if (selectedKTile != std::numeric_limits<unsigned>::max())
    inputKDim = std::min<unsigned>(inputKDim, selectedKTile);
  maybeMfmaIntrinsic = MfmaIntrinsic::selectFor(
      dot.getLoc(), mfmaVersion, 16, 16, inputKDim, aElemType, bElemType,
      /*withScale=*/withScale, /*allowXF32=*/false, features);
  if (failed(maybeMfmaIntrinsic))
    return emitError(dot.getLoc(), withScale
                                       ? "no matching scaled matrix core intrinsic"
                                       : "no matching matrix core intrinsic due to unsupported element type");
  auto kDim = maybeMfmaIntrinsic->kDim;
  if (kDim == 0) {
    mlir::emitRemark(dot.getLoc())
        << "Selected MFMA intrinsic '" << maybeMfmaIntrinsic->name
        << "' is invalid for MLS dot because it reports kDim=0.";
    return failure();
  }
  if (M % mDim != 0 || N % nDim != 0) {
    mlir::emitRemark(dot.getLoc())
        << "Unable to use MFMA intrinsic '" << maybeMfmaIntrinsic->name
        << "' for MLS dot because the selected tiles do not divide the result "
           "shape: result-shape=("
        << M << "x" << N << "), selected-tiles=(" << mDim << "x" << nDim
        << "), remainders=(" << (M % mDim) << "x" << (N % nDim) << ").";
    return failure();
  }
  if (inputKDim % kDim != 0) {
    mlir::emitRemark(dot.getLoc())
        << "Unable to select MFMA intrinsic '" << maybeMfmaIntrinsic->name
        << "' as MFMA intrinsic k-dimension size kDim=" << kDim
        << ", which is not a multiple of tile k-dimension size inputKDim="
        << inputKDim
        << ". Using this intrinsic would introduce data duplication.";
    return failure();
  }
  // Store the chosen dimensions back to MatrixLoadInfo
  if (aMatInfo.has_value())
    aMatInfo.value().choosedElemsTile = {mDim, mKDim};
  if (bMatInfo.has_value())
    bMatInfo.value().choosedElemsTile = {nDim, nKDim};
  return maybeMfmaIntrinsic;
}
// Chooses a proper MFMA instruction that can used to compute the given dot op.
// If enforcedNonKDim is not zero, it will be used to overwrite the default
// logic to choose a MFMA with matching M/N dim.
FailureOr<MfmaIntrinsic>
chooseMfmaInstruction(Location loc, int mfmaVersion, RankedTensorType cType,
                      Type aElemType, Type bElemType, int inputKSize,
                      int enforcedNonKDim, bool withScale, bool allowXF32,
                      HCUISAFeature features = HCUISAFeature::NONE,
                      StringRef arch = llvm::StringRef()) {
  // number of matrix elements along k dim per one MFMA instruction
  unsigned kDim = 0;
  auto resShape = cType.getShape();
  auto rank = resShape.size();
  auto M = resShape[rank - 2];
  auto N = resShape[rank - 1];
  unsigned mDim = 0;
  unsigned nDim = 0;
  if (enforcedNonKDim != 0) {
    mDim = nDim = enforcedNonKDim;
  } else {
    int minSize = std::min(M, N);
    // HCUs only feature 16x16xk matrix core instructions for now.
    if (minSize >= 16) {
      mDim = 16;
      nDim = 16;
    }
  }
  if (mDim == 0 || nDim == 0)
    return failure();
  FailureOr<MfmaIntrinsic> maybeMfmaIntrinsic =
      MfmaIntrinsic::selectFor(loc, mfmaVersion, mDim, nDim, inputKSize,
                               aElemType, bElemType, withScale, allowXF32,
                               features);
  // Fallback to FMA if the M/N dim is not supported by MFMA.
  if (failed(maybeMfmaIntrinsic)) {
    mlir::emitRemark(loc) << "Unable to select MFMA intrinsic for the request: "
                          << "version=" << mfmaVersion << ", result-shape=("
                          << M << "x" << N << "), selected-tiles=(" << mDim
                          << "x" << nDim << "), inputKSize=" << inputKSize
                          << ", aElemType=" << aElemType
                          << ", bElemType=" << bElemType
                          << ", withScale=" << (withScale ? "true" : "false")
                          << ", allowXF32=" << (allowXF32 ? "true" : "false")
                          << (enforcedNonKDim != 0
                                  ? (llvm::Twine(", enforcedNonKDim=") +
                                     llvm::Twine(enforcedNonKDim))
                                        .str()
                                  : "");
    return failure();
  }
  kDim = maybeMfmaIntrinsic->kDim;
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // If inputKSize % kDim != 0 (including the case where inputKSize < kDim),
  // this layout will introduce data duplication.
  if (inputKSize % kDim != 0) {
    mlir::emitRemark(loc)
        << "Unable to select MFMA intrinsic '" << maybeMfmaIntrinsic->name
        << "' as MFMA intrinsic k-dimension size kDim=" << kDim
        << ", which is not a multiple of tile k-dimension size inputKSize="
        << inputKSize
        << ". Using this intrinsic would introduce data duplication.";
    return failure();
  }
  return maybeMfmaIntrinsic;
}
FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotOp dot, int mfmaVersion,
                                               int nonKDim,
                                               bool withScale = false,
                                               HCUISAFeature features = HCUISAFeature::NONE) {
  RankedTensorType aType = dot.getA().getType();
  bool allowXF32 =
      dot.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  return chooseMfmaInstruction(
    dot.getLoc(), mfmaVersion, dot.getC().getType(), aType.getElementType(),
      dot.getB().getType().getElementType(), aType.getShape().back(), nonKDim,
      withScale, allowXF32, features);
}
FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotScaledOp dot,
                                               int mfmaVersion, int nonKDim) {
  using ::mlir::LLVM::AMD::scaleDotElemTypeToMLIRType;
  auto ctx = dot.getContext();
  int64_t inputKDim = dot.getA().getType().getShape().back();
  llvm::dbgs() << "[chooseMfmaInstruction] inputKDim before : " << inputKDim << "\n";
  if (dot.getAElemType() == ScaleDotElemType::E2M1 && dot.getLhsKPack()) {
    // Since two fp4 are packed into int8, to get the correct K dim size, we
    // need to multiply it by 2.
    inputKDim *= 2;
  }
  Type aElemType = scaleDotElemTypeToMLIRType(ctx, dot.getAElemType());
  Type bElemType = scaleDotElemTypeToMLIRType(ctx, dot.getBElemType());
  auto arch = getAMDArch(dot->getParentOfType<ModuleOp>());
  assert(arch.has_value() && "expected arch");
  return chooseMfmaInstruction(dot.getLoc(), mfmaVersion, dot.getC().getType(),
                               aElemType, bElemType, inputKDim, nonKDim,
                               /*withScale=*/true, /*allowXF32=*/false,
                               /*features=*/HCUISAFeature::NONE, *arch);
}
FailureOr<MfmaIntrinsic> chooseMfmaInstruction(tt::DotScaledOp dot,
                                               int mfmaVersion, int nonKDim,
                                               bool useFp16) {
  // For scaled dot, we handle it with fp16 or bf16 emulation for now.
  Builder b(dot.getContext());
  Type elemType = useFp16 ? b.getF16Type() : b.getBF16Type();
  return chooseMfmaInstruction(dot.getLoc(), mfmaVersion, dot.getC().getType(),
                               elemType, elemType,
                               dot.getA().getType().getShape().back(), nonKDim,
                               /*withScale=*/false, /*allowXF32=*/false);
}
using OperandTypesVector = SmallVector<Type, 4>;
OperandTypesVector
selectMatrixCoreOperandTypes(tt::DotOp dot,
                             ArrayRef<OperandTypesVector> applicableTypes) {
  SmallVector<Value> dotOperands = {dot.getA(), dot.getB(), dot.getC(),
                                    dot.getD()};
  OperandTypesVector initElemTypes;
  llvm::transform(dotOperands, std::back_inserter(initElemTypes), [](Value v) {
    return cast<RankedTensorType>(v.getType()).getElementType();
  });
  // Use simple costmodel to define optimal set of the dot operands.
  // Most expensive - accuracy loss conversions:
  //   - any larger type -> any smaller type;
  //   - float -> int;
  //   - int -> float (not supported for now);
  //   - signed int -> unsigned int;
  //   - unsigned int -> signed int with same or less size.
  // They are never performed, better to use FMA.
  // Supported conversion for now costs `1`, no conversion costs `0`.
  // The model could be improved in the future. For example taken into account
  // chain dot could be detected and result conversion score is decreased.
  int maxConvertCost =
      std::numeric_limits<int32_t>::max() / applicableTypes.front().size();
  auto calcConvertCost = [&](Type fromTy, Type toTy) -> int32_t {
    if (fromTy == toTy)
      return 0;
    // Skip conversion between int and float. Int16/int32 cases are lowered to
    // FMA.
    if (fromTy.isIntOrIndex() != toTy.isIntOrIndex())
      return maxConvertCost;
    if (fromTy.isIntOrIndex() && toTy.isIntOrIndex() &&
        fromTy.isUnsignedInteger() != toTy.isUnsignedInteger())
      return fromTy.isUnsignedInteger() && fromTy.getIntOrFloatBitWidth() <
                                               toTy.getIntOrFloatBitWidth()
                 ? 1
                 : maxConvertCost;
    return fromTy.getIntOrFloatBitWidth() <= toTy.getIntOrFloatBitWidth()
               ? 1
               : maxConvertCost;
  };
  auto minCost = maxConvertCost;
  auto optTypes = OperandTypesVector();
  for (auto types : applicableTypes) {
    assert(types.size() == initElemTypes.size());
    int accumulatedConvertCost = 0;
    for (int i = 0; i < initElemTypes.size(); ++i) {
      accumulatedConvertCost += calcConvertCost(initElemTypes[i], types[i]);
    }
    if (accumulatedConvertCost < minCost) {
      minCost = accumulatedConvertCost;
      optTypes = types;
    }
  }
  return optTypes;
}
OperandTypesVector getOperandTypesForWmmaOp(PatternRewriter &rewriter,
                                            tt::DotOp dot, int version) {
  Type f16 = rewriter.getF16Type();
  Type f32 = rewriter.getF32Type();
  Type bf16 = rewriter.getBF16Type();
  Type i8 = rewriter.getIntegerType(8);
  Type i32 = rewriter.getIntegerType(32);
  SmallVector<OperandTypesVector> applicableTypes = {
      // clang-format off
      {f16, f16, f32, f32},
      {bf16, bf16, f32, f32},
      {i8, i8, i32, i32},
      // {f16, f16, f16, f16},
      // {bf16, bf16, bf16, bf16},
      // {i4, i4, i32, i32} - are supported configurations
      // by WMMA instruction, but not supported by triton
      // clang-format on
  };
  if (version == 2 || version == 3) {
    Type fp8e4nv = rewriter.getType<Float8E4M3FNType>();
    Type fp8e5 = rewriter.getType<Float8E5M2Type>();
    applicableTypes.append({
        // clang-format off
        {fp8e4nv, fp8e4nv, f32, f32},
        {fp8e4nv, fp8e5, f32, f32},
        {fp8e5, fp8e4nv, f32, f32},
        {fp8e5, fp8e5, f32, f32},
        // clang-format on
    });
  }
  return selectMatrixCoreOperandTypes(dot, applicableTypes);
}
//===---------------------------------------------------------------------===//
// @brief Convert layout and cast element type of a given tensor
//
// If old element type is different from new element type, this function
// creates two new operations:
// 1. %converted_value = layout_convert %value, newEncoding
// 2. %casted_value = cast(fext, ftrunc, etc.) %value, newElemType
//
// If old element type is same as new element type, this function creates only
// one operation: %converted_value = layout_convert %value, newEncoding
//
// @param rewriter
// @param value original tensor value, which we need to convert and cast
// @param newEncoding new encoding for the tensor
// @param newElemType new element type for the tensor
// @return converted and optionally casted tensor value
//===---------------------------------------------------------------------===//
Value convertAndCastTensor(PatternRewriter &rewriter, Value value,
                           Attribute newEncoding, Type newElemType,
                           std::optional<ttg::DotOperandEncodingAttr> compatibleEncoding = std::nullopt) {
  assert(newElemType.isIntOrFloat());
  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();
  assert(oldElemType.isIntOrFloat());
  assert(oldElemType.isIntOrIndex() == newElemType.isIntOrIndex());
  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);
  Value convertedTensor =
      ttg::ConvertLayoutOp::create(rewriter, loc, convertedType, value);
  if (compatibleEncoding.has_value()) {
    convertedType = RankedTensorType::get(oldType.getShape(), oldElemType,
                                          compatibleEncoding.value());
    convertedTensor = ttg::ConvertLayoutOp::create(
        rewriter, loc, convertedType, convertedTensor);
  }
  if (newElemType == oldElemType)
    return convertedTensor;
  Type castedType = convertedType.cloneWith(std::nullopt, newElemType);
  Value castedTensor;
  if (newElemType.isIntOrIndex()) {
    unsigned oldWidth = oldElemType.getIntOrFloatBitWidth();
    unsigned newWidth = newElemType.getIntOrFloatBitWidth();
    if (oldWidth == newWidth)
      castedTensor = arith::BitcastOp::create(rewriter, loc, convertedType,
                                                       convertedTensor);
    else if (oldWidth > newWidth)
      castedTensor =
          arith::TruncIOp::create(rewriter, loc, castedType, convertedTensor);
    else if (oldElemType.isSignedInteger())
      castedTensor =
          arith::ExtSIOp::create(rewriter, loc, castedType, convertedTensor);
    else
      castedTensor =
          arith::ExtUIOp::create(rewriter, loc, castedType, convertedTensor);
  } else {
    if (oldElemType.isF16() && newElemType.isF32())
      castedTensor =
          arith::ExtFOp::create(rewriter, loc, castedType, convertedTensor);
    else if (oldElemType.isF32() && newElemType.isF16())
      castedTensor =
          arith::TruncFOp::create(rewriter, loc, castedType, convertedTensor);
    else
      castedTensor =
          tt::FpToFpOp::create(rewriter, loc, castedType, convertedTensor);
  }
  return castedTensor;
}
Value findScaleAsDecompositionSource(Value v) {
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  SetVector<Operation *> slice;
  (void)getBackwardSlice(v, &slice, options);
  for (auto &op : slice) {
    if (op->hasAttrOfType<BoolAttr>(AttrDecomposedDotScaledSource)) {
      op->removeAttr(AttrDecomposedDotScaledSource);
      return op->getResult(0);
    }
  }
  return {};
}
// Figure out the best tilesPerWarp that gives largest vector size for |scale|
// tensors feeding into dot_scaled op.
SmallVector<unsigned, 2> deduceTilesPerWarpForScale(
    TypedValue<RankedTensorType> scaleA, TypedValue<RankedTensorType> scaleB,
    unsigned nonKDim, unsigned m, unsigned n, ArrayRef<unsigned> warpsPerCTA) {
  // Source code have flexibility to preshuffle scale tensor to achieve better
  // global load vectorization. That preshuffle scheme is conveyed via some
  // tl.reshape and tl.trans op combinations. Instead of hardcoding one case or
  // pattern match the op chain here, we try certain scale tensor layouts and
  // see which one gives us better vectorization when pushed upwards to the
  // global load.
  auto inferScaleSrcVecSize =
      [&](unsigned opIdx, TypedValue<RankedTensorType> scale,
          SmallVector<unsigned, 2> tilesPerWarp) -> unsigned {
    if (!scale)
      return 1;
    LinearLayout layout = ttg::chooseScaledMfmaScaleLayout(
        scale.getContext(), opIdx, scale.getType().getShape(), nonKDim,
        tilesPerWarp, warpsPerCTA);
    LLVM_DEBUG(llvm::dbgs() << "trying scale layout: " << layout << "\n");
    auto scaleDef = scale.getDefiningOp();
    // assume vec=4 for constant scale
    if (isa_and_nonnull<arith::ConstantOp, triton::SplatOp>(scaleDef))
      return 4;
    // Infer source layout used for global load using the current scale layout.
    auto loadLayoutPair = ttg::inferSourceLoadLayout(layout, scaleDef);
    if (!loadLayoutPair)
      return 1;
    tt::LoadOp loadOp = loadLayoutPair->first;
    const LinearLayout &inferredLayout = loadLayoutPair->second;
    LLVM_DEBUG(llvm::dbgs()
               << "inferred load layout: " << inferredLayout << "\n");
    auto loadType = cast<RankedTensorType>(loadOp.getType());
    auto loadOrder = ttg::getOrder(loadType);
    auto loadCTALayout = ttg::getCTALayout(loadType.getEncoding());
    // Reuse existing shared memory vectorization utilities by constructing a
    // pass through layout that does linear element mapping.
    MLIRContext *context = scale.getContext();
    auto passThruShared = ttg::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, loadOrder, loadCTALayout);
    auto sharedLL =
        triton::gpu::toLinearLayout(loadType.getShape(), passThruShared);
    auto composedLL = inferredLayout.invertAndCompose(sharedLL).flattenOuts();
    LLVM_DEBUG(llvm::dbgs()
               << "inferred composed layout: " << composedLL << "\n");
    auto [v, _] =
        largestVectorisation(context, composedLL, /*bitwidth=*/8, std::nullopt);
    return v;
  };
  unsigned largest = 2;
  SmallVector<unsigned, 2> chosen{1, 1};
  // For scaled MFMA intrinsic, each thread only reads one i8 value.
  // For better vectorization, we prefer to stick tilesPerWarp 2x2 for 16x16x128
  // and 1x1 for 32x32x64 so that each thread can read 4xi8 values.
  // limit tilesPerWarp to block boundary
  for (unsigned mDimTiles = 1; mDimTiles <= std::min(2u, m / nonKDim);
       mDimTiles++) {
    for (unsigned nDimTiles = 1; nDimTiles <= std::min(2u, n / nonKDim);
         nDimTiles++) {
      SmallVector<unsigned, 2> tilesPerWarp{mDimTiles, nDimTiles};
      unsigned vecSizeA = inferScaleSrcVecSize(0, scaleA, tilesPerWarp);
      unsigned vecSizeB = inferScaleSrcVecSize(1, scaleB, tilesPerWarp);
      LLVM_DEBUG(llvm::dbgs() << "when tilesPerWarp: " << tilesPerWarp[0]
                              << ", " << tilesPerWarp[1] << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "inferred scaleA vecSize: " << vecSizeA << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "inferred scaleB vecSize: " << vecSizeB << "\n");
      unsigned score = vecSizeA + vecSizeB;
      if (score > largest) {
        largest = score;
        chosen = tilesPerWarp;
      }
    }
  }
  assert(largest <= 8 && "at most pack 4 scales for scale a & b respectively");
  // fixup: align with dimension that has scale
  if (!scaleA && scaleB)
    chosen[0] = std::min(ceil<unsigned>(m, nonKDim), chosen[1]);
  if (!scaleB && scaleA)
    chosen[1] = std::min(ceil<unsigned>(n, nonKDim), chosen[0]);
  return chosen;
}
class BlockedToMFMA : public OpRewritePattern<tt::DotOp> {
  int mfmaVersion;
  int nonKDim;
  int kPack;
  HCUISAFeature features;
  MmacLayout mmacLayout;  // Added for HCUs
public:
  BlockedToMFMA(MLIRContext *context, int mfmaVersion, int nonKDim, int kPack,
                MmacLayout mmacLayout, HCUISAFeature features,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), mfmaVersion(mfmaVersion),
        nonKDim(nonKDim), kPack(kPack),
        features(features), mmacLayout(mmacLayout) {}
  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    using TensorValue = TypedValue<RankedTensorType>;
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();
    if (!isa_and_nonnull<BlockedEncodingAttr>(dotOp.getType().getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");
    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());
    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    int numWarps = ttg::lookupNumWarps(dotOp);
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto ctx = oldAType.getContext();
    Type aElemType = oldAType.getElementType();
    Type bElemType = oldBType.getElementType();
    // bool withScale =
    //     mfmaVersion == 4 && isF8F6F4(aElemType) && isF8F6F4(bElemType);
    bool withScale = false;
    auto [aMatInfo, bMatInfo] = getDotOperandMatrixLoadInfo(dotOp, features);
    bool useMatrixLoad = aMatInfo.has_value() || bMatInfo.has_value();
    auto wdraPlan = HCU::getWdraSplitPlan(dotOp);
    FailureOr<MfmaIntrinsic> mfmaInstr;
    if (useMatrixLoad) {
      mfmaInstr = chooseMfmaInstructionWithMLS(dotOp, mfmaVersion, nonKDim,
                                               aMatInfo, bMatInfo, features,
                                               wdraPlan ? &*wdraPlan : nullptr);
    } else {
      // If mfmaVersion == 4 and both inputs are of F8F6F4 types, we will try to
      // use the V_MFMA_*_F8F6F4 instructions since it has higher FLOPs per cycle.
      // If we can't find a proper instruction, we will fall back to select from
      // normal mfma instructions.
      mfmaInstr = chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, withScale, features);
      if (failed(mfmaInstr)) {
        if (!withScale) {
        return rewriter.notifyMatchFailure(
            dotOp,
            "Unable to choose preferable MFMA intrinsic for dot operation.");
        }
        mfmaInstr = chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, false, features);
        if (failed(mfmaInstr)) {
          return rewriter.notifyMatchFailure(
              dotOp, "Unable to choose MFMA intrinsic for dot operation.");
        }
        withScale = false;
      }
    }
    auto mDim = mfmaInstr->mDim;
    auto nDim = mfmaInstr->nDim;
    auto kDim = mfmaInstr->kDim;
    auto kBase = mfmaInstr->kBase;
    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, retShape, numWarps, {mDim, nDim});
    unsigned rank = oldRetType.getRank();
    SmallVector<unsigned> tilesPerWarp = {1, 1};
    // if (mDim == nDim && (mDim == 32 || mDim == 16)) {
    //   Value scaleA = findScaleAsDecompositionSource(a);
    //   Value scaleB = findScaleAsDecompositionSource(b);
    //   tilesPerWarp = deduceTilesPerWarpForScale(
    //       dyn_cast_if_present<TensorValue>(scaleA),
    //       dyn_cast_if_present<TensorValue>(scaleB), mDim, retShape[0],
    //       retShape[1], warpsPerTile);
    // }
    bool hasPreShuffledScale = (tilesPerWarp[0] > 1 && tilesPerWarp[1] > 1);
    if (rank == 3) {
      tilesPerWarp.insert(tilesPerWarp.begin(), 1);
    }
    if (useMatrixLoad) {
      auto instM = mfmaInstr->mDim;
      auto instN = mfmaInstr->nDim;
      auto tileM = aMatInfo.has_value() ? aMatInfo.value().choosedElemsTile.first : instM;
      auto tileN = bMatInfo.has_value() ? bMatInfo.value().choosedElemsTile.first : instN;
      assert(tileM >= instM && tileN >= instN && tileM % instM == 0 && tileN % instN == 0);
      bool hasBatchDim = rank == 3;
      int mIndex = 0 + hasBatchDim;
      int nIndex = 1 + hasBatchDim;
      tilesPerWarp[mIndex] = tileM/instM;
      tilesPerWarp[nIndex] = tileN/instN;
      warpsPerTile = warpsPerTileMFMA(dotOp, retShape, numWarps, {tileM, tileN});
    }

    // ds_read_m on buffer-load B (not MLS): tileN=32 panel, same split as MLS.
    // b16 -> K=16; b8 -> K=32.
    bool dsReadMDot = false;
    unsigned bits =
        aElemType.isIntOrFloat() ? aElemType.getIntOrFloatBitWidth() : 0;
    unsigned dsKPerTile = bits == 16 ? 16 : bits == 8 ? 32 : 0;
    int64_t bK = oldBType.getShape()[0];
    int64_t bN = oldBType.getShape()[1];
    if (isDsReadMEnabled() && !useMatrixLoad && rank == 2 && mDim == 16 &&
        nDim == 16 && dsKPerTile && kDim == dsKPerTile &&
        bElemType.isIntOrFloat() &&
        bElemType.getIntOrFloatBitWidth() == bits && bK >= dsKPerTile &&
        bK % dsKPerTile == 0 && bN >= 32 && bN % 32 == 0 &&
        bN == retShape[1] && oldAType.getShape()[1] == bK) {
      unsigned tileM = mDim, tileN = 32;
      tilesPerWarp = {tileM / mDim, tileN / nDim}; // {1, 2}
      warpsPerTile =
          warpsPerTileMFMA(dotOp, retShape, numWarps, {tileM, tileN});
      dsReadMDot = true;
    }

    Type mfmaAccType;
    if (oldRetType.getElementType().isIntOrIndex())
      mfmaAccType = rewriter.getIntegerType(32);
    else if (oldRetType.getElementType().isF64())
      mfmaAccType = rewriter.getF64Type();
    else
      mfmaAccType = rewriter.getF32Type();
    // HCU MMAC only supports 16x16 instructions (no 4x64/64x4/32x32),
    // so isTransposed is always true as the 4x64 exception never applies.
    bool isTransposed = true;
    auto aElemTy = mfmaInstr->aElementType;
    auto is16BitElemTy = (aElemTy.isF16() || aElemTy.isBF16());
    // Match AMD chained-dot transpose selection, but materialize the choice
    // through mmacLayout on HCU instead of encoding isTransposed=true.
    if (!dsReadMDot && mfmaVersion >= 3 && is16BitElemTy &&
        mDim == 16 && nDim == 16 && rank == 2 && !hasPreShuffledScale &&
        !useMatrixLoad && (features & HCUISAFeature::MMAC_LAYOUT) != 0) {
      if (isChainDotHead(dotOp, 0u) &&
          retShape.front() >= 16 * 2 * warpsPerTile.front() &&
          retShape.back() == 16 && warpsPerTile.back() == 1) {
        isTransposed = true;
        tilesPerWarp = {2, 1};
      } else if (isChainDotHead(dotOp, 1u) && retShape.front() == 16 &&
                 retShape.back() >= 16 * 2 * warpsPerTile.back() &&
                 warpsPerTile.front() == 1) {
        isTransposed = false;
        tilesPerWarp = {1, 2};
      }
    }
    auto effectiveMmacLayout =
        mmacLayout == MmacLayout::MFMA
            ? getMfmaMappedMmacLayout(
                  isTransposed,
                  (features & HCUISAFeature::MMAC_LAYOUT) != 0)
            : mmacLayout;
    ttg::AMDMfmaEncodingAttr mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        oldRetType.getContext(), /*verison=*/mfmaVersion, warpsPerTile,
        {mDim, nDim, kDim}, /*isTransposed=*/false, CTALayout,
        tilesPerWarp, mfmaAccType.getIntOrFloatBitWidth(),
        effectiveMmacLayout);
    // convert accumulator
    auto oldAcc = dotOp.getC();
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, mfmaEnc, mfmaAccType);
    // Here is a brief explanation of kWidth, kBase, and kDim
    // 1. kWidth: the number of **consecutive** elements each thread loads from
    //    shared memory in preparation for mfma instructions. In theory, each
    //    thread can issue multiple ds_read to load elements from non-contiguous
    //    addresses in shared memory for one mfma instruction, but that won't be
    //    good for performance. So in practice for better vectorization, we
    //    make sure the kWidth elements can be loaded from shared memory by a
    //    single ds_read instruction by setting vecSize of the sharedLayout
    //    to be kWidth.
    // 2. kDim: the k dimension size of the mfma instruction. E.g. instruction
    //    mfma_32x32x16 has kDim = 16, meaning this mfma instruction can compute
    //    a matmul of operands with shape 32x16 and 16x32.
    // 3. kBase: the number of elements each thread holds for a single mfma
    //    instruction.
    // 4. relation between kBase and kDim:
    //    4.1 For mfma_32, kBase = kDim / 2
    //    4.2 For mfma_16, kBase = kDim / 4
    //    4.3 For mfma_4, kBase = kDim / 16
    // 5. relation between kWidth and kBase: For now it supports two cases
    //    5.1 kWidth = kBase, i.e. kPack = 1. In this case, each load from
    //        shared memory results in one mfma instruction.
    //    5.2 kWidth = 2 * kBase, i.e. kPack = 2. In this case, each load from
    //        shared memory results in two mfma instructions, since one mfma
    //        can only consume kBase elements from each thread.
    //    Note that we cannot have larger kPack since kPack = 2 means
    //    ds_read_b128, which is the largest vector size for shared memory load.
    auto kWidth = kBase;
    // We want to extend kWidth by kPack (kPack=1 means no extension)
    // to increase ds_read vector size
    // However, in FA, the second dot can only use kWidth = kBase since it's
    // limited by the result of the first dot, which is of mfmaLayout.
    auto isDotChainTail = isChainDotTail(dotOp);
    if (!isDotChainTail)
      kWidth *= kPack;
    // For FA fwd kernel with f16 elementTy, we limit the 2nd dot to have
    // kWidth = 4 so that the coversion from #mma (result of 1st dot)
    // to #dotOp (operand 0 of 2nd dot) is a no-op.
    // TODO (lixun): relax the condition for 8-bit elementTy.
    if (is16BitElemTy && isDotChainTail) {
      kWidth = 4;
    }
    // TODO: HCU need check!
    // // For FA bwd kernel (detected using hasTransInDefChain), depending on
    // // whether the dot is a head or tail in the chain, we adjust the kWidth
    // // accordingly. This will enable us to create the same shared encoding per
    // // pair of tt.dot ops that both use the same tt.load result, one directly
    // // and one via tt.trans, later in the pass pipeline.
    // if (is16BitElemTy && hasTransInDefChain(dotOp, 1u)) {
    //   if (isChainDotHead(dotOp)) {
    //     kWidth = 4;
    //   } else if (isDotChainTail) {
    //     kWidth = 8;
    //   }
    // }
    Value newDot;
    if (withScale) {
      // If a scaled mfma instruction is chosen, we will rewrite the DotOp to a
      // DotScaledOp.
      auto aScaledElemTy = mlirTypeToScaledElemType(aElemType);
      auto bScaledElemTy = mlirTypeToScaledElemType(bElemType);
      if (failed(aScaledElemTy) || failed(bScaledElemTy))
        return failure();
      assert(kWidth == 32);
      auto newAEncoding =
          DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth / 2);
      auto newBEncoding =
          DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth / 2);
      a = convertAndCastTensor(rewriter, a, newAEncoding,
                               mfmaInstr->aElementType);
      b = convertAndCastTensor(rewriter, b, newBEncoding,
                               mfmaInstr->bElementType);
      newDot = triton::DotScaledOp::create(
          rewriter, dotOp.getLoc(), newAcc.getType(), a, b, newAcc, Value(),
          Value(), aScaledElemTy.value(), bScaledElemTy.value(),
          /*fastMath=*/false);
    } else {
      auto newAEncoding =
          ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc,
                                           aMatInfo.has_value() ? aMatInfo.value().dotMfmaKWidth : kWidth);
      auto newBEncoding =
          ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc,
                                           bMatInfo.has_value() ? bMatInfo.value().dotMfmaKWidth : kWidth);
      auto compatibleAEncoding = getCompatibleDotOpEncoding(0, newAEncoding,
                                                           newBEncoding, aMatInfo, bMatInfo);
      auto compatibleBEncoding = getCompatibleDotOpEncoding(1, newAEncoding,
                                                           newBEncoding, aMatInfo, bMatInfo);
      a = convertAndCastTensor(rewriter, a, newAEncoding,
                               mfmaInstr->aElementType, compatibleAEncoding);
      b = convertAndCastTensor(rewriter, b, newBEncoding,
                               mfmaInstr->bElementType, compatibleBEncoding);
      newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newAcc.getType(), a,
                                          b, newAcc, dotOp.getInputPrecision(),
                                          dotOp.getMaxNumImpreciseAcc());
    }
    if (wdraPlan)
      HCU::setWdraSplitPlan(newDot.getDefiningOp(), *wdraPlan);
    Value dotOutput =
        convertAndCastTensor(rewriter, newDot, oldRetType.getEncoding(),
                             oldRetType.getElementType());
    rewriter.replaceOp(dotOp, dotOutput);
    return success();
  }
};
class ScaledBlockedToMFMA final : public OpRewritePattern<triton::DotScaledOp> {
  int mfmaVersion;
  int nonKDim;
  int kPack;
  HCUISAFeature features;
  MmacLayout mmacLayout;
public:
  ScaledBlockedToMFMA(MLIRContext *context, int mfmaVersion, int nonKDim,
                      int kPack, MmacLayout mmacLayout, HCUISAFeature features,
                      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), mfmaVersion(mfmaVersion),
        nonKDim(nonKDim), kPack(kPack),
        features(features), mmacLayout(mmacLayout) {}
  LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    // TODO: add support for m/n packed formats.
    if (!dotOp.getLhsKPack() || !dotOp.getRhsKPack())
      return failure();
    using TensorValue = TypedValue<RankedTensorType>;
    RankedTensorType oldRetType = dotOp.getType();
    if (!isa_and_nonnull<BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");
    unsigned rank = oldRetType.getRank();
    if (rank == 3)
      return rewriter.notifyMatchFailure(dotOp, "NYI: 3d case");
    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    TensorValue aScale = dotOp.getAScale();
    TensorValue bScale = dotOp.getBScale();
    if (aScale && bScale)
      return rewriter.notifyMatchFailure(dotOp, "NYI: both LHS and RHS scale");
    ScaleDotElemType aElemType = dotOp.getAElemType();
    ScaleDotElemType bElemType = dotOp.getBElemType();
    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2 ||
             elemType == ScaleDotElemType::BF16 ||
             elemType == ScaleDotElemType::FP16;
    };
    if (!supportsTypes(aElemType) || !supportsTypes(bElemType))
      return rewriter.notifyMatchFailure(dotOp, "NYI: mxfp6 operand");
    MLIRContext *ctx = dotOp.getContext();
    auto moduleOp = dotOp->getParentOfType<ModuleOp>();
    int numWarps = ttg::lookupNumWarps(dotOp);
    ttg::CTAEncodingAttr ctaLayout = ttg::getCTALayout(oldRetType.getEncoding());
    int numThreads = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    // Choose a suitable MFMA instruction for this scaled dot op.
    bool useFp16 = aElemType == ScaleDotElemType::FP16 ||
                   bElemType == ScaleDotElemType::FP16;
    FailureOr<MfmaIntrinsic> mfmaInstr =
        chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim, useFp16);
    if (failed(mfmaInstr))
      return rewriter.notifyMatchFailure(dotOp, "cannot choose mfma intrinsic");
    if (useFp16) {
      dotOp.emitRemark(
          "Warning: detected one dot_scaled operand is fp16 tensor so "
          "upcasting to fp16 for computation, which impacts precision; "
          "experimental behavior and may change in future");
    }
    unsigned mDim = mfmaInstr->mDim;
    unsigned nDim = mfmaInstr->nDim;
    unsigned kDim = mfmaInstr->kDim;
    unsigned kBase = mfmaInstr->kBase;
    // For mxfp4 A/B tensor, we pack every two values into one int8 value there.
    // For such cases, we have different initial kWidth for LHS and RHS, which
    // will be "fixed" later by using upcast_mxfp to convert LHS to unpacked
    // values. For such packed cases, we cannot support flexible kPack choices
    // from the developer--it just does not apply here. So mandate the choice
    // here.
    bool isAPacked = aElemType == ScaleDotElemType::E2M1;
    bool isBPacked = bElemType == ScaleDotElemType::E2M1;
    bool isPacked = isAPacked || isBPacked;
    unsigned kWidths[] = {isPacked ? (isAPacked ? 4 : 8) : kBase * kPack,
                          isPacked ? (isAPacked ? 8 : 4) : kBase * kPack};
    // For A/B tensor, 32 consecutive elements along K dim share the same scale.
    // We'd like to keep the scale values together with the base values in the
    // same warp to avoid cross-warp data exchange. It means we want warpsPerCTA
    // = 1 along the N/M dimension for the mxfp A/B case. We achieve that by
    // setting the M/N dimension as numWarps.
    SmallVector<unsigned, 2> mfmaWarpsPerCTA(rank, 1);
    mfmaWarpsPerCTA[aScale ? 0 : 1] = numWarps;
    auto effectiveMmacLayout =
        mmacLayout == MmacLayout::MFMA
            ? getMfmaMappedMmacLayout(
                  /*logicalIsTransposed=*/true,
                  (features & HCUISAFeature::MMAC_LAYOUT) != 0)
            : mmacLayout;
    auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        ctx, /*version=*/mfmaVersion, mfmaWarpsPerCTA, {mDim, nDim, kDim},
        /*isTransposed=*/false, ctaLayout, {},
        oldRetType.getElementType().getIntOrFloatBitWidth(),
        effectiveMmacLayout);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mfmaEnc);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        dotOp.getC().getLoc(), newRetType, dotOp.getC());
    auto upcastForMMA = [&](TensorValue v, int idx,
                            ScaleDotElemType type) -> TensorValue {
      auto vType = v.getType();
      auto newVEncoding = DotOperandEncodingAttr::get(
          ctx, idx, newRetType.getEncoding(), kWidths[idx]);
      auto newVType = RankedTensorType::get(
          vType.getShape(), vType.getElementType(), newVEncoding);
      v = rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
      // Don't need to covert int8 holding mxfp4--the upcast_mxfp op can
      // take int8 tensor as input.
      if (type == ScaleDotElemType::BF16 || type == ScaleDotElemType::FP16 ||
          type == ScaleDotElemType::E2M1 ||
          (mfmaVersion == 4 &&
           (type == ScaleDotElemType::E4M3 || type == ScaleDotElemType::E5M2)))
        return v;
      auto upcastedType = RankedTensorType::get(
          vType.getShape(),
          useFp16 ? rewriter.getF16Type() : rewriter.getBF16Type(),
          newVEncoding);
      return cast<TensorValue>(
          rewriter.create<FpToFpOp>(v.getLoc(), upcastedType, v).getResult());
    };
    a = upcastForMMA(a, 0, aElemType);
    b = upcastForMMA(b, 1, bElemType);
    // We need to have "matching" encoding between the main tensor and scale
    // tensor to make sure the scale values needed is in the same warp. So we
    // adopt the same CTA layout and warps per CTA. The warp dimensions needs to
    // match along M/N dimension too. With in a warp, we have 64 threads. We let
    // each thread read in one scale value. So we need a threadsPerWarp =
    // mDim/nDim along M/N dimension. Note that For MFMA intrinsics, mDim is
    // always the same as nDim. And for scaled dot scale tensor, we always have
    // K as the innermost dimension. So we have the same threadsPerWarp in the
    // below no matter A or B scale. Similarly for warpsPerCTA, the non-K
    // dimension is always at index 0.
    assert(mDim == nDim);
    SmallVector<unsigned, 2> threadsPerWarp = {mDim, numThreads / mDim};
    SmallVector<unsigned, 2> blockWarpsPerCTA(rank, 1);
    blockWarpsPerCTA[0] = numWarps;
    auto newScaleEncoding = triton::gpu::BlockedEncodingAttr::get(
        ctx, {1, 1}, threadsPerWarp, blockWarpsPerCTA, {1, 0}, ctaLayout);
    auto upcastMXFP = [&](TensorValue v, TensorValue scale,
                          ScaleDotElemType elemType, bool fastMath) -> Value {
      if (!scale)
        return v;
      auto newScaleType = RankedTensorType::get(
          scale.getType().getShape(), scale.getType().getElementType(),
          newScaleEncoding);
      auto convOp = rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(),
                                                          newScaleType, scale);
      Builder b(v.getContext());
      // TODO: Emit device assert to check scale tensor range fitting into fp16?
      Type outputElemType = useFp16 ? b.getF16Type() : b.getBF16Type();
      auto outputType =
          amdgpu::UpcastMXFPOp::deduceOutputType(v, elemType, outputElemType);
      return rewriter.create<amdgpu::UpcastMXFPOp>(
          dotOp.getLoc(), outputType, v, convOp, elemType, fastMath);
    };
    Value scaledA =
        upcastMXFP(a, aScale, dotOp.getAElemType(), dotOp.getFastMath());
    Value scaledB =
        upcastMXFP(b, bScale, dotOp.getBElemType(), dotOp.getFastMath());
    auto newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, scaledA,
                                         scaledB, newAcc);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot);
    return success();
  }
};
template <typename Op> Op getDefOpBeforeConvertLayout(Value op) {
  while (auto cvtOp = op.getDefiningOp<ttg::ConvertLayoutOp>()) {
    op = cvtOp.getSrc();
  }
  return op.getDefiningOp<Op>();
}
bool isScaleShuffled(Value scale) {
  if (!scale) {
    return false;
  }
  auto shape = cast<RankedTensorType>(scale.getType()).getShape();
  int rank = shape.size();
  int blockNonK = shape[rank - 2];
  // 1 scale always scales 32 elements along K dim
  int blockK = shape[rank - 1] * 32;
  auto reshapeOp2D = getDefOpBeforeConvertLayout<triton::ReshapeOp>(scale);
  if (!reshapeOp2D || reshapeOp2D.getType().getShape() != shape) {
    return false;
  }
  const std::array<int, 7> transposeOrder{0, 5, 3, 1, 4, 2, 6};
  auto transOp =
      getDefOpBeforeConvertLayout<triton::TransOp>(reshapeOp2D.getSrc());
  if (!transOp || transOp.getOrder() != ArrayRef<int>(transposeOrder)) {
    return false;
  }
  const std::array<int64_t, 7> reshape7DShape{
      blockNonK / 32, blockK / 32 / 8, 4, 16, 2, 2, 1};
  auto reshapeOp7D =
      getDefOpBeforeConvertLayout<triton::ReshapeOp>(transOp.getSrc());
  if (!reshapeOp7D ||
      reshapeOp7D.getType().getShape() != ArrayRef<int64_t>(reshape7DShape)) {
    return false;
  }
  return true;
}
SmallVector<unsigned, 2> getTilesPerWarp(Value aScale, Value bScale) {
  if (isScaleShuffled(aScale) || isScaleShuffled(bScale)) {
    return {2, 2};
  }
  return {1, 1};
}
class ScaledBlockedToScaledMFMAF8F6F4 final
    : public OpRewritePattern<triton::DotScaledOp> {
  int mfmaVersion;
  int nonKDim;
  MmacLayout mmacLayout;
  HCUISAFeature hcuFeatures;
public:
  ScaledBlockedToScaledMFMAF8F6F4(MLIRContext *context, int mfmaVersion,
                                  int nonKDim, MmacLayout mmacLayout,
                                  HCUISAFeature hcuFeatures,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), mfmaVersion(mfmaVersion),
        nonKDim(nonKDim), mmacLayout(mmacLayout),
        hcuFeatures(hcuFeatures) {}
  LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    using TensorValue = TypedValue<RankedTensorType>;
    if (mfmaVersion != 4) {
      return rewriter.notifyMatchFailure(
          dotOp,
          "F8F6F4 scaled dot requires CDNA4 MFMA or HCU MMAC_FP6FP4");
    }
    RankedTensorType oldRetType = dotOp.getType();
    if (!isa_and_nonnull<BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(
          dotOp, "expected blocked encoding result tensor");
    unsigned rank = oldRetType.getRank();
    if (rank == 3)
      return rewriter.notifyMatchFailure(dotOp, "NYI: 3d case");
    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    TensorValue aScale = dotOp.getAScale();
    TensorValue bScale = dotOp.getBScale();
    auto oldShape = oldRetType.getShape();
    ScaleDotElemType aElemType = dotOp.getAElemType();
    ScaleDotElemType bElemType = dotOp.getBElemType();
    auto supportsTypes = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E2M1 ||
             elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2 ||
             elemType == ScaleDotElemType::E2M3 ||
             elemType == ScaleDotElemType::E3M2;
    };
    if (!supportsTypes(aElemType) || !supportsTypes(bElemType)) {
      return rewriter.notifyMatchFailure(dotOp, "NYI: unsupported element type");
    }
    bool bothScalesAbsent = !aScale && !bScale;
    auto isF8 = [](ScaleDotElemType elemType) {
      return elemType == ScaleDotElemType::E4M3 ||
             elemType == ScaleDotElemType::E5M2;
    };
    if (bothScalesAbsent && isF8(aElemType) && isF8(bElemType)) {
      // Preserve native F8 dot semantics so the plain tt.dot rewrite can
      // match non-scaled MFMA instructions instead of forcing a dot_scaled
      // path with null scales.
      rewriter.replaceOpWithNewOp<tt::DotOp>(dotOp, oldRetType, a, b,
                                             dotOp.getC());
      return success();
    }
    MLIRContext *ctx = dotOp.getContext();
    ttg::CTAEncodingAttr ctaLayout = ttg::getCTALayout(oldRetType.getEncoding());
    unsigned numWarps = ttg::lookupNumWarps(dotOp);
    // if (numWarps == 1)
    //   return rewriter.notifyMatchFailure(dotOp,
    //                                      "num_warps==1 is not supported");
    auto [aMatInfo, bMatInfo] = getDotOperandMatrixLoadInfo(dotOp, hcuFeatures);
    bool useMatrixLoad = aMatInfo.has_value() || bMatInfo.has_value();
    auto wdraPlan = HCU::getWdraSplitPlan(dotOp);
    unsigned aScaledExt = aMatInfo.has_value() ? MLS_GEN_SCALED_EXT(aMatInfo.value().mlsElemBitTyKind, aMatInfo.value().isNonKPack) : 0;
    unsigned bScaledExt = bMatInfo.has_value() ? MLS_GEN_SCALED_EXT(bMatInfo.value().mlsElemBitTyKind, bMatInfo.value().isNonKPack) : 0;
    FailureOr<MfmaIntrinsic> mfmaInstr;
    if (useMatrixLoad) {
      mfmaInstr = chooseMfmaInstructionWithMLS(dotOp, mfmaVersion, nonKDim,
                                               aMatInfo, bMatInfo, hcuFeatures,
                                               wdraPlan ? &*wdraPlan : nullptr);
    } else {
      // Choose a suitable Scaled MFMA instruction for this scaled dot op.
      mfmaInstr = chooseMfmaInstruction(dotOp, mfmaVersion, nonKDim);
      if (failed(mfmaInstr))
        return rewriter.notifyMatchFailure(dotOp,
                                          "cannot choose scaled mfma intrinsic");
    }
    auto mDim = mfmaInstr->mDim;
    auto nDim = mfmaInstr->nDim;
    auto kDim = mfmaInstr->kDim;
    auto kBase = mfmaInstr->kBase;
    assert(mDim == nDim);
    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, oldShape, numWarps, {mDim, nDim});
    // For scale tensor preshuffling, the minimum block size is 32x32x256.
    // When using MFMA16 instructions, each warp should compute two MFMA ops
    // along the non-K dimension. To support this, we must set tilesPerWarp to
    // {2, 2}. Failing to do so won't break correctness, but it will prevent
    // vectorized local_loads, as the data each thread needs won't be contiguous
    // due to the shuffle pattern. This requirement doesn’t apply to MFMA32
    // instructions, since only one MFMA op spans the non-K dimension at the
    // minimal shuffling size.
    SmallVector<unsigned> tilesPerWarp = getTilesPerWarp(aScale, bScale);
    if (rank == 3) {
      tilesPerWarp.insert(tilesPerWarp.begin(), 1);
    }
    if (useMatrixLoad) {
      auto instM = mfmaInstr->mDim;
      auto instN = mfmaInstr->nDim;
      auto tileM = aMatInfo.has_value() ? aMatInfo.value().choosedElemsTile.first : instM;
      auto tileN = bMatInfo.has_value() ? bMatInfo.value().choosedElemsTile.first : instN;
      assert(tileM >= instM && tileN >= instN && tileM % instM == 0 && tileN % instN == 0);
      bool hasBatchDim = rank == 3;
      int mIndex = 0 + hasBatchDim;
      int nIndex = 1 + hasBatchDim;
      tilesPerWarp[mIndex] = tileM/instM;
      tilesPerWarp[nIndex] = tileN/instN;
      warpsPerTile = warpsPerTileMFMA(dotOp, oldShape, numWarps,
                                      {tileM, tileN});
    }
    auto effectiveMmacLayout =
        mmacLayout == MmacLayout::MFMA
            ? getMfmaMappedMmacLayout(
                  /*logicalIsTransposed=*/true,
                  (hcuFeatures & HCUISAFeature::MMAC_LAYOUT) != 0)
            : mmacLayout;
    mlir::Attribute mfmaEnc;
    if (llvm::any_of(tilesPerWarp, [](int x) { return x != 1; })) {
      mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
          ctx, /*verison=*/mfmaVersion, warpsPerTile, {mDim, nDim, kDim},
          /*isTransposed=*/false, ctaLayout, tilesPerWarp,
          oldRetType.getElementType().getIntOrFloatBitWidth(),
          effectiveMmacLayout);
    } else {
      mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
          ctx, /*verison=*/mfmaVersion, warpsPerTile, {mDim, nDim, kDim},
          /*isTransposed=*/false, ctaLayout, {},
          oldRetType.getElementType().getIntOrFloatBitWidth(),
          effectiveMmacLayout);
    }
    auto newRetType =
        RankedTensorType::get(oldShape, oldRetType.getElementType(), mfmaEnc);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        dotOp.getC().getLoc(), newRetType, dotOp.getC());
    auto order = ttg::getMatrixOrder(rank, /*rowMajor=*/true);
    auto standardOutDims = standardOutDimNames(ctx, rank);
    // B4: For the mfma_scale_f32_*_f8f6f4 instructions, each thread consumes 16
    // elements. But since two fp4 elements are packed into one int8, the kWidth is 8 for fp4.
    // F8F6F4: For the mfma_scale_f32_*_f8f6f4 instructions, each thread consumes 8
    // elements which is padding from fp4, so the kWidth is 8 for fp4 also.
    const unsigned kWidth = kBase;
    assert(kWidth == 8 || kWidth == 16);
    using basisT = std::vector<std::vector<int32_t>>;
    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();
    auto aEncLL = LinearLayout::empty();
    auto bEncLL = LinearLayout::empty();
    auto convertInputLayout = [&](TensorValue v,
                                  unsigned opIdx) -> TensorValue {
      auto vType = v.getType();
      // auto newEnc =
      //     DotOperandEncodingAttr::get(ctx, opIdx, mfmaEnc, kWidth/2);
      auto newEnc = DotOperandEncodingAttr::get(ctx, opIdx, mfmaEnc, 8);
      auto newEncA = DotOperandEncodingAttr::get(ctx, opIdx, mfmaEnc, 8, aScaledExt);
      auto newEncB = DotOperandEncodingAttr::get(ctx, opIdx, mfmaEnc, 8, bScaledExt);
      if (aMatInfo.has_value() && opIdx == 0) {
        aEncLL *= newEncA.toLinearLayout(aShape);
        auto newVType = RankedTensorType::get(vType.getShape(),
        vType.getElementType(), newEncA);
        return rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
      } else if (bMatInfo.has_value() && opIdx == 1) {
        bEncLL *= newEncB.toLinearLayout(bShape);
        auto newVType = RankedTensorType::get(vType.getShape(),
        vType.getElementType(), newEncB);
        return rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
      } else {
        bool kPacked = opIdx == 0 ? dotOp.getLhsKPack() : dotOp.getRhsKPack();
        if (kPacked == false) {
          // This is FP4 with M/N packing. Create local alloc + local load here
          // so we have control of the shared layout
          // A, M packed: tensor<16x64xi8> --> 32x32
          // B, N packed: tensor<64x16xi8> --> 32x32
          SmallVector<int64_t> newShape(vType.getShape());
          newShape[opIdx == 0 ? 0 : 1] = newShape[opIdx == 0 ? 0 : 1] * 2;
          newShape[opIdx == 0 ? 1 : 0] = newShape[opIdx == 0 ? 1 : 0] / 2;
          auto newVType =
              RankedTensorType::get(newShape, vType.getElementType(), newEnc);
          OpBuilder builder(dotOp);
          auto srcEncoding = vType.getEncoding();
          auto originalOrder = triton::gpu::getOrderForMemory(vType);
          SmallVector<unsigned> newOrder = originalOrder;
          if (opIdx == 1) {
            newOrder = {1, 0};
          } else {
            newOrder = {0, 1};
          }
          auto sharedMemorySpace =
              triton::gpu::SharedMemorySpaceAttr::get(vType.getContext());
          auto tmpType = triton::gpu::MemDescType::get(
              vType.getShape(), vType.getElementType(),
              triton::gpu::SwizzledSharedEncodingAttr::get(
                  v.getContext(), newEnc, vType.getShape(), newOrder,
                  triton::gpu::getCTALayout(srcEncoding), vType.getElementType()),
              sharedMemorySpace);
          auto tmp = builder.create<triton::gpu::LocalAllocOp>(dotOp.getLoc(),
                                                               tmpType, v);
          auto newConvert =
              builder.create<triton::amdgpu::LocalLoadPackedTransposedOp>(
                  dotOp.getLoc(), newVType, tmp);
          if (opIdx == 0) {
            aShape = newConvert.getType().getShape();
            aEncLL *= newEnc.toLinearLayout(aShape);
          } else {
            bShape = newConvert.getType().getShape();
            bEncLL *= newEnc.toLinearLayout(bShape);
          }
          return newConvert;
        } else {
          if (opIdx == 0)
            aEncLL *= newEnc.toLinearLayout(aShape);
          else
            bEncLL *= newEnc.toLinearLayout(bShape);
          auto newVType = RankedTensorType::get(vType.getShape(),
          vType.getElementType(), newEnc);
          return rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
        }
      }
    };
    a = convertInputLayout(a, 0);
    b = convertInputLayout(b, 1);
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    auto convertScaleLayout = [&](TensorValue scale,
                                  llvm::ArrayRef<int64_t> valShape,
                                  LinearLayout dotLL, int idx) -> Value {
      if (bothScalesAbsent)
        return Value();
      SmallVector<int64_t> shape;
      if (!scale) {
        int64_t nonKDim = idx == 0 ? valShape[0] : valShape[1];
        int64_t k = idx == 0 ? valShape[1] : valShape[0];
        ScaleDotElemType &elemType = idx == 0 ? aElemType : bElemType;
        int packSize = elemType == ScaleDotElemType::E2M1 ? 2 : 1;
        shape = {nonKDim, k * packSize / 32};
      } else {
        shape = llvm::to_vector(scale.getType().getShape());
      }
      LinearLayout newLL = chooseScaledMfmaScaleLayout(
          ctx, idx, shape, mDim, tilesPerWarp, warpsPerTile);
      Attribute newScaleEncoding = ttg::LinearEncodingAttr::get(ctx, newLL);
      // Scale's data type is always i8
      auto newScaleType = RankedTensorType::get(shape, i8_ty, newScaleEncoding);
      if (!scale) {
        // 0x7F is 1.0 in E8M0
        return rewriter.create<arith::ConstantOp>(
            dotOp->getLoc(), newScaleType,
            DenseElementsAttr::get(newScaleType, llvm::APInt(8, 0x7F)));
      } else {
        return rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(),
                                                     newScaleType, scale);
      }
    };
    auto newAScale =
        convertScaleLayout(aScale, aShape, aEncLL, /*dotOperandIdx=*/0);
    auto newBScale =
        convertScaleLayout(bScale, bShape, bEncLL, /*dotOperandIdx=*/1);
    auto newDot = rewriter.create<triton::DotScaledOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, newAScale, newBScale,
        aElemType, bElemType, dotOp.getFastMath(), dotOp.getLhsKPack(), dotOp.getRhsKPack());
    if (wdraPlan)
      HCU::setWdraSplitPlan(newDot, *wdraPlan);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot);
    return success();
  }
};
static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
}
// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](triton::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    if (isa<ttg::AMDMfmaEncodingAttr>(D.getType().getEncoding())) {
      Type BElType = dotOp.getB().getType().getElementType();
      auto maxBitWidth = std::max(AElType.getIntOrFloatBitWidth(),
                                  BElType.getIntOrFloatBitWidth());
      // TODO check mfma tensor core version compatibility
      if (maxBitWidth == 8)
        return;
      if (AElType == BElType)
        return;
      if (maxBitWidth < 16)
        promoteType = builder.getF16Type();
      else if (maxBitWidth <= 32)
        promoteType = builder.getF32Type();
    } else if (isa<ttg::AMDWmmaEncodingAttr>(D.getType().getEncoding())) {
      Type BElType = dotOp.getB().getType().getElementType();
      if (AElType == BElType)
        return;
      // Other cases must be filtered earlier
      promoteType =
          AElType.getIntOrFloatBitWidth() > BElType.getIntOrFloatBitWidth()
              ? AElType
              : BElType;
    } else {
      // FMA case is processed in AccelerateBlocked
      return;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}
FailureOr<WmmaIntrinsic> chooseWmmaInstruction(Location loc, int wmmaVersion,
                                               RankedTensorType cType,
                                               Type aElemType, Type bElemType,
                                               Type cElemType, int inputKSize,
                                               int enforcedNonKDim) {
  // number of matrix elements along k dim per one WMMA instruction
  unsigned kDim = 0;
  auto resShape = cType.getShape();
  auto rank = resShape.size();
  auto M = resShape[rank - 2];
  auto N = resShape[rank - 1];
  unsigned mDim = 0;
  unsigned nDim = 0;
  if (enforcedNonKDim != 0) {
    mDim = nDim = enforcedNonKDim;
  } else {
    int minSize = std::min(M, N);
    if (minSize >= 16) {
      mDim = 16;
      nDim = 16;
    }
  }
  if (mDim == 0 || nDim == 0)
    return failure();
  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic = WmmaIntrinsic::selectFor(
      wmmaVersion, mDim, nDim, inputKSize, aElemType, bElemType, cElemType);
  if (failed(maybeWmmaIntrinsic))
    return emitError(loc, "no matching matrix core intrinsic due to "
                          "unsupported element type");
  kDim = maybeWmmaIntrinsic->kDim;
  assert(kDim != 0);
  assert(enforcedNonKDim != 0 || (M % mDim == 0 && N % nDim == 0));
  // if inputKSize % kDim != 0 this layout will introduce data duplication,
  // consider FMA dot is preferred, except cases Wmma layout is enforced.
  if (enforcedNonKDim == 0 && inputKSize % kDim != 0)
    return failure();
  return maybeWmmaIntrinsic;
}
FailureOr<WmmaIntrinsic> chooseWmmaInstruction(tt::DotOp dot,
                                               OperandTypesVector operandTypes,
                                               int wmmaVersion, int nonKDim) {
  return chooseWmmaInstruction(dot.getLoc(), wmmaVersion, dot.getC().getType(),
                               operandTypes[0], operandTypes[1],
                               operandTypes[2],
                               dot.getA().getType().getShape().back(), nonKDim);
}
class BlockedToWMMA : public OpRewritePattern<tt::DotOp> {
  int wmmaVersion;
  int nonKDim;
public:
  BlockedToWMMA(MLIRContext *context, int wmmaVersion, int nonKDim,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), wmmaVersion(wmmaVersion),
        nonKDim(nonKDim) {}
  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto ctx = dotOp->getContext();
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    auto oldRetEncoding = oldRetType.getEncoding();
    if (!oldRetEncoding || !isa<ttg::BlockedEncodingAttr>(oldRetEncoding))
      return failure();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto retShape = oldRetType.getShape();
    auto aShape = oldAType.getShape();
    auto bShape = oldBType.getShape();
    // get operand types
    auto operandTypes = getOperandTypesForWmmaOp(rewriter, dotOp, wmmaVersion);
    if (operandTypes.empty())
      return failure();
    // check shape
    FailureOr<WmmaIntrinsic> wmmaInstr =
        chooseWmmaInstruction(dotOp, operandTypes, wmmaVersion, nonKDim);
    if (failed(wmmaInstr)) {
      return failure();
    }
    auto mDim = wmmaInstr->mDim;
    auto nDim = wmmaInstr->nDim;
    auto kDim = wmmaInstr->kDim;
    auto kBase = wmmaInstr->kBase;
    // get WMMA encoding for the given number of warps
    int numWarps = ttg::lookupNumWarps(dotOp);
    ttg::AMDWmmaEncodingAttr wmmaEnc;
    auto warpsPerTile =
        warpsPerTileWMMA(dotOp, retShape, numWarps, {mDim, nDim});
    auto CTALayout = ttg::getCTALayout(oldRetEncoding);
    // TODO implement heuristic/option for this parameter
    bool isTransposed = false;
    wmmaEnc = ttg::AMDWmmaEncodingAttr::get(ctx, wmmaVersion, isTransposed,
                                            warpsPerTile, CTALayout,
                                            {mDim, nDim, kDim});
    auto newRetType = RankedTensorType::get(retShape, operandTypes[3], wmmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getC();
    auto newAcc =
        convertAndCastTensor(rewriter, oldAcc, wmmaEnc, operandTypes[2]);
    // kBase, kWidth and kDim follow the same logic as in mfma
    // for now kwidth = kbase always
    auto kWidth = kBase;
    auto newAType = RankedTensorType::get(
        aShape, operandTypes[0],
        ttg::DotOperandEncodingAttr::get(ctx, 0, wmmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        bShape, operandTypes[1],
        ttg::DotOperandEncodingAttr::get(ctx, 1, wmmaEnc, kWidth));
    Value castedA = convertAndCastTensor(rewriter, a, newAType.getEncoding(),
                                         operandTypes[0]);
    Value castedB = convertAndCastTensor(rewriter, b, newBType.getEncoding(),
                                         operandTypes[1]);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newRetType, castedA, castedB, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
    Value dotOutput = convertAndCastTensor(rewriter, newDot, oldRetEncoding,
                                           oldRetType.getElementType());
    rewriter.replaceOp(dotOp, dotOutput);
    return success();
  }
};
class AccelerateBlocked : public OpRewritePattern<DotOp> {
  StringRef arch;
public:
  AccelerateBlocked(MLIRContext *context, StringRef arch,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), arch(arch) {}
  bool isFloat(Type t) const { return t.isIntOrFloat() && !t.isIntOrIndex(); }
  Value castToElTy(PatternRewriter &rewriter, Value v, Type elTy) const {
    Location loc = v.getLoc();
    auto srcTy = cast<RankedTensorType>(v.getType());
    auto dstTy = srcTy.cloneWith(std::nullopt, elTy);
    if (srcTy == dstTy)
      return v;
    auto srcElTy = srcTy.getElementType();
    auto dstElTy = dstTy.getElementType();
    if (isFloat(srcElTy) && isFloat(dstElTy)) {
      auto rmode =
          RoundingModeAttr::get(rewriter.getContext(), RoundingMode::RTNE);
      return rewriter.create<FpToFpOp>(loc, dstTy, v, rmode);
    }
    if (!isFloat(srcElTy) && isFloat(dstElTy))
      return rewriter.create<arith::SIToFPOp>(loc, dstTy, v);
    if (isFloat(srcElTy) && !isFloat(dstElTy))
      return rewriter.create<arith::FPToSIOp>(loc, dstTy, v);
    assert(false && "int -> int cast is unexpected in FMA legalization");
    return Value();
  }
  struct DotElTypes {
    Type a, b, c, d;
  };
  bool isLegalFMAForm(DotOp dotOp, const DotElTypes &dotTypes) const {
    if (AMD::supportsVDot(arch)) {
      auto aOpType = dotOp.getA().getType();
      int rank = aOpType.getRank();
      int k = aOpType.getShape()[rank - 1];
      // Try Fp16 x Fp16 -> Fp32 v_dot
      // if k % 2 != 0: can not use fp V_DOT instruction
      if (dotTypes.a.isF16() && dotTypes.b.isF16() && dotTypes.c.isF32() &&
          dotTypes.d.isF32() && k % 2 == 0) {
        return true;
      }
      // CDNA4 has Bf16 v_dot2
      if (AMD::deduceISAFamily(arch) == ISAFamily::CDNA4 &&
          dotTypes.a.isBF16() && dotTypes.b.isBF16() && dotTypes.c.isF32() &&
          dotTypes.d.isF32() && k % 2 == 0) {
        return true;
      }
      // TODO: enable this condition, when fp32 -> fp16 cast works correctly
      // Consider this case as non legal, despite this case is covered by fp16
      // FMA. Because v_dot expected to give both better performance and
      // computational precision.
      if (false && dotTypes.a.isF16() && dotTypes.b.isF16() &&
          dotTypes.c.isF16() && dotTypes.d.isF16() && k % 2 == 0) {
        return false;
      }
      // Try I8 x I8 -> I32 v_dot
      // if k % 4 != 0: can not use integer V_DOT instruction
      if (dotTypes.a.isInteger(8) && dotTypes.b.isInteger(8) &&
          dotTypes.c.isInteger(32) && dotTypes.d.isInteger(32) && k % 4 == 0) {
        return true;
      }
    }
    auto expectedElTy = dotTypes.a;
    for (auto operand : dotOp.getOperands()) {
      auto opTy = cast<RankedTensorType>(operand.getType());
      auto elTy = opTy.getElementType();
      if (elTy != expectedElTy)
        return false;
      if (!elTy.isF16() && !elTy.isF32() && !elTy.isF64())
        return false;
    }
    return true;
  }
  LogicalResult tryAccelerateF16WithVDot(DotOp dotOp, PatternRewriter &rewriter,
                                         const DotElTypes &dotTypes) const {
    if (!AMD::supportsVDot(arch))
      return failure();
    // If this is fp16 x fp16 ->fp16 case prioritize using v_dot.
    auto aOpType = dotOp.getA().getType();
    int rank = aOpType.getRank();
    int k = aOpType.getShape()[rank - 1];
    if (dotTypes.a.isF16() && dotTypes.b.isF16() && dotTypes.c.isF16() &&
        dotTypes.d.isF16() && k % 2 == 0) {
      auto newC = castToElTy(rewriter, dotOp.getC(), f32_ty);
      auto newDot = rewriter.create<DotOp>(
          dotOp.getLoc(), newC.getType(), dotOp.getA(), dotOp.getB(), newC,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
      auto newD = castToElTy(rewriter, newDot.getResult(), f16_ty);
      rewriter.replaceOp(dotOp, newD);
      return success();
    }
    return failure();
  }
  LogicalResult tryLegalizeFMA(DotOp dotOp, PatternRewriter &rewriter,
                               const DotElTypes &dotTypes) const {
    // Legalize dot for plain FMA case, i.e. same operands and result type.
    // Find common type, larger or equal of all operand types
    SmallVector<Type> opElTy{dotTypes.a, dotTypes.b, dotTypes.c, dotTypes.d};
    unsigned maxBitsize = 8;
    for (auto elTy : opElTy)
      maxBitsize = std::max(maxBitsize, elTy.getIntOrFloatBitWidth());
    assert(maxBitsize <= 32);
    Type commonTy =
        maxBitsize <= 16 ? rewriter.getF16Type() : rewriter.getF32Type();
    // Check that type is compatible with all operands; fallback to fp32 if not.
    if (commonTy.isF16()) {
      for (auto elTy : opElTy) {
        if (elTy.isInteger() && elTy.getIntOrFloatBitWidth() > 8) {
          commonTy = rewriter.getF32Type();
          break;
        }
        if (elTy.isBF16()) {
          commonTy = rewriter.getF32Type();
          break;
        }
      }
    }
    auto newA = castToElTy(rewriter, dotOp.getA(), commonTy);
    auto newB = castToElTy(rewriter, dotOp.getB(), commonTy);
    auto newC = castToElTy(rewriter, dotOp.getC(), commonTy);
    auto newDot = rewriter.create<DotOp>(dotOp.getLoc(), newC.getType(), newA,
                                         newB, newC, dotOp.getInputPrecision(),
                                         dotOp.getMaxNumImpreciseAcc());
    auto newD = castToElTy(rewriter, newDot.getResult(), dotTypes.d);
    rewriter.replaceOp(dotOp, newD);
    return success();
  }
  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<BlockedEncodingAttr>(dotOp.getD().getType().getEncoding()))
      return failure();
    DotElTypes dotTypes;
    dotTypes.a = dotOp.getA().getType().getElementType();
    dotTypes.b = dotOp.getB().getType().getElementType();
    dotTypes.c = dotOp.getC().getType().getElementType();
    dotTypes.d = dotOp.getD().getType().getElementType();
    // Check that dot is not legalized already
    if (isLegalFMAForm(dotOp, dotTypes)) {
      return failure();
    }
    // TODO: enable this condition, when fp32 -> fp16 cast works correctly
    if (false &&
        tryAccelerateF16WithVDot(dotOp, rewriter, dotTypes).succeeded()) {
      return success();
    }
    return tryLegalizeFMA(dotOp, rewriter, dotTypes);
  }
};
} // namespace

// HCU pass: full implementation with all matmul optimization patterns.
#define GEN_PASS_DEF_TRITONHCUACCELERATEMATMUL
#include "TritonHCU/Passes.h.inc"

struct TritonHCUAccelerateMatmulPass
    : impl::TritonHCUAccelerateMatmulBase<TritonHCUAccelerateMatmulPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet mfmaPatterns(context);
    auto features = mlir::triton::HCU::deduceHCUISAFeature(archGenerationName);
    bool supportsMmacLayout = (features & HCUISAFeature::MMAC_LAYOUT) != 0;
    auto mmacLayout = mmacLayoutForce != -1 ? *ttg::symbolizeMmacLayout(mmacLayoutForce) : MmacLayout::MFMA;
    if (mmacLayoutForce != -1 && !supportsMmacLayout) {
      emitWarning(m.getLoc(), "MMAC_LAYOUT feature is not supported on " + archGenerationName);
      mmacLayout = getMmacLayoutDefault(archGenerationName);
    }
    switch (auto isaFamily = triton::AMD::deduceISAFamily(archGenerationName)) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4:
      if (isaFamily == ISAFamily::CDNA4 ||
          (features & HCUISAFeature::MMAC_FP6FP4)) {
        mfmaPatterns.add<::ScaledBlockedToScaledMFMAF8F6F4>(
            context, /*mfmaVersion=*/4, matrixInstructionSize, mmacLayout,
            features,
            /*benefit=*/10);
      }
      mfmaPatterns.add<::BlockedToMFMA, ::ScaledBlockedToMFMA>(
          context, getMfmaVersion(isaFamily), matrixInstructionSize, kPack,
          mmacLayout, features,
          /*benefit=*/2);
      break;
    case ISAFamily::RDNA3:
      ttg::populateDecomposeScaledBlockedPatterns(mfmaPatterns,
                                                  /*benefit=*/3);
      mfmaPatterns.add<::BlockedToWMMA>(
          context, getWmmaVersion(archGenerationName), matrixInstructionSize,
                                    /*benefit=*/2);
      break;
    default:
      break;
    }
    if (applyPatternsGreedily(m, std::move(mfmaPatterns)).failed())
      signalPassFailure();
    RewritePatternSet patterns(context);
    patterns.add<AccelerateBlocked>(context, archGenerationName, /*benefit=*/1);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
    decomposeMixedModeDotOp(m);
  }
};

} // namespace mlir
