#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/MlsGroup.h"
#include "TritonHCU/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <limits>

namespace tt = triton;
namespace ttg = triton::gpu;
namespace tta = triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUUPDATEASYNCWAITCOUNT
#include "TritonHCU/Passes.h.inc"

namespace {

int getNumberOfLoadInstructions(RankedTensorType srcTy, ttg::MemDescType dstTy,
                                Value mask, int contig,
                                ModuleAxisInfoAnalysis &axisInfo) {
  LinearLayout srcLayout = tt::gpu::toLinearLayout(srcTy);
  LinearLayout sharedLayout;
  if (auto paddedEnc = dyn_cast<ttg::PaddedSharedEncodingAttr>(
          dstTy.getEncoding())) {
    sharedLayout = paddedEnc.getLinearComponent();
  } else {
    sharedLayout = tt::gpu::toLinearLayout(dstTy);
  }
  LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);
  contig = std::min(contig, srcToSharedLayout.getNumConsecutiveInOut());

  if (mask)
    contig = std::min<int>(contig, axisInfo.getMaskAlignment(mask));

  int numberOfRegisters = srcToSharedLayout.getInDimSize(
      StringAttr::get(srcTy.getContext(), "register"));
  return std::max(1, numberOfRegisters / contig);
}

int getMlsInstructionCount(tta::MatrixLoadToLocalOp op) {
  auto dstTy = cast<ttg::MemDescType>(op.getDest().getType());
  auto blockLayout =
      op->getAttrOfType<tta::MlsEncodingAttr>(tta::MlsEncodingAttr::getMnemonic());
  if (!blockLayout)
    return 0;

  ArrayRef<unsigned> mlsTile = blockLayout.getMlsTile();
  ArrayRef<unsigned> order = blockLayout.getOrder();
  ArrayRef<unsigned> warpsPerCTA = blockLayout.getWarpsPerCTA();
  ArrayRef<int64_t> shape = dstTy.getShape();
  if (mlsTile.size() != 2 || order.size() != 2 || warpsPerCTA.size() != 2 ||
      shape.size() != 2 || blockLayout.getOpIdx() > 1)
    return 0;

  unsigned opIdx = blockLayout.getOpIdx();
  unsigned kDimIdx = opIdx == 0 ? 1 : 0;
  unsigned nonKDimIdx = opIdx == 0 ? 0 : 1;
  if (mlsTile[kDimIdx] == 0 || mlsTile[nonKDimIdx] == 0 ||
      warpsPerCTA[0] == 0 || warpsPerCTA[1] == 0)
    return 0;

  auto maybeMlsInsn = tt::HCU::MlsInsn::selectOrGetMlsInsn(
      mlsTile[nonKDimIdx], mlsTile[kDimIdx], blockLayout.getElemBitWidth(),
      static_cast<tt::HCU::MlsElemBitTyKind>(blockLayout.getElemBitTyKind()),
      opIdx, order[0] == kDimIdx, blockLayout.getVersion(),
      static_cast<tt::HCU::MlsInterleaveKind>(blockLayout.getAlt2Kind()));
  if (failed(maybeMlsInsn))
    return 0;

  auto mlsTileG = maybeMlsInsn->getMlsTile(ttg::MlsTileKind::Global);
  const auto &mlInsnAttr = maybeMlsInsn->getMatrixLoadInsnAttr();
  unsigned numRepsX =
      std::max<unsigned>(1, shape[0] / (warpsPerCTA[0] * mlsTileG[0]));
  unsigned numRepsY =
      std::max<unsigned>(1, shape[1] / (warpsPerCTA[1] * mlsTileG[1]));
  return numRepsX * numRepsY * mlInsnAttr.instrsPerWarp[0] *
         mlInsnAttr.instrsPerWarp[1];
}

int getOpNumberOfAsyncLoadInstructions(Operation *op,
                                       ModuleAxisInfoAnalysis &axisInfo,
                                       bool emitRemarkOnNonAsyncOp) {
  if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    int contig = LLVM::AMD::getVectorSize(copyOp.getSrc(), axisInfo);
    return getNumberOfLoadInstructions(copyOp.getSrc().getType(),
                                       copyOp.getResult().getType(),
                                       copyOp.getMask(), contig, axisInfo);
  }
  if (auto bufferOp = dyn_cast<tta::BufferLoadToLocalOp>(op)) {
    auto ptrType = cast<RankedTensorType>(LLVM::AMD::getPointerTypeWithShape(
        bufferOp.getPtr(), bufferOp.getOffsets()));
    int contig = LLVM::AMD::getVectorSize(bufferOp.getPtr(),
                                          bufferOp.getOffsets(), axisInfo);
    return getNumberOfLoadInstructions(ptrType, bufferOp.getDest().getType(),
                                       bufferOp.getMask(), contig, axisInfo);
  }
  if (auto mlsOp = dyn_cast<tta::MatrixLoadToLocalOp>(op))
    return getMlsInstructionCount(mlsOp);

  if (emitRemarkOnNonAsyncOp) {
    SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    if (auto memEffectIface = dyn_cast<MemoryEffectOpInterface>(op))
      memEffectIface.getEffectsOnResource(tt::GlobalMemory::get(), effects);
    if (!effects.empty()) {
      op->emitRemark("Global memory operation between async wait and "
                     "async_loads. This will hinder the interleaving of memory "
                     "operations and might impact performance.");
    }
  }
  return 0;
}

using MemoCache = llvm::DenseSet<std::tuple<Operation *, int, int>>;

int computeMinCountBackward(Operation *cursor, Operation *cameFrom,
                            int numOutstanding, int pathSum, int bestPath,
                            MemoCache &branchStateCache,
                            llvm::function_ref<int(Operation *)> countFunc) {
  assert(cameFrom != nullptr);
  auto getPredecessor = [&cameFrom](Operation *op) {
    auto prevOp = op->getPrevNode();
    if (!prevOp) {
      prevOp = op->getParentOp();
      if (isa<ModuleOp>(prevOp))
        prevOp = nullptr;
    }
    return prevOp;
  };

  auto continueWalkFrom = [&](Operation *newCursor) {
    auto pathResult =
        computeMinCountBackward(newCursor, cursor, numOutstanding, pathSum,
                                bestPath, branchStateCache, countFunc);
    bestPath = std::min(bestPath, pathResult);
    return pathResult;
  };

  while (cursor) {
    if (numOutstanding < 0 || pathSum >= bestPath)
      return std::min(bestPath, pathSum);

    if (auto ifOp = dyn_cast<scf::IfOp>(cursor)) {
      bool cameFromThenOrElse = cameFrom->getParentOp() == ifOp;
      if (cameFromThenOrElse) {
        continueWalkFrom(getPredecessor(ifOp));
      } else {
        continueWalkFrom(ifOp.getThenRegion().front().getTerminator());
        if (!ifOp.getElseRegion().empty())
          continueWalkFrom(ifOp.getElseRegion().front().getTerminator());
        else
          continueWalkFrom(getPredecessor(ifOp));
      }
      return bestPath;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(cursor)) {
      continueWalkFrom(getPredecessor(forOp));
      auto cameFromBody = cameFrom->getBlock() == forOp.getBody();
      auto cacheKey = std::make_tuple(cursor, numOutstanding, pathSum);
      if (!cameFromBody || branchStateCache.insert(cacheKey).second)
        continueWalkFrom(forOp.getBody()->getTerminator());
      return bestPath;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(cursor)) {
      Block *lastBlock = cameFrom->getBlock();
      bool cameFromBefore = lastBlock == whileOp.getBeforeBody();
      bool cameFromAfter = lastBlock == whileOp.getAfterBody();
      bool cameFromSuccessor = !cameFromAfter && !cameFromBefore;

      if (cameFromAfter || cameFromSuccessor) {
        continueWalkFrom(whileOp.getBeforeBody()->getTerminator());
      } else if (cameFromBefore) {
        continueWalkFrom(getPredecessor(whileOp));
        auto cacheKey = std::make_tuple(cursor, numOutstanding, pathSum);
        if (branchStateCache.insert(cacheKey).second)
          continueWalkFrom(whileOp.getAfterBody()->getTerminator());
      }
      return bestPath;
    }
    if (isa<tt::FuncOp>(cursor))
      return std::min(bestPath, pathSum);
    if (cursor->getNumRegions() > 0 && !isa<tt::ReduceOp>(cursor)) {
      cursor->emitRemark(
          "has subregions but is not analyzed when determining async "
          "wait count; this yields conservative waits");
      return 0;
    }

    pathSum += countFunc(cursor);
    if (isa<ttg::AsyncCommitGroupOp>(cursor))
      numOutstanding--;

    cameFrom = cursor;
    cursor = getPredecessor(cursor);
  }
  return std::min(pathSum, bestPath);
}

int computeMinCountBackward(ttg::AsyncWaitOp waitOp,
                            llvm::function_ref<int(Operation *)> countFunc) {
  MemoCache memoCache;
  return computeMinCountBackward(waitOp, waitOp, waitOp.getNum(), 0,
                                 std::numeric_limits<int>::max(), memoCache,
                                 countFunc);
}

void updateWaitCount(ttg::AsyncWaitOp waitOp,
                     llvm::function_ref<int(Operation *)> computeCountForOp,
                     RewriterBase &rewriter) {
  int waitCnt = std::numeric_limits<int>::max();
  if (waitOp.getNumOperands() > 0) {
    for (auto token : waitOp.getOperands()) {
      auto tokenWaitCnt =
          deduceMinCountOnDefChain(token, waitOp, computeCountForOp);
      waitCnt = std::min(waitCnt, tokenWaitCnt);
    }
  } else {
    waitCnt = computeMinCountBackward(waitOp, computeCountForOp);
  }

  if (waitCnt == std::numeric_limits<int>::max())
    waitCnt = 0;

  auto tokens = waitOp.getAsyncToken();
  rewriter.setInsertionPointAfter(waitOp);
  rewriter.replaceOpWithNewOp<tta::AsyncWaitOp>(waitOp, tokens, waitCnt);
}

struct TritonHCUUpdateAsyncWaitCountPass
    : impl::TritonHCUUpdateAsyncWaitCountBase<
          TritonHCUUpdateAsyncWaitCountPass> {
  using Base::Base;

  void runOnOperation() override {
    tt::AMD::TargetInfo targetInfo(archGenerationName);
    if (!isCDNA(targetInfo.getISAFamily()) &&
        targetInfo.getISAFamily() != tt::AMD::ISAFamily::GFX1250)
      return;

    // asyncmark/wait_asyncmark preserves the number of outstanding commit
    // groups in ttg.async_wait. LLVM then derives the final hardware waitcnt
    // from all async operations in those groups, including HCU matrix loads.
    // Rewriting the wait to amdg.async_wait here would instead reinterpret the
    // group count as an instruction count and lose that dependency structure.
    if (targetInfo.useAsyncMarks())
      return;

    bool supportsAsyncLoads = true;
    switch (targetInfo.getISAFamily()) {
    case tt::AMD::ISAFamily::CDNA3:
    case tt::AMD::ISAFamily::CDNA4:
      supportsAsyncLoads = false;
      break;
    default:
      break;
    }

    ModuleOp module = getOperation();
    SmallVector<ttg::AsyncWaitOp> waitOps;
    module.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.push_back(waitOp); });

    ModuleAxisInfoAnalysis axisInfo(module);
    DenseMap<Operation *, int> intrinsicCountCache;
    auto countAsyncLoadInstructions = [&](Operation *op) {
      auto found = intrinsicCountCache.find(op);
      if (found != intrinsicCountCache.end())
        return found->second;
      auto count = getOpNumberOfAsyncLoadInstructions(op, axisInfo,
                                                      !supportsAsyncLoads);
      intrinsicCountCache[op] = count;
      return count;
    };

    for (auto waitOp : waitOps) {
      IRRewriter builder(waitOp->getContext());
      updateWaitCount(waitOp, countAsyncLoadInstructions, builder);
    }
  }
};

} // namespace

} // namespace mlir
