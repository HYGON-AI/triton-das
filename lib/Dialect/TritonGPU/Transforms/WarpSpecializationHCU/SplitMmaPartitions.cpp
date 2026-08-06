// SplitMmaPartitions.cpp
//
// HCU-specific pass that refines the warp-specialization partitioning by
// splitting MMA (and for TwoPTwoC, Load) partitions into multiple partitions
// based on async task ids assigned by the data-partitioning pass.
//
// OnePTwoC result (after LowerWarpGroup drops default; before OptimizePartitionWarps
// Load-first reorder):
//   [MMA0, Load, MMA1]
// TwoPTwoC result (before reorder):
//   [MMA0, Load0, MMA1, Load1]
// After OptimizePartitionWarps reorder (Load-first, main before tail):
//   OnePTwoC: [Load, MMA_main, MMA_tail]
//   TwoPTwoC: [Load_main, Load_tail, MMA_main, MMA_tail]

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace tta = mlir::triton::amdgpu;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUSPLITMMAPARTITIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {

using AsyncTaskId = int;

static SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      if (asyncTaskIds.empty() ||
          asyncTaskIds[asyncTaskIds.size() - 1] != asyncTaskId)
        asyncTaskIds.push_back(asyncTaskId);
    }
  }
  return asyncTaskIds;
}

// Split `partitionIdx` into keep (min task id) + appended move (max task id)
// when the partition's ops form exactly two async_task_id groups.
// Returns the new partition index, or nullopt if no split occurred.
static std::optional<int>
splitPartitionByAsyncTaskId(scf::ForOp loop, int partitionIdx,
                            SmallVector<Attribute> &stagesAttrs, Builder &b) {
  SmallVector<Operation *> partOps;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (!hasPartition(&op))
      continue;
    auto ids = getPartitionIds(&op);
    if (ids.size() != 1 || ids[0] != partitionIdx)
      continue;
    partOps.push_back(&op);
  }
  if (partOps.empty())
    return std::nullopt;

  DenseMap<AsyncTaskId, SmallVector<Operation *>> groupToOps;
  for (Operation *op : partOps) {
    auto asyncIds = getAsyncTaskIds(op);
    if (asyncIds.empty())
      continue;
    for (AsyncTaskId id : asyncIds)
      groupToOps[id].push_back(op);
  }

  if (groupToOps.size() != 2)
    return std::nullopt;

  SmallVector<AsyncTaskId> keys;
  for (auto &kv : groupToOps)
    keys.push_back(kv.first);
  llvm::sort(keys);

  AsyncTaskId keepKey = keys.front();
  AsyncTaskId moveKey = keys.back();

  if (partitionIdx < 0 ||
      static_cast<size_t>(partitionIdx) >= stagesAttrs.size())
    return std::nullopt;
  Attribute stageAttr = stagesAttrs[partitionIdx];
  int newPartitionIdx = static_cast<int>(stagesAttrs.size());
  stagesAttrs.push_back(stageAttr);

  SetVector<Operation *> partOpSet(partOps.begin(), partOps.end());
  SmallVector<Operation *> sortedOps =
      topologicalSort(partOpSet).takeVector();

  DenseMap<Operation *, Operation *> origToMoveClone;

  for (Operation *op : sortedOps) {
    auto asyncIds = getAsyncTaskIds(op);

    bool inKeep = false;
    bool inMove = false;

    if (asyncIds.empty()) {
      inKeep = true;
      inMove = true;
    } else {
      inKeep = llvm::is_contained(asyncIds, keepKey);
      inMove = llvm::is_contained(asyncIds, moveKey);
    }

    if (inKeep && !inMove)
      continue;

    if (!inKeep && inMove) {
      SetVector<int> newIds;
      newIds.insert(newPartitionIdx);
      setPartition(op, newIds);
      for (auto &operand : op->getOpOperands()) {
        Value val = operand.get();
        if (Operation *defOp = val.getDefiningOp()) {
          auto it = origToMoveClone.find(defOp);
          if (it != origToMoveClone.end()) {
            if (auto res = dyn_cast<OpResult>(val)) {
              unsigned resIdx = res.getResultNumber();
              operand.set(it->second->getResult(resIdx));
            }
          }
        }
      }
      continue;
    }

    if (inKeep && inMove) {
      OpBuilder opBuilder(op->getContext());
      opBuilder.setInsertionPointAfter(op);
      IRMapping mapping;
      for (auto &operand : op->getOpOperands()) {
        Value val = operand.get();
        if (Operation *defOp = val.getDefiningOp()) {
          auto it = origToMoveClone.find(defOp);
          if (it != origToMoveClone.end()) {
            if (auto res = dyn_cast<OpResult>(val)) {
              unsigned resIdx = res.getResultNumber();
              mapping.map(val, it->second->getResult(resIdx));
            }
          }
        }
      }
      Operation *clone = opBuilder.clone(*op, mapping);
      SetVector<int> newIds;
      newIds.insert(newPartitionIdx);
      setPartition(clone, newIds);
      origToMoveClone[op] = clone;

      for (auto it : llvm::enumerate(op->getResults())) {
        OpResult origResult = it.value();
        OpResult cloneResult = clone->getResult(it.index());
        for (OpOperand &use :
             llvm::make_early_inc_range(origResult.getUses())) {
          Operation *user = use.getOwner();
          if (auto it2 = origToMoveClone.find(user);
              it2 != origToMoveClone.end()) {
            use.set(cloneResult);
            continue;
          }
          auto userIds = getAsyncTaskIds(user);
          bool userInMove = llvm::is_contained(userIds, moveKey) &&
                            !llvm::is_contained(userIds, keepKey);
          if (userInMove)
            use.set(cloneResult);
        }
      }
    }
  }

  // Companion ops with async_task_id but no ttg.partition.
  for (Operation &opRef : loop.getBody()->without_terminator()) {
    Operation *op = &opRef;
    if (hasPartition(op))
      continue;
    auto asyncIds = getAsyncTaskIds(op);
    if (asyncIds.empty())
      continue;
    bool inKeep = llvm::is_contained(asyncIds, keepKey);
    bool inMove = llvm::is_contained(asyncIds, moveKey);
    if (!inKeep && !inMove)
      continue;

    SetVector<int> keepIds, moveIds;
    if (inKeep && !inMove) {
      keepIds.insert(partitionIdx);
      setPartition(op, keepIds);
    } else if (!inKeep && inMove) {
      moveIds.insert(newPartitionIdx);
      setPartition(op, moveIds);
    } else {
      keepIds.insert(partitionIdx);
      setPartition(op, keepIds);
      OpBuilder opBuilder(op->getContext());
      opBuilder.setInsertionPointAfter(op);
      IRMapping mapping;
      Operation *clone = opBuilder.clone(*op, mapping);
      moveIds.insert(newPartitionIdx);
      setPartition(clone, moveIds);
      for (auto it : llvm::enumerate(op->getResults())) {
        OpResult cloneResult = clone->getResult(it.index());
        for (OpOperand &use :
             llvm::make_early_inc_range(it.value().getUses())) {
          Operation *user = use.getOwner();
          if (user == clone)
            continue;
          auto userIds = getAsyncTaskIds(user);
          bool userInMove = llvm::is_contained(userIds, moveKey) &&
                            !llvm::is_contained(userIds, keepKey);
          if (userInMove)
            use.set(cloneResult);
        }
      }
    }
  }

  return newPartitionIdx;
}

static bool isLoadLikePartition(ArrayRef<Operation *> ops) {
  for (Operation *op : ops) {
    if (isa<DotOpInterface>(op))
      return false;
  }
  for (Operation *op : ops) {
    if (isa<LoadOp, DescriptorLoadOp, LocalStoreOp, tta::MatrixLoadToLocalOp>(
            op) ||
        isa<ROCDL::HCUAbarrierTryWaitOp, ROCDL::HCUAbarrierArriveOp>(op))
      return true;
  }
  return false;
}

// Only split a Load partition when producer data ops (not just abarriers) form
// two async_task_id groups. Barrier-only splits would duplicate full-tile
// loads into both Load partitions and race on the same LDS buffer.
static bool hasProducerTaskSplit(ArrayRef<Operation *> ops) {
  DenseMap<AsyncTaskId, int> groups;
  for (Operation *op : ops) {
    if (!isa<LoadOp, DescriptorLoadOp, LocalStoreOp, tta::MatrixLoadToLocalOp>(
            op))
      continue;
    for (AsyncTaskId id : getAsyncTaskIds(op))
      groups[id] = 1;
  }
  return groups.size() == 2;
}

struct SplitMmaPartitions
    : public mlir::triton::gpu::impl::TritonGPUSplitMmaPartitionsBase<
          SplitMmaPartitions> {
  using TritonGPUSplitMmaPartitionsBase<
      SplitMmaPartitions>::TritonGPUSplitMmaPartitionsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder b(mod.getContext());

    mod.walk([&](scf::ForOp loop) {
      auto stagesAttr =
          loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
      if (!stagesAttr)
        return;

      SmallVector<Attribute> newStagesAttrs(stagesAttr.begin(),
                                            stagesAttr.end());

      DenseMap<int, SmallVector<Operation *>> opsPerPartition;
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (!hasPartition(&op))
          continue;
        auto ids = getPartitionIds(&op);
        if (ids.size() != 1)
          continue;
        opsPerPartition[ids[0]].push_back(&op);
      }

      // 1) Split the MMA partition (contains DotOpInterface).
      std::optional<int> mmaPartitionIdx;
      for (auto &kv : opsPerPartition) {
        for (Operation *op : kv.second) {
          if (isa<DotOpInterface>(op)) {
            if (mmaPartitionIdx && *mmaPartitionIdx != kv.first)
              return;
            mmaPartitionIdx = kv.first;
          }
        }
      }
      if (mmaPartitionIdx)
        (void)splitPartitionByAsyncTaskId(loop, *mmaPartitionIdx,
                                          newStagesAttrs, b);

      // 2) TwoPTwoC: split the Load partition the same way. Re-scan because
      // MMA split may have appended a partition and retagged ops.
      opsPerPartition.clear();
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (!hasPartition(&op))
          continue;
        auto ids = getPartitionIds(&op);
        if (ids.size() != 1)
          continue;
        opsPerPartition[ids[0]].push_back(&op);
      }
      std::optional<int> loadPartitionIdx;
      for (auto &kv : opsPerPartition) {
        if (!isLoadLikePartition(kv.second))
          continue;
        if (!hasProducerTaskSplit(kv.second))
          continue;
        loadPartitionIdx = kv.first;
        break;
      }
      if (loadPartitionIdx)
        (void)splitPartitionByAsyncTaskId(loop, *loadPartitionIdx,
                                          newStagesAttrs, b);

      loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(newStagesAttrs));
    });
  }
};

} // namespace
