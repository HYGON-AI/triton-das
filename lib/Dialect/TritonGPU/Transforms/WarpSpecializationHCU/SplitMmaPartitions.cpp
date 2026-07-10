// SplitMmaPartitions.cpp
//
// HCU-specific pass that refines the warp-specialization partitioning by
// splitting a single MMA partition into multiple partitions based on
// async task ids that were assigned by the data-partitioning pass.

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

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

// For each warp-specialized loop, split the MMA partition into multiple
// partitions according to async task ids. This prepares the IR so that the
// subsequent loop-partitioning pass can materialize separate warp-group
// branches per MMA consumer group.
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

      SmallVector<int32_t> stages;
      stages.reserve(stagesAttr.size());
      for (Attribute a : stagesAttr) {
        auto ia = dyn_cast<IntegerAttr>(a);
        if (!ia)
          return;
        stages.push_back(static_cast<int32_t>(ia.getInt()));
      }

      // Collect ops per existing partition index. The partition attribute is a
      // DenseI32ArrayAttr (upstream faeb1eb54 changed it from IntegerAttr); use
      // getPartitionIds to read it. Ops in this HCU pipeline belong to a single
      // partition, so take the sole id.
      DenseMap<int, SmallVector<Operation *>> opsPerPartition;
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (!hasPartition(&op))
          continue;
        auto ids = getPartitionIds(&op);
        if (ids.size() != 1)
          continue;
        int idx = ids[0];
        opsPerPartition[idx].push_back(&op);
      }

      // Find the (single) partition that contains MMA ops.
      std::optional<int> mmaPartitionIdx;
      for (auto &kv : opsPerPartition) {
        int idx = kv.first;
        for (Operation *op : kv.second) {
          if (isa<DotOpInterface>(op)) {
            if (mmaPartitionIdx && *mmaPartitionIdx != idx) {
              // Multiple MMA partitions are not expected in this HCU pipeline.
              return;
            }
            mmaPartitionIdx = idx;
          }
        }
      }
      if (!mmaPartitionIdx)
        return;

      SmallVector<Operation *> &mmaOps = opsPerPartition[*mmaPartitionIdx];
      if (mmaOps.empty())
        return;

      // Group MMA-partition ops by their async task id "group". An op with
      // multiple async task ids (e.g. [0, 1]) is added to each corresponding
      // group so that we can split the partition by consumer group.
      DenseMap<AsyncTaskId, SmallVector<Operation *>> groupToOps;
      for (Operation *op : mmaOps) {
        auto asyncIds = getAsyncTaskIds(op);
        if (asyncIds.empty())
          continue;
        for (AsyncTaskId id : asyncIds)
          groupToOps[id].push_back(op);
      }

      if (groupToOps.size() <= 1)
        return;

      // Currently we only support splitting into two groups. Bail out if the
      // configuration does not match.
      if (groupToOps.size() != 2)
        return;

      // Keep the group with the smallest asyncTaskId in the original
      // partition; move the other group into a new partition appended at the
      // end of the partition list.
      SmallVector<AsyncTaskId> keys;
      for (auto &kv : groupToOps)
        keys.push_back(kv.first);
      llvm::sort(keys);

      AsyncTaskId keepKey = keys.front();
      AsyncTaskId moveKey = keys.back();

      SmallVector<Attribute> newStagesAttrs(stagesAttr.begin(),
                                            stagesAttr.end());
      if (*mmaPartitionIdx < 0 ||
          static_cast<size_t>(*mmaPartitionIdx) >= newStagesAttrs.size())
        return;
      Attribute mmaStageAttr = newStagesAttrs[*mmaPartitionIdx];
      int newPartitionIdx = static_cast<int>(newStagesAttrs.size());
      newStagesAttrs.push_back(mmaStageAttr);

      // Process MMA-partition ops in topological order so that when an op is
      // cloned for the move group, its operands can be rerouted to use the
      // move-group clones of their defining ops (if those were also cloned).
      SetVector<Operation *> mmaOpSet(mmaOps.begin(), mmaOps.end());
      SmallVector<Operation *> sortedMmaOps = topologicalSort(mmaOpSet).takeVector();

      // Map from original op (in the MMA partition) to its move-group clone.
      // Used to reroute cloned ops' operands to the corresponding move-group
      // clone results.
      DenseMap<Operation *, Operation *> origToMoveClone;

      // For all ops in the MMA partition, adjust their partition index
      // according to async_task_id:
      //  - ops without async_task_id are treated as belonging to both groups
      //    and will be duplicated into both partitions;
      //  - ops only in keepKey stay;
      //  - ops only in moveKey move to the new partition;
      //  - ops in both are cloned so that each partition gets its own copy.
      for (Operation *op : sortedMmaOps) {
        auto asyncIds = getAsyncTaskIds(op);

        bool inKeep = false;
        bool inMove = false;

        if (asyncIds.empty()) {
          // No async_task_id: treat as belonging to both groups so that the op
          // is duplicated into both MMA partitions.
          inKeep = true;
          inMove = true;
        } else {
          inKeep = llvm::is_contained(asyncIds, keepKey);
          inMove = llvm::is_contained(asyncIds, moveKey);
        }

        // Only in keep group: keep original partition index.
        if (inKeep && !inMove)
          continue;

        // Only in move group: retag op to new partition. Also reroute its
        // operands to use move-group clones where available, so the retagged
        // op does not reference keep-partition SSA values.
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

        // In both groups: clone op and route uses by move-group consumers to
        // the clone, keeping the original for keep-group consumers.
        if (inKeep && inMove) {
          OpBuilder opBuilder(op->getContext());
          opBuilder.setInsertionPointAfter(op);
          IRMapping mapping;
          // Before cloning, reroute operands that have move-group clones so
          // the clone in the move partition uses move-partition SSA values
          // instead of referencing the original (keep-partition) operands.
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
              // If the user is a move-group clone we already created, reroute
              // it to use this clone's result.
              if (auto it2 = origToMoveClone.find(user);
                  it2 != origToMoveClone.end()) {
                use.set(cloneResult);
                continue;
              }
              auto userIds = getAsyncTaskIds(user);
              bool userInMove =
                  llvm::is_contained(userIds, moveKey) &&
                  !llvm::is_contained(userIds, keepKey);
              if (userInMove)
                use.set(cloneResult);
            }
          }
        }
      }

      // Companion ops created by the data-partitioning pass (e.g.
      // memdesc_subslice views of partitioned local_allocs) carry an
      // async_task_id but no ttg.partition attribute, so by the legacy
      // convention they are implicitly treated as root and would be cloned
      // into every partition by PartitionLoops. After the split above, their
      // producing local_alloc may live in a specific MMA sub-partition, so
      // cloning the companion into all partitions leaves clones that reference
      // the original (about-to-be-erased) local_alloc -> "operation destroyed
      // but still has uses". Assign each unannotated companion op to the MMA
      // sub-partition matching its async_task_id so it follows its producer.
      for (Operation &opRef : loop.getBody()->without_terminator()) {
        Operation *op = &opRef;
        // Skip ops that already have an explicit partition assignment.
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
          keepIds.insert(*mmaPartitionIdx);
          setPartition(op, keepIds);
        } else if (!inKeep && inMove) {
          moveIds.insert(newPartitionIdx);
          setPartition(op, moveIds);
        } else {
          // Shared by both sub-partitions: keep the original in the keep
          // partition and clone it into the move partition, routing move-group
          // consumers to the clone (mirroring the both-groups handling above).
          keepIds.insert(*mmaPartitionIdx);
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
              bool userInMove =
                  llvm::is_contained(userIds, moveKey) &&
                  !llvm::is_contained(userIds, keepKey);
              if (userInMove)
                use.set(cloneResult);
            }
          }
        }
      }

      // Write back the updated stages.
      loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(newStagesAttrs));
    });
  }
};

} // namespace

