#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

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

struct WarpGroupBuilder : public OpBuilder {
  WarpGroupBuilder(Block *block, Block::iterator insertPoint,
                   size_t partitionId)
      : OpBuilder(block, insertPoint), partitionId(partitionId) {}

  IRMapping mapping;
  size_t partitionId;
};

// This is computed per loop and partition
enum class LoopVarCategory {
  // The given loop variable is not used by the given partition. For example,
  // the use-D flag for MMA is only used by the MMA partition, and thus
  // is `Unused` for any other partition.
  Unused,
  // The given loop variable is used by the given partition. For example, a loop
  // index might be used to compute a relevant stage or phase value for the
  // given partition.
  Used,
  // The results of warp_group op are defined to be those of the first
  // partition. If the original loop results include a tensor which is computed
  // only by a non-default partition, such tensor cannot be returned from the
  // first partition and and must be passed through shared memory. The
  // corresponding loop variable falls into this category.
  // Recognizing this category is necessary for the first partition. For other
  // partitions, some loop variables might be assigned this category, but that
  // information is not used.
  TensorResultFromOtherPartition,
};

bool isTensorResultComputedBy(scf::ForOp loop, size_t resultIdx,
                              const Partition *partition,
                              const WarpSchedule &schedule) {
  bool ret = false;
  schedule.iterateOutputs(loop, partition, [&](Operation *op, OpOperand &use) {
    if (isa<scf::YieldOp>(op) && use.getOperandNumber() == resultIdx &&
        isa<RankedTensorType>(loop.getResult(resultIdx).getType())) {
      ret = true;
    }
  });
  return ret;
}

SmallVector<LoopVarCategory> classifyLoopVars(scf::ForOp loop,
                                              const Partition *partition,
                                              const WarpSchedule &schedule) {
  auto inPartition = [&](Operation *op) {
    const Partition *opPartition =
        schedule.getPartition(loop.getBody()->findAncestorOpInBlock(*op));
    return llvm::is_contained({partition, schedule.getRootPartition()},
                              opPartition);
  };
  auto isTensorResultFromOtherPartition = [&](int i) {
    for (auto otherPartition : schedule.getPartitions()) {
      if (&otherPartition == partition) {
        continue;
      }
      if (isTensorResultComputedBy(loop, i, &otherPartition, schedule)) {
        return true;
      }
    }
    return false;
  };

  SmallVector<LoopVarCategory> categories(loop.getNumRegionIterArgs());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (llvm::any_of(arg.getUsers(), inPartition)) {
      categories[i] = LoopVarCategory::Used;
    } else if (isTensorResultFromOtherPartition(i)) {
      categories[i] = LoopVarCategory::TensorResultFromOtherPartition;
    } else {
      categories[i] = LoopVarCategory::Unused;
    }
  }

  return categories;
}

std::pair<SmallVector<size_t>, SmallVector<std::optional<size_t>>>
getLoopVarIndicesToKeep(scf::ForOp loop, const Partition *partition,
                        ArrayRef<LoopVarCategory> loopVarCategories) {
  SmallVector<size_t> indices;
  // The null index means an invalid index, the corresponding loop variable in
  // the original loop is removed in the cloned loop
  SmallVector<std::optional<size_t>> reverseIndices(loop.getNumRegionIterArgs(),
                                                    std::nullopt);
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    // For the default partition, keep non-tensor results used outside of the
    // loop even if the corresponding loop variable is not used in that
    // partition.
    if (loopVarCategories[i] == LoopVarCategory::Used ||
        (partition->getIndex() == 0 && !loop.getResult(i).use_empty() &&
         loopVarCategories[i] !=
             LoopVarCategory::TensorResultFromOtherPartition)) {
      reverseIndices[i] = indices.size();
      indices.push_back(i);
    }
  }
  return std::make_pair(indices, reverseIndices);
}

std::pair<SmallVector<size_t>, SmallVector<std::optional<size_t>>>
getLoopVarIndicesToKeep(scf::ForOp loop, const Partition *partition,
                        const WarpSchedule &schedule) {
  auto loopVarCategories = classifyLoopVars(loop, partition, schedule);
  return getLoopVarIndicesToKeep(loop, partition, loopVarCategories);
}

const Partition *getPartition(Operation *op, const WarpSchedule &schedule) {
  auto origOp = op;
  while (op && !schedule.getPartition(op)) {
    op = op->getParentOp();
  }
  if (op) {
    return schedule.getPartition(op);
  }

  // Some yield ops, e.g. automatically added one, might not have a partition
  assert(isa<scf::YieldOp>(origOp) && "No partition is found for an op.");
  return nullptr;
}

void mapRange(ValueRange fromRange, ValueRange toRange, IRMapping &mapping) {
  for (auto [from, to] : llvm::zip(fromRange, toRange)) {
    mapping.map(from, to);
  }
}

SmallVector<size_t>
getPartitionIndicesToCloneInto(const Partition *partition,
                               const WarpSchedule &schedule) {
  SmallVector<size_t> partitionIndices;

  if (!partition || partition == schedule.getRootPartition()) {
    for (size_t i = 0; i < schedule.getNumPartitions(); ++i) {
      partitionIndices.push_back(i);
    }
  } else {
    partitionIndices.push_back(partition->getIndex());
  }

  return partitionIndices;
}
int getPartitionIndex(Operation *op) {
  if (isa<nvws::WarpGroupOp>(op->getParentOp()))
    return op->getParentRegion()->getRegionNumber();
  return getPartitionIndex(op->getParentOp());
}

void cloneOpsInBlock(Block *block, SmallVector<WarpGroupBuilder> &builders,
                     const WarpSchedule &schedule);

void cloneForOp(scf::ForOp forOp, SmallVector<WarpGroupBuilder> &builders,
                const WarpSchedule &schedule) {
  SmallVector<scf::ForOp> newForOps;
  for (auto [b, partition] : llvm::zip(builders, schedule.getPartitions())) {
    auto [newLoopIndices, _] =
        getLoopVarIndicesToKeep(forOp, &partition, schedule);
    auto lb = b.mapping.lookupOrDefault(forOp.getLowerBound());
    auto ub = b.mapping.lookupOrDefault(forOp.getUpperBound());
    auto step = b.mapping.lookupOrDefault(forOp.getStep());
    SmallVector<Value> initArgs;
    for (auto idx : newLoopIndices) {
      initArgs.push_back(forOp.getInitArgs()[idx]);
    }
    auto newForOp =
        b.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, initArgs);
    newForOp->setAttrs(forOp->getAttrs());
    newForOps.push_back(newForOp);

    b.mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    auto oldIterArgs = forOp.getRegionIterArgs();
    auto newIterArgs = newForOp.getRegionIterArgs();
    for (auto [newIdx, oldIdx] : llvm::enumerate(newLoopIndices)) {
      b.mapping.map(oldIterArgs[oldIdx], newIterArgs[newIdx]);
      b.mapping.map(forOp.getResult(oldIdx), newForOp.getResult(newIdx));
    }

    b.setInsertionPointToStart(newForOp.getBody());
  }

  cloneOpsInBlock(forOp.getBody(), builders, schedule);

  for (auto newForOp : newForOps) {
    builders[getPartitionIndex(newForOp)].setInsertionPointAfter(newForOp);
    WarpSchedule::eraseFrom(newForOp);
  }
}

void cloneIfOp(scf::IfOp ifOp, SmallVector<WarpGroupBuilder> &builders,
               const WarpSchedule &schedule) {
  auto partition = getPartition(ifOp, schedule);
  auto partitionIndices = getPartitionIndicesToCloneInto(partition, schedule);

  SmallVector<scf::IfOp> newIfOps;
  for (size_t idx : partitionIndices) {
    auto &b = builders[idx];
    auto cond = b.mapping.lookupOrDefault(ifOp.getCondition());
    auto newIfOp = b.create<scf::IfOp>(ifOp.getLoc(), ifOp.getResultTypes(),
                                       cond, ifOp.elseBlock() ? true : false);
    newIfOp->setAttrs(ifOp->getAttrs());
    newIfOps.push_back(newIfOp);

    mapRange(ifOp.getResults(), newIfOp.getResults(), b.mapping);
    mapRange(ifOp.thenBlock()->getArguments(),
             newIfOp.thenBlock()->getArguments(), b.mapping);

    if (ifOp.elseBlock()) {
      mapRange(ifOp.elseBlock()->getArguments(),
               newIfOp.elseBlock()->getArguments(), b.mapping);
    }

    b.setInsertionPointToStart(newIfOp.thenBlock());
  }

  cloneOpsInBlock(ifOp.thenBlock(), builders, schedule);

  if (auto elseBlock = ifOp.elseBlock()) {
    for (auto [builder, newIfOp] : llvm::zip(builders, newIfOps)) {
      builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    cloneOpsInBlock(elseBlock, builders, schedule);
  }

  for (auto [idx, newIfOp] : llvm::zip(partitionIndices, newIfOps)) {
    builders[idx].setInsertionPointAfter(newIfOp);
  }
}

void cloneReduceOp(triton::ReduceOp reduceOp,
                   SmallVector<WarpGroupBuilder> &builders,
                   const WarpSchedule &schedule) {
  auto partition = getPartition(reduceOp, schedule);
  auto partitionIndices = getPartitionIndicesToCloneInto(partition, schedule);

  SmallVector<ReduceOp> newReduceOps;
  for (size_t idx : partitionIndices) {
    auto &b = builders[idx];

    SmallVector<Value> srcs;
    for (auto src : reduceOp.getSrcs()) {
      srcs.push_back(b.mapping.lookupOrDefault(src));
    }
    auto axis = reduceOp.getAxis();
    auto newReduceOp =
        b.create<triton::ReduceOp>(reduceOp.getLoc(), srcs, axis);
    newReduceOp->setAttrs(reduceOp->getAttrs());
    newReduceOps.push_back(newReduceOp);

    mapRange(reduceOp.getResults(), newReduceOp.getResults(), b.mapping);

    auto &region = newReduceOp.getRegion();
    Block *block = &region.emplaceBlock();
    for (auto arg : reduceOp.getRegion().getBlocks().front().getArguments()) {
      auto newArg = block->addArgument(arg.getType(), arg.getLoc());
      b.mapping.map(arg, newArg);
    }

    b.setInsertionPointToStart(block);
  }

  cloneOpsInBlock(reduceOp.getBody(), builders, schedule);

  for (auto [idx, newReduceOp] : llvm::zip(partitionIndices, newReduceOps)) {
    builders[idx].setInsertionPointAfter(newReduceOp);
  }
}

void cloneOp(Operation *op, SmallVector<WarpGroupBuilder> &builders,
             SmallVector<size_t> const &partitionIndices) {
  if (op->getNumRegions() != 0) {
    llvm::report_fatal_error(
        "Ops are expected to be regionless at this point.");
  }

  for (size_t idx : partitionIndices) {
    auto &builder = builders[idx];
    auto newOp = builder.clone(*op, builder.mapping);
    mapRange(op->getResults(), newOp->getResults(), builder.mapping);
  }
}

void cloneOpsInBlock(Block *block, SmallVector<WarpGroupBuilder> &builders,
                     const WarpSchedule &schedule) {
  for (auto &op_ : *block) {
    auto op = &op_;
    auto partition = getPartition(op, schedule);
    auto partitionIndices = getPartitionIndicesToCloneInto(partition, schedule);

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      cloneForOp(forOp, builders, schedule);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      cloneIfOp(ifOp, builders, schedule);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      cloneReduceOp(reduceOp, builders, schedule);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      if (yieldOp.getOperands().empty()) {
        continue;
      }

      for (size_t idx : partitionIndices) {
        auto &builder = builders[idx];
        SmallVector<size_t> newOperandIndices;
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          newOperandIndices =
              getLoopVarIndicesToKeep(
                  forOp, schedule.getPartition(builder.partitionId), schedule)
                  .first;
        } else {
          for (size_t i = 0; i < yieldOp.getOperands().size(); ++i) {
            newOperandIndices.push_back(i);
          }
        }

        SmallVector<Value> newYieldOperands;
        for (size_t i : newOperandIndices) {
          newYieldOperands.push_back(
              builder.mapping.lookupOrDefault(yieldOp.getOperand(i)));
        }

        if (!newYieldOperands.empty()) {
          builder.create<scf::YieldOp>(op->getLoc(), newYieldOperands);
        }
      }
    } else {
      cloneOp(op, builders, partitionIndices);
    }
  }
}
} // namespace

namespace mlir::triton::gpu::hcu {
LogicalResult partitionLoop(scf::ForOp loop) {
  FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
  if (failed(scheduleOr))
    return failure();
  WarpSchedule schedule = std::move(*scheduleOr);
  if (failed(schedule.verify(loop)))
    return failure();

  // Only the root node should have consumers at this point. The root (default)
  // partition is replicated into every warp group region, so its outputs are
  // allowed to be consumed by any partition; skip it here. Non-root partitions
  // must have had their cross-partition SSA dependencies cut by the
  // rewrite-partition-dependencies pass.
  for (const Partition &partition : schedule.getPartitions()) {
    if (&partition == schedule.getRootPartition())
      continue;
    bool failed = false;
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      const Partition *usePartition = schedule.getPartition(owner);
      if (usePartition == schedule.getRootPartition() ||
          usePartition == &partition)
        return;
      failed = true;
      Operation *defOp = output.getDefiningOp();
      InFlightDiagnostic diag =
          mlir::emitError(output.getLoc(), "non-root partition #")
          << partition.getIndex() << " has direct SSA consumer in partition #"
          << usePartition->getIndex() << " (distance=" << distance << ")";
      if (defOp)
        diag.attachNote(defOp->getLoc()) << "producer op: " << defOp->getName();
      diag.attachNote(use.getOwner()->getLoc())
          << "consumer op: " << use.getOwner()->getName()
          << " operand #" << use.getOperandNumber();
    };
    schedule.iterateUses(loop, &partition, callback);
    if (failed)
      return failure();
  }

  // There is nothing to do if the loop has 1 or fewer partitions.
  if (llvm::size(schedule.getPartitions()) <= 1)
    return success();

  auto numPartitions = schedule.getNumPartitions();
  auto defaultPartition = schedule.getPartition((int)0);
  auto loopVarCategories = classifyLoopVars(loop, defaultPartition, schedule);
  auto [loopVarIndices, newResultIndices] =
      getLoopVarIndicesToKeep(loop, defaultPartition, loopVarCategories);

  ImplicitLocOpBuilder topBuilder(loop.getLoc(), loop);
  SmallVector<Value> tensorResultAllocs(loop.getNumRegionIterArgs());
  SmallVector<Value> tensorResults(loop.getNumRegionIterArgs());
  //for (auto [i, res] : llvm::enumerate(loop.getResults())) {
  //  // FIXME: If the result if computed by a mmac partition, do not allocate SMEM.
  //  if (loopVarCategories[i] ==
  //      LoopVarCategory::TensorResultFromOtherPartition) {
  //    auto ty = cast<RankedTensorType>(res.getType());
  //    auto memdesc = MemDescType::get(
  //        ty.getShape(), ty.getElementType(), getSharedEncoding(ty),
  //        SharedMemorySpaceAttr::get(ty.getContext()), /*mutable=*/true);
  //    tensorResultAllocs[i] = topBuilder.create<LocalAllocOp>(memdesc);
  //  }
  //}

  SmallVector<Type> resultTypes;
  for (auto i : loopVarIndices) {
    resultTypes.push_back(loop.getResultTypes()[i]);
  }

  SmallVector<int32_t> numWarps(numPartitions, lookupNumWarps(loop));
  auto wgOp = topBuilder.create<nvws::WarpGroupOp>(resultTypes, numWarps,
                                                   numPartitions);

  SmallVector<WarpGroupBuilder> builders;
  for (Region &region : wgOp.getPartitionRegions()) {
    auto partitionId = builders.size();
    auto &block = region.emplaceBlock();
    builders.push_back(WarpGroupBuilder(&block, block.end(), partitionId));
  }

  SmallVector<Operation *> opsToErase;
  for (auto &op_ : *loop->getBlock()) {
    auto op = &op_;
    auto wsTag = op->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
    if (!wsTag || wsTag.getInt() != schedule.getTag())
      continue;
    // The partition attribute is a DenseI32ArrayAttr (upstream faeb1eb54
    // changed it from IntegerAttr); companion ops carry a single partition id.
    if (hasPartition(op)) {
      auto ids = getPartitionIds(op);
      assert(ids.size() == 1 && "expected single partition for companion op");
      cloneOp(op, builders, {static_cast<size_t>(ids[0])});
      opsToErase.push_back(op);
    } else {
      assert(loop.getOperation() == op && "Unexpected op");
      cloneForOp(loop, builders, schedule);
      opsToErase.push_back(loop);
    }
  }

  SmallVector<size_t> mmaPartitionIds;
  for (auto [b, region, partition] : llvm::zip(
           builders, wgOp.getPartitionRegions(), schedule.getPartitions())) {
    auto newForOp = *region.front().getOps<scf::ForOp>().begin();
    auto outputs = newForOp.getResults();

    // Partition 0 is the default (root) partition that produces the final
    // results of the warp group.
    if (b.partitionId == 0) {
      b.create<nvws::WarpGroupYieldOp>(wgOp.getLoc(), outputs);
    } else {
      // Tensor results computed by non-default partitions are communicated back
      // via SMEM.
      // The calls to getLoopVarIndicesToKeep and isTensorResultComputedBy
      // below are unnecessary if we can encode the partition index and the
      // corresponding result tensor index of newForOp in
      // LoopVarCategory::TensorResultFromOtherPartition. In the absence of such
      // language support, we end up computing the same information multiple
      // times.
      auto [_, reverseIndices] =
          getLoopVarIndicesToKeep(loop, &partition, schedule);
      for (size_t i = 0; i < loop.getNumRegionIterArgs(); ++i) {
        if (loopVarCategories[i] ==
                LoopVarCategory::TensorResultFromOtherPartition &&
            isTensorResultComputedBy(loop, i, &partition, schedule)) {
          assert(reverseIndices[i] && "A valid index is expected.");
          auto result = newForOp.getResult(*reverseIndices[i]);
          tensorResults[i] = result;
        }
      }
        b.create<nvws::WarpGroupReturnOp>(wgOp.getLoc());
    }

    // Record partitions that contain MMA ops so we can later route epilogue
    // ops into the correct MMA partitions based on async_task_id.
    bool hasMma = false;
    for (Operation &op : *newForOp.getBody()) {
      if (isa<DotOpInterface>(&op)) {
        hasMma = true;
        break;
      }
    }
    if (hasMma)
      mmaPartitionIds.push_back(b.partitionId);
  }

  topBuilder.setInsertionPointAfter(wgOp);

  for (auto [i, res] : llvm::enumerate(loop.getResults())) {
    if (res.use_empty())
      continue;

    if (loopVarCategories[i] ==
        LoopVarCategory::TensorResultFromOtherPartition) {
      auto ty = cast<RankedTensorType>(loop.getResult(i).getType());
      (void)ty;
      // auto output = topBuilder.create<LocalLoadOp>(ty, tensorResultAllocs[i]);
      // topBuilder.create<LocalDeallocOp>(tensorResultAllocs[i]);
      res.replaceAllUsesWith(tensorResults[i]);
    } else {
      assert(newResultIndices[i] && "A valid index is expected.");
      res.replaceAllUsesWith(wgOp.getResult(*newResultIndices[i]));
    }
  }

  // Clone all operations after wgOp into MMA partitions.
  //
  // - If there are **two** MMA partitions, we use async_task_id to route:
  //     * async_task_id contains only 0  -> first MMA partition
  //     * async_task_id contains only 1  -> second MMA partition
  //     * no async_task_id or {0, 1}     -> both MMA partitions
  //
  // - If there is **only one** MMA partition (e.g. "one mma + load" case
  //   without async_task_id), all epilogue ops are cloned into that single
  //   MMA partition.
  //
  // Other partitions (e.g. load) are not used for these epilogue ops.
  Block *wgOpBlock = wgOp->getBlock();
  auto wgOpIt = wgOp->getIterator();

  // Collect all operations after wgOp (excluding the block terminator).
  SmallVector<Operation *> opsToClone;
  Operation *wgOpBlockTerminator = wgOpBlock->getTerminator();
  for (auto it = std::next(wgOpIt); it != wgOpBlock->end(); ++it) {
    Operation *op = &*it;
    if (op == wgOpBlockTerminator)
      continue;
    opsToClone.push_back(op);
  }

  // Helper: given async_task_id (when available), decide target MMA partitions.
  // We map async task 0/1 to the first/second MMA partition we discovered
  // above when there are two MMA partitions. When there is only one MMA
  // partition, all epilogue ops go to that partition regardless of
  // async_task_id.
  auto getTargetMmaPartitions = [&](Operation *op) -> SmallVector<size_t> {
    // No MMA partitions detected: conservatively do nothing.
    if (mmaPartitionIds.empty())
      return {};

    // Single MMA partition: route all epilogue ops into that partition. This
    // covers the "one mma + load" case where there is no async_task_id.
    if (mmaPartitionIds.size() == 1)
      return {mmaPartitionIds.front()};

    // Two MMA partitions: use async_task_id 0/1 to select the target(s).
    if (mmaPartitionIds.size() != 2)
      return {};

    size_t async0Partition = mmaPartitionIds[0];
    size_t async1Partition = mmaPartitionIds[1];

    SmallVector<AsyncTaskId> ids = getAsyncTaskIds(op);
    bool has0 = llvm::is_contained(ids, 0);
    bool has1 = llvm::is_contained(ids, 1);

    // No async_task_id: duplicate into both partitions.
    if (ids.empty())
      return {async0Partition, async1Partition};

    // Exactly {0} or {1}.
    if (has0 && !has1)
      return {async0Partition};
    if (!has0 && has1)
      return {async1Partition};

    // Contains both 0 and 1 (or other combinations): duplicate into both.
    return {async0Partition, async1Partition};
  };

  // Clone each op into the selected MMA partitions, using the existing
  // WarpGroupBuilder mappings so that operands are remapped correctly to the
  // per-partition SSA values. The original epilogue ops are erased afterwards.
  for (Operation *op : opsToClone) {
    if (op->getNumRegions() != 0)
      continue; // Skip ops with regions for now.

    SmallVector<size_t> targets = getTargetMmaPartitions(op);
    for (size_t idx : targets) {
      if (idx >= wgOp.getPartitionRegions().size())
        continue;

      auto &region = wgOp.getPartitionRegions()[idx];
      auto &builder = builders[idx];
      Operation *terminator = region.front().getTerminator();
      builder.setInsertionPoint(terminator);
      Operation *newOp = builder.clone(*op, builder.mapping);
      mapRange(op->getResults(), newOp->getResults(), builder.mapping);
    }

  }

  // All ops in opsToClone are epilogue ops in the same block after wgOp.
  // After cloning them into the MMA partitions, the originals should no longer
  // execute, so we erase them unconditionally in reverse order.
  for (Operation *op : llvm::reverse(opsToClone))
    op->erase();

  // Some ops in opsToErase might already have been erased as part of
  // opsToClone. Guard on getBlock() to avoid double-erasing.
  for (Operation *op : llvm::reverse(opsToErase))
    if (op->getBlock())
      op->erase();

  return success();
}
} // namespace mlir::triton::gpu::hcu

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONLOOPSHCU
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionLoops
    : triton::gpu::impl::TritonGPUPartitionLoopsHCUBase<PartitionLoops> {
  using TritonGPUPartitionLoopsHCUBase::TritonGPUPartitionLoopsHCUBase;

  void runOnOperation() override;
};
} // namespace

void PartitionLoops::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to already
  // be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  auto mod = getOperation();
  for (scf::ForOp loop : loops) {
    if (failed(mlir::triton::gpu::hcu::partitionLoop(loop)))
      return signalPassFailure();
  }
}
