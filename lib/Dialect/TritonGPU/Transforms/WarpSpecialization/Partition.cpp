#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Use.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// Partition
//===----------------------------------------------------------------------===//

bool Partition::hasOp(Operation *op) const {
  if (!hasPartition(op)) {
    return false;
  }
  auto partitionIds = getPartitionIds(op);
  return partitionIds.contains(getIndex());
}

void Partition::iterateInputs(scf::ForOp loop,
                              function_ref<void(OpOperand &)> callback) const {
  for (Operation *op : getOps()) {
    visitNestedOperands(op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      std::optional<SetVector<int>> partitionIds;
      if (hasPartition(value.getDefiningOp()))
        partitionIds = getPartitionIds(value.getDefiningOp());
      if (value.getParentBlock() != loop.getBody())
        return;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        assert(arg.getOwner() == loop.getBody());
        // Ignore the induction variable.
        if (arg == loop.getInductionVar())
          return;
        // This value originates from a previous iteration.
        assert(llvm::is_contained(loop.getRegionIterArgs(), arg));
        callback(operand);
      } else if (!partitionIds ||
                 !llvm::is_contained(*partitionIds, getIndex())) {
        // This value originates from a different partition in the same
        // iteration.
        assert(value.getDefiningOp()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}

void Partition::iterateOutputs(
    scf::ForOp loop,
    function_ref<void(Operation *, OpOperand &)> callback) const {
  for (Operation *op : getOps()) {
    for (OpOperand &use : op->getUses()) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (!owner) {
        continue;
      }
      std::optional<SetVector<int>> partitionIds;
      if (hasPartition(owner))
        partitionIds = getPartitionIds(owner);
      if (isa<scf::YieldOp>(owner)) {
        // This value is used in a subsequent iteration.
        callback(owner, use);
      } else if (!partitionIds ||
                 !llvm::is_contained(*partitionIds, getIndex())) {
        // This value is used in a different partition in the same iteration.
        callback(owner, use);
      }
    }
  }
}

void Partition::iterateDefs(
    scf::ForOp loop, function_ref<void(OpResult, unsigned)> callback) const {
  iterateInputs(loop, [&](OpOperand &input) {
    auto [def, distance] = getDefinitionAndDistance(loop, input.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def, distance);
  });
}

void Partition::iterateUses(
    scf::ForOp loop,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
  SmallVector<std::tuple<OpResult, OpOperand *, unsigned>> uses;
  iterateOutputs(loop, [&](Operation *owner, OpOperand &use) {
    uses.emplace_back(cast<OpResult>(use.get()), &use, 0);
  });
  while (!uses.empty()) {
    auto [output, use, distance] = uses.pop_back_val();
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());
    if (!owner) {
      continue;
    }
    if (!isa<scf::YieldOp>(owner)) {
      callback(output, *use, distance);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(output, &use, distance + 1);
  }
}

//===----------------------------------------------------------------------===//
// PartitionSet
//===----------------------------------------------------------------------===//

Partition *PartitionSet::addPartition(unsigned stage) {
  partitions.push_back(std::make_unique<Partition>(partitions.size(), stage));
  return partitions.back().get();
}

Partition *PartitionSet::getPartition(unsigned idx) {
  return partitions[idx].get();
}

const Partition *PartitionSet::getPartition(unsigned idx) const {
  return partitions[idx].get();
}

Partition *PartitionSet::getPartition(Operation *op) {
  auto id = getPartitionIds(op);
  assert(id.size() == 1);
  return getPartition(id[0]);
}

FailureOr<PartitionSet> PartitionSet::fromLoop(scf::ForOp loop) {
  auto stages = loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
  if (!stages)
    return failure();

  auto tag = loop->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
  if (!tag)
    return failure();

  PartitionSet result;
  result.tag = tag.getInt();
  for (auto [idx, attr] : llvm::enumerate(stages)) {
    auto stage = dyn_cast<IntegerAttr>(attr);
    if (!stage || stage.getInt() < 0) {
      return mlir::emitError(loop.getLoc(), "partition stages attribute '")
             << kPartitionStagesAttrName << "' has invalid element " << attr;
    }

    result.partitions.push_back(
        std::make_unique<Partition>(idx, stage.getInt()));
  }

  for (Operation &op : loop.getBody()->without_terminator()) {
    auto attrs = getPartitionIds(&op);
    for (auto idx : attrs) {
      if (idx < 0 || idx >= result.partitions.size())
        return mlir::emitError(op.getLoc(), "invalid partition index ") << idx;
      result.partitions[idx]->addOp(&op);
    }
  }

  return result;
}

void PartitionSet::dump() const {
  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    llvm::errs() << "=== PARTITION #" << i << " ===\n";
    for (Operation *op : partition.getOps()) {
      op->print(llvm::errs(), OpPrintingFlags().skipRegions());
      llvm::errs() << "\n";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";
}

namespace mlir::triton::gpu {

Partition *WarpSchedule::getRootPartition() {
  return getNumPartitions() > 0 ? PartitionSet::getPartition(0u) : nullptr;
}

const Partition *WarpSchedule::getRootPartition() const {
  return getNumPartitions() > 0 ? partitions[0].get() : nullptr;
}

Partition *WarpSchedule::getPartition(Operation *op) {
  if (!hasPartition(op))
    return nullptr;
  auto ids = getPartitionIds(op);
  assert(ids.size() == 1 && "Expected a uniquely assigned partition");
  return PartitionSet::getPartition(ids[0]);
}

const Partition *WarpSchedule::getPartition(Operation *op) const {
  if (!hasPartition(op))
    return nullptr;
  auto ids = getPartitionIds(op);
  assert(ids.size() == 1 && "Expected a uniquely assigned partition");
  return partitions[ids[0]].get();
}

bool WarpSchedule::isScheduled(Operation *op) const { return hasPartition(op); }

void WarpSchedule::insert(Partition *partition, Operation *op) {
  setPartition(op, partition);
}

bool WarpSchedule::trySchedule(Partition *partition, Operation *op) {
  if (isScheduled(op))
    return false;
  insert(partition, op);
  return true;
}

void WarpSchedule::iterateDefs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, unsigned)> callback) const {
  partition->iterateDefs(loop, callback);
}

void WarpSchedule::iterateUses(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
  partition->iterateUses(loop, callback);
}

void WarpSchedule::iterateOutputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(Operation *, OpOperand &)> callback) const {
  partition->iterateOutputs(loop, callback);
}

void WarpSchedule::serialize(scf::ForOp loop) const {
  Builder b(loop.getContext());
  SmallVector<Attribute> stages;
  for (const Partition &partition : getPartitions())
    stages.push_back(b.getI32IntegerAttr(partition.getStage()));
  loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
  setWarpSpecializeTag(loop, getTag());
}

LogicalResult WarpSchedule::verify(scf::ForOp loop) const {
  return succeeded(PartitionSet::fromLoop(loop)) ? success() : failure();
}

FailureOr<WarpSchedule> WarpSchedule::deserialize(scf::ForOp loop) {
  FailureOr<PartitionSet> partitionSet = PartitionSet::fromLoop(loop);
  if (failed(partitionSet))
    return failure();

  WarpSchedule schedule;
  schedule.tag = partitionSet->getTag();
  schedule.partitions = partitionSet->takePartitions();
  return schedule;
}

void WarpSchedule::eraseFrom(Operation *op) {
  if (!op)
    return;
  op->removeAttr(kPartitionAttrName);
  op->removeAttr(kPartitionOutputsAttrName);
  op->removeAttr(kPartitionStagesAttrName);
  op->removeAttr(kWarpSpecializeTagAttrName);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        eraseFrom(&nested);
}

void setPartition(Operation *op, ArrayRef<int> partitionIds) {
  Builder b(op->getContext());
  auto sorted = llvm::to_vector(partitionIds);
  llvm::sort(sorted);
  auto sortedSet = SetVector<int>(sorted.begin(), sorted.end());
  op->setAttr(kPartitionAttrName, b.getDenseI32ArrayAttr(sorted));
  // The TritonGPU dialect verifier requires every non-yield/poison child op of
  // a partitioned op to carry ttg.partition, and the parent's partition to
  // contain the union of the children's partitions. Ops with "combine" regions
  // such as tt.reduce get a partition during scheduling, but the ops inside
  // their regions (e.g. the arith.maxnumf combiner) may retain a stale
  // partition from an earlier pass (e.g. AnnotateRootPartition assigning root
  // 0). Propagate the parent's partition to any region child whose current
  // partition is not already a subset of the new parent partition, so the
  // verifier stays satisfied. Children that already hold a legitimate
  // sub-partition (e.g. independently partitioned scf.for/scf.if bodies) are
  // left untouched.
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (Operation &child : block.getOperations()) {
        if (isa<scf::YieldOp, ub::PoisonOp>(child))
          continue;
        if (!hasPartition(&child) ||
            !llvm::all_of(getPartitionIds(&child),
                          [&](int id) { return sortedSet.contains(id); }))
          child.setAttr(kPartitionAttrName, b.getDenseI32ArrayAttr(sorted));
      }
    }
  }
}

void setPartitionOutputs(Operation *op,
                         ArrayRef<SetVector<int>> partitionOutputsIds) {
  if (partitionOutputsIds.empty()) {
    op->removeAttr(kPartitionOutputsAttrName);
    return;
  }
  SmallVector<Attribute> attrs;
  Builder b(op->getContext());
  for (auto partitionIds : partitionOutputsIds) {
    auto sorted = llvm::to_vector(partitionIds);
    llvm::sort(sorted);
    attrs.push_back(b.getDenseI32ArrayAttr(sorted));
  }
  op->setAttr(kPartitionOutputsAttrName, b.getArrayAttr(attrs));
}

void setPartition(Operation *op, const SetVector<int> &partitionIds) {
  SmallVector<int> partitions(partitionIds.begin(), partitionIds.end());
  setPartition(op, partitions);
}

void setPartition(Operation *op, Partition *partition) {
  SmallVector<int> partitions{partition->getIndex()};
  setPartition(op, partitions);
  partition->addOp(op);
}

void setPartition(Operation *op, const SetVector<Partition *> &partitions) {
  SmallVector<int> partitionIds;
  for (auto partition : partitions) {
    partitionIds.push_back(partition->getIndex());
    partition->addOp(op);
  }
  setPartition(op, partitionIds);
}

void setWarpSpecializeTag(Operation *op, int tag) {
  Builder b(op->getContext());
  op->setAttr(kWarpSpecializeTagAttrName, b.getI32IntegerAttr(tag));
}

} // namespace mlir::triton::gpu
