// This file implements the data partitioning logic used by the Warp
// Specialization pass for the HCU backend.  The goal is to split
// computations across the two MMA partitions that the hardware provides.
//
// The implementation includes utilities for tracking asynchronous task
// identifiers, propagating them through the def-use graph, and building a
// partitioning scheme that records how each operation should be duplicated
// or split.  The HCU-specific helpers are kept local to avoid symbol
// conflicts with the Hopper implementation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
  namespace hcu {
    // ------------------------------------------------------------------
    //  HCU-specific async task id helpers
    //
    // The Warp Specialization pass marks portions of the IR with an
    // "async_task_id" attribute to indicate which half of the MMA unit they
    // belong to.  The generic Hopper pipeline defines similar helpers, but we
    // replicate a minimal subset here in a nested namespace to avoid ODR
    // collisions when both pipelines are linked into the same binary.
    // ------------------------------------------------------------------

    typedef int AsyncTaskId;

    SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
      (void)op;
      // For HCU WS data partition, always assume two async tasks {0, 1}
      // corresponding to the two MMA partitions.
      SmallVector<AsyncTaskId> asyncTaskIds = {0, 1};
      return asyncTaskIds;
    }

    bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
      return llvm::is_contained(hcu::getAsyncTaskIds(op), asyncTaskId);
    }

    void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
      SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                                  asyncTaskIds.end());
      sort(sortedAsyncTaskIds);
      auto i32Ty = IntegerType::get(op->getContext(), 32);
      auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
      auto vecTy = VectorType::get(size, i32Ty);
      op->setAttr("async_task_id",
                  DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
    }

    void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks) {
      auto asyncTasksVec = hcu::getAsyncTaskIds(op);
      DenseSet<AsyncTaskId> asyncTasksSet(asyncTasksVec.begin(),
                                          asyncTasksVec.end());
      for (auto a : asyncTasks) {
        if (!asyncTasksSet.contains(a))
          asyncTasksVec.push_back(a);
      }
    }

    // Local builder that automatically stamps created ops with async task ids.
    class OpBuilderWithAsyncTaskIds : public OpBuilder {
    public:
      OpBuilderWithAsyncTaskIds(MLIRContext *context) : OpBuilder(context) {}

      explicit OpBuilderWithAsyncTaskIds(Operation *op) : OpBuilder(op) {
        setAsyncTaskIdsFromOp(op);
      }

      void setAsynTaskIdsFromArray(ArrayRef<AsyncTaskId> newAsyncTaskIds) {
        asyncTaskIds.assign(newAsyncTaskIds.begin(), newAsyncTaskIds.end());
      }

      void setAsyncTaskIdsFromOp(Operation *op) {
        setAsynTaskIdsFromArray(getAsyncTaskIds(op));
      }

      template <typename OpTy, typename... Args>
      OpTy createWithAsyncTaskIds(Args &&...args) {
        OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);
        if (!asyncTaskIds.empty())
          setAsyncTaskIds(op, asyncTaskIds);
        return op;
      }

    private:
      SmallVector<AsyncTaskId> asyncTaskIds;
    };
  } // namespace hcu

  // Make AsyncTaskId directly visible as `mlir::AsyncTaskId` inside this file.
  using hcu::AsyncTaskId;

#define DEBUG_TYPE "tritongpu-data-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Utility used by task-id propagation routines.  Returns true if every
// element of `subset` appears in `superset`.
static bool containsAll(const SmallVector<AsyncTaskId> &superset,
                        const SmallVector<AsyncTaskId> &subset) {
  for (AsyncTaskId id : subset) {
    if (!llvm::is_contained(superset, id))
      return false;
  }
  return true;
}

// Simple predicate to identify operations that represent control flow
// boundaries.  These ops are treated specially when propagating async task
// identifiers because they may be inserted/removed by canonicalization passes.
static bool isControlFlowOp(Operation *op) {
  return isa<ReturnOp, FuncOp, scf::YieldOp, scf::ForOp, scf::IfOp>(op);
}

// Walk the function and perform an iterative fix-up of async task ids.
//
// Backward propagation: if a user has a task id that its defining value
// doesn't, and the def is a pure arithmetic op, then give the def the
// missing id.  Forward propagation: if a value with a single use has task
// ids that the user doesn't, copy them into the user.  Special handling is
// required for scf::YieldOp and scf::IfOp as they may lose attributes during
// canonicalization.
static void fixTaskId(triton::FuncOp &funcOp) {
  bool changed = false;
  do {
    changed = false;
    funcOp.walk([&](Operation *op) {
      auto asyncTaskIds = hcu::getAsyncTaskIds(op);
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        // Do not update loads.
        if (isa<LoadOp, DescriptorLoadOp>(defOp))
          continue;
        auto defTaskIds = hcu::getAsyncTaskIds(defOp);
        // Backward propagation: ensure def covers op's task IDs.
        if (!containsAll(defTaskIds, asyncTaskIds)) {
          // Skip control flow ops.
          if (isa<scf::YieldOp, scf::ForOp, scf::IfOp>(op))
            continue;
          // Only propagate backward to arithmetic ops (e.g. constants).
          // Const ops with same value but different task ids can be folded.
          if (defOp->getDialect()->getNamespace() == "arith") {
            LLVM_DEBUG({
              LDBG("backward fixing taskId for");
              defOp->dump();
            });
            hcu::addAsyncTaskIds(defOp, asyncTaskIds);
            changed = true;
            LLVM_DEBUG({
              LDBG("resulting");
              defOp->dump();
            });
          }
        }

        // Forward propagation: ensure op covers def's task IDs
        if (operand.hasOneUse() && !containsAll(asyncTaskIds, defTaskIds)) {
          // YieldOp may lose task attribute during MLIR canonicalization.
          if (isa<scf::YieldOp, scf::IfOp>(op)) {
            LLVM_DEBUG({
              LDBG("forward fixing taskId for");
              defOp->dump();
            });
            hcu::addAsyncTaskIds(op, defTaskIds);
            changed = true;
            LLVM_DEBUG({
              LDBG("resulting");
              defOp->dump();
            });
          }
        }
      }
    });
  } while (changed);
}

// Represents a candidate partitioning of a region of the IR.  The scheme
// records which operations should be duplicated versus split, along with the
// dimension and operand for dot operations.  It also tracks values that need
// to be rematerialized in multiple partitions, and a set of ops to skip if
// rematerialization is not possible.
struct DataPartitionScheme {
  unsigned numPartitions = 0;
  // All operations that participate in this partitioning scheme.
  SetVector<Operation *> ops;
  // Which dimension to partition. For dot, dim 0 means along M dimension, 1
  // means along N dimension.
  DenseMap<Operation *, unsigned> opPartitionDims;
  // For dot, which operand to partition along opPartitionDims.
  DenseMap<Operation *, unsigned> dotPartitionOperand;
  // Ops that are rematerialized through both dimensions.
  DenseMap<Operation *, SetVector<unsigned>> rematerializedOps;
  // Ops should not be partitioned due to rematerialization.
  DenseSet<Operation *> opsToSkip;

  // op with noOpPartitionDim will be duplicated instead of partitioned.
  // Use -2 to avoid conflict with Empty/Tombstone value.
  static const unsigned noOpPartitionDim = ~0U - 2;

  void append(DataPartitionScheme &other) {
    for (auto op : other.ops)
      ops.insert(op);
    for (auto op : other.opPartitionDims)
      opPartitionDims.insert(op);
    for (auto op : other.dotPartitionOperand)
      dotPartitionOperand.insert(op);
    for (auto &op : other.rematerializedOps)
      rematerializedOps.insert(op);
    for (auto op : other.opsToSkip)
      opsToSkip.insert(op);
  }

  bool partitionIsCompatible() { return true; }

  bool isValidPartitionDim(unsigned dim) const {
    return dim < numPartitions || dim == DataPartitionScheme::noOpPartitionDim;
  }

  unsigned flipPartitionDim(unsigned dim) const {
    if (dim == DataPartitionScheme::noOpPartitionDim)
      return dim;
    return numPartitions - 1 - dim;
  }

  bool isPartitioned(Operation *op) const {
    return opPartitionDims.contains(op) || rematerializedOps.contains(op);
  }

  bool isSkipped(Operation *op) const { return opsToSkip.contains(op); }

  void undoPartition(Operation *op) {
    if (opPartitionDims.contains(op)) {
      opPartitionDims.erase(op);
      ops.remove(op);
      opsToSkip.insert(op);
    }
  }

  void dump() const {
    LDBG("=================== DataPartitionScheme ====================");
    LDBG(" numPartitions " << numPartitions);
    LDBG(" ops to partition:");
    for (auto &op : ops) {
      std::string operand;
      if (dotPartitionOperand.contains(op)) {
        operand = "operand " + std::to_string(dotPartitionOperand.at(op));
      }
      assert(opPartitionDims.contains(op) && "missing partition dim");
      LDBG(" dim " << opPartitionDims.at(op) << " " << operand);
      op->dump();
    }
    LDBG("\n");
    if (!rematerializedOps.empty()) {
      LDBG(" ops to rematerialize\n");
      for (auto &op : rematerializedOps) {
        op.first->dump();
        LDBG(" along dim ");
        for (auto &dim : op.second) {
          LDBG(dim << " ");
        }
      }
      LDBG("\n");
    }

    if (!opsToSkip.empty()) {
      LDBG(" ops to skip\n");
      for (auto &op : opsToSkip)
        op->dump();
      LDBG("\n");
    }

    LDBG("===========================================================");
  };
};

static SmallVector<int64_t> getShape(Type type) {
  if (auto descType = dyn_cast<MemDescType>(type))
    return {descType.getShape().begin(), descType.getShape().end()};
  else if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return {tensorType.getShape().begin(), tensorType.getShape().end()};
  else if (auto tensorDescType = dyn_cast<TensorDescType>(type))
    return {tensorDescType.getBlockType().getShape().begin(),
            tensorDescType.getBlockType().getShape().end()};
  else if (auto ptrType = dyn_cast<PointerType>(type))
    return getShape(ptrType.getPointeeType());
  return {};
}

static SmallVector<int64_t> getShape(Value v) { return getShape(v.getType()); }

static bool needToSlice(Value v, unsigned dim, int size) {
  if (dim == DataPartitionScheme::noOpPartitionDim)
    return true;
  auto shape = getShape(v);
  return shape.size() > dim && shape[dim] > size;
}

// Duplicate the op for different partition dims.
static bool rematerializeOp(Operation *op, DataPartitionScheme &partitionScheme,
                            unsigned currentDim) {
  // Bail out if op is already rematerialized.
  if (partitionScheme.rematerializedOps.contains(op)) {
    partitionScheme.rematerializedOps[op].insert(currentDim);
    return true;
  }

  if (isa<LocalAllocOp, arith::ConstantOp>(op)) {
    // assert op has a conflicting partition dim.
    auto existingDim = partitionScheme.opPartitionDims[op];
    assert(existingDim != currentDim && "op has no conflicting partition dim");
    partitionScheme.rematerializedOps[op].insert(existingDim);
    partitionScheme.rematerializedOps[op].insert(currentDim);
    // Undo the partition of the dependency ops in the backward slice.
    SetVector<Operation *> slice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    (void)getBackwardSlice(op, &slice);
    for (auto depOp : slice)
      partitionScheme.undoPartition(depOp);
    return true;
  }
  return false;
}

static bool getBackwardSliceToPartition(Value v,
                                        DataPartitionScheme &partitionScheme,
                                        unsigned currentDim) {
  assert(partitionScheme.isValidPartitionDim(currentDim) && "invalid dim");
  if (!needToSlice(v, currentDim, partitionScheme.numPartitions))
    return true;
  if (auto op = v.getDefiningOp()) {
    // Check dim compatibility
    if (!partitionScheme.ops.insert(op)) {
      if (!isControlFlowOp(op) &&
          partitionScheme.opPartitionDims[op] != currentDim) {
        // Duplicate the op if possible.
        if (!rematerializeOp(op, partitionScheme, currentDim)) {
          LLVM_DEBUG({
            LDBG("incompatible partitioning during backwards:");
            LDBG("dim " << currentDim);
            op->dump();
          });
          return false;
        }
      }
      return true;
    }
    partitionScheme.opPartitionDims[op] = currentDim;

    // Flip dim when op is trans
    if (isa<TransOp, MemDescTransOp>(op))
      currentDim = partitionScheme.flipPartitionDim(currentDim);

    if (auto expandDimsOp = dyn_cast<ExpandDimsOp>(op)) {
      // currentDim is the dim after expansion.
      assert(expandDimsOp.getAxis() != currentDim &&
             "expanded dim always has shape 1");
      // Parition along currentDim - 1 for ExpandDimsOp.
      if (expandDimsOp.getAxis() < currentDim)
        currentDim--;
    }

    // Recusively process operands backwards.
    if (op->hasTrait<OpTrait::Elementwise>() ||
        isa<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp,
            BroadcastOp, ExpandDimsOp, MakeRangeOp, SplatOp, ConvertLayoutOp,
            LoadOp, TransOp, MemDescTransOp,
            AtomicRMWOp, triton::AddPtrOp, DescriptorLoadOp,
            LocalLoadOp,
            nvidia_gpu::TMEMAllocOp, nvidia_gpu::TMEMLoadOp, FpToFpOp>(op)) {
      for (Value operand : op->getOperands())
        if (!getBackwardSliceToPartition(operand, partitionScheme, currentDim))
          return false;
    } else if (isa<triton::gpu::LocalStoreOp>(op)) {
      for (Value operand : op->getOperands())
        if (!getBackwardSliceToPartition(operand, partitionScheme, currentDim))
          return false;
    } else if (isa<MemDescIndexOp>(op)) {
      for (Value operand : op->getOperands())
        if (!getBackwardSliceToPartition(operand, partitionScheme, currentDim + 1))
          return false;
    } else if (auto dotOp = dyn_cast<DotOpInterface>(op)) {
      if (!getBackwardSliceToPartition(dotOp.getA(), partitionScheme, currentDim))
        return false;
      if (!getBackwardSliceToPartition(dotOp.getB(), partitionScheme, currentDim))
        return false;
      if (!getBackwardSliceToPartition(dotOp->getOperand(2), partitionScheme, currentDim))
        return false;
      partitionScheme.dotPartitionOperand[dotOp] = currentDim == 0 ? 0 : 1;
    } else if (isa<ttng::ReinterpretTensorDescOp, MakeTensorDescOp>(op)) {
      return true;
    } else if (auto allocOp = dyn_cast<triton::gpu::LocalAllocOp>(op)) {
      // Recurse into the source operand so that the data chain feeding the
      // allocation (e.g. truncf -> exp2 -> ... -> dot) is also partitioned.
      // Without this, the original full-width chain stays alive because the
      // local_alloc still references it.
      if (allocOp.getSrc())
        if (!getBackwardSliceToPartition(allocOp.getSrc(), partitionScheme,
                                         currentDim))
          return false;
      return true;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // track yield value
      // find result index of v
      unsigned resultIndex = 0;
      for (int i = 0; i < op->getNumResults(); ++i) {
        if (op->getResult(i) == v) {
          resultIndex = i;
          break;
        }
      }
      partitionScheme.ops.insert(ifOp.thenYield());
      partitionScheme.opPartitionDims[ifOp.thenYield()] = currentDim;
      partitionScheme.ops.insert(ifOp.elseYield());
      partitionScheme.opPartitionDims[ifOp.elseYield()] = currentDim;
      auto thenYieldArg = ifOp.thenYield().getOperand(resultIndex);
      auto elseYieldArg = ifOp.elseYield().getOperand(resultIndex);
      if (getBackwardSliceToPartition(thenYieldArg, partitionScheme,
                                      currentDim))
        return false;
      if (!getBackwardSliceToPartition(elseYieldArg, partitionScheme,
                                       currentDim))
        return false;
    } else {
      llvm_unreachable("Unexpected op");
    }
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
      // Make sure the forOp itself participates in the partition scheme so that
      // it can be sliced later.
      if (!partitionScheme.ops.insert(forOp)) {
        if (!isControlFlowOp(forOp) &&
            partitionScheme.opPartitionDims[forOp] != currentDim) {
          LLVM_DEBUG({
            LDBG("incompatible partitioning for forOp during backwards:");
            LDBG("dim " << currentDim);
            forOp->dump();
          });
          return false;
        }
      }
      partitionScheme.opPartitionDims[forOp] = currentDim;
      // track initial value
      auto initArg = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
      if (!getBackwardSliceToPartition(initArg, partitionScheme, currentDim))
        return false;
      // track yield value
      auto yieldArg = forOp.getYieldedValues()[bbArg.getArgNumber() - 1];
      if (!getBackwardSliceToPartition(yieldArg, partitionScheme, currentDim))
        return false;
    }
  }

  return true;
};

// Return false if the partition is not possible.
static bool getForwardSliceToPartition(Value v,
                                       DataPartitionScheme &partitionScheme,
                                       unsigned currentDim,
                                       DenseSet<Value> &seen) {
  assert(partitionScheme.isValidPartitionDim(currentDim) && "invalid dim");
  if (!seen.insert(v).second)
    return true;
  if (!needToSlice(v, currentDim, partitionScheme.numPartitions))
    return true;

  // Recusively process operands forwards.
  unsigned originalDim = currentDim;
  for (Operation *depOp : v.getUsers()) {
    currentDim = originalDim;
    // Flip dim when op is trans
    if (isa<TransOp, MemDescTransOp>(depOp))
      currentDim = partitionScheme.flipPartitionDim(currentDim);

    // Check dim compatibility
    if (!partitionScheme.ops.insert(depOp)) {
      if (!isControlFlowOp(depOp) &&
          partitionScheme.opPartitionDims[depOp] != currentDim) {
        LLVM_DEBUG({
          LDBG("incompatible partitioning during forwards:");
          depOp->dump();
        });
        return false;
      }
      // YieldOp can be partitioned multiple times, one for each of its
      // operands.
      if (!isa<scf::YieldOp>(depOp))
        continue;
    }

    partitionScheme.opPartitionDims[depOp] = currentDim;

    auto onlyUsedByAtomicStore = [](Value v) {
      SetVector<Operation *> forwardSlice;
      getForwardSlice(v, &forwardSlice);
      Operation *atomicStore;
      for (auto op : forwardSlice) {
        if (isa<AtomicRMWOp, DescriptorReduceOp>(op)) {
          atomicStore = op;
          break;
        }
      }

      if (!atomicStore)
        return false;

      // Check all ops in fowardSlice are only connected to atomicStore
      SmallVector<Operation *> queue = {atomicStore};
      forwardSlice.remove(atomicStore);
      while (!queue.empty()) {
        auto op = queue.back();
        queue.pop_back();
        for (Value operand : op->getOperands()) {
          if (auto defOp = operand.getDefiningOp()) {
            if (forwardSlice.contains(defOp)) {
              forwardSlice.remove(defOp);
              queue.push_back(defOp);
            }
          }
        }
      }

      return forwardSlice.empty();
    };

    if (auto dotOp = dyn_cast<DotOpInterface>(depOp)) {
      // WarpGroupDotOp and generic tt.dot share the same partition rules. Dots
      // reached only via forward (e.g. PV dot after QK root in flash-attn)
      // must still get dotPartitionOperand or sliceOp will assert.
      if ((currentDim == 0 && v == dotOp.getB()) ||
          (currentDim == 1 && v == dotOp.getA())) {
        // Partitioning along K of A/B: only valid if the dot output is a
        // partial result consumed only by atomic store.
        Value dotOut = dotOp->getResult(0);
        if (onlyUsedByAtomicStore(dotOut)) {
          partitionScheme.dotPartitionOperand[dotOp] =
              v == dotOp.getA() ? 0 : 1;
          currentDim = DataPartitionScheme::noOpPartitionDim;
        } else {
          LLVM_DEBUG({
            auto opnd = (v == dotOp.getA()) ? "A" : "B";
            LDBG("skip partitioning along K of " << opnd << " of dot\n");
            dotOp.dump();
          });
          return false;
        }
      } else {
        partitionScheme.dotPartitionOperand[dotOp] = currentDim == 0 ? 0 : 1;
      }
    }

    for (Value result : depOp->getResults())
      if (!getForwardSliceToPartition(result, partitionScheme, currentDim,
                                      seen))
        return false;

    if (auto yieldOp = dyn_cast<scf::YieldOp>(depOp)) {
      auto parentOp = yieldOp->getParentOp();
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == v) {
          partitionScheme.ops.insert(parentOp);
          if (!getForwardSliceToPartition(
                  parentOp->getResult(operand.getOperandNumber()),
                  partitionScheme, currentDim, seen))
            return false;
          ;
        }
      }
    }
  }

  return true;
};

// Compute a closure of all ops originated from
// or being dependent on by the root op.
static bool getSliceToPartition(Value root,
                                DataPartitionScheme &partitionScheme,
                                unsigned currentDim) {
  if (!getBackwardSliceToPartition(root, partitionScheme, currentDim))
    return false;
  DataPartitionScheme forwardPartitionScheme = partitionScheme;
  DenseSet<Value> seen;
  LLVM_DEBUG({
    LDBG("backward slice:\n");
    partitionScheme.dump();
  });
  bool forwardSuccess = getForwardSliceToPartition(root, forwardPartitionScheme,
                                                   currentDim, seen);
  LLVM_DEBUG({
    LDBG("forward slice:\n");
    forwardPartitionScheme.dump();
  });
  // Merge the two partition schemes
  partitionScheme.append(forwardPartitionScheme);
  if (!forwardSuccess)
    return false;

  for (auto op : forwardPartitionScheme.ops) {
    // skip ops that have noOpPartitionDim
    currentDim = partitionScheme.opPartitionDims[op];
    if (currentDim == DataPartitionScheme::noOpPartitionDim)
      continue;
    if (op->hasTrait<OpTrait::Elementwise>() ||
        isa<StoreOp, DescriptorStoreOp, AtomicRMWOp>(op)) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (!getBackwardSliceToPartition(operand.get(), partitionScheme,
                                         currentDim))
          return false;
      }
    } else if (isa<nvidia_gpu::WarpGroupDotOp, nvidia_gpu::TCGen5MMAOp>(op)) {
      unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
      if (!getBackwardSliceToPartition(op->getOperand(opndIndx),
                                       partitionScheme, currentDim))
        return false;
      Value accumulator;
      if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op)) {
        accumulator = dotOp.getC();
      } else if (auto dotOp = dyn_cast<nvidia_gpu::TCGen5MMAOp>(op)) {
        accumulator = dotOp.getD();
      }

      if (currentDim == 0 && opndIndx == 0 ||
          currentDim == 1 && opndIndx == 1) {
        // Hanlde accumulator
        if (!getBackwardSliceToPartition(accumulator, partitionScheme,
                                         currentDim))
          return false;
      } else {
        // slice the other operand
        unsigned otherOpndIndx = 1 - opndIndx;
        if (!getBackwardSliceToPartition(op->getOperand(otherOpndIndx),
                                         partitionScheme, 1 - currentDim))
          return false;
        // Hanlde accumulator
        if (!getBackwardSliceToPartition(accumulator, partitionScheme,
                                         DataPartitionScheme::noOpPartitionDim))
          return false;
      }
    }
  }

  return true;
}

static bool computePartitionScheme(triton::FuncOp &funcOp,
                                   DataPartitionScheme &partitionScheme) {
  // Use dot to drive the partition
  SetVector<Operation *> dots;

  // check all dot ops that have more than one async task id
  funcOp.walk([&](Operation *op) {
    if (isa<DotOpInterface>(op))
      dots.insert(op);
  });

  if (dots.empty())
    return true;

  // Checking if all dots can be partitioned in the same way
  int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
  for (auto op : dots) {
    if (partitionScheme.isPartitioned(op) || partitionScheme.isSkipped(op)) {
      continue;
    }

    // partition along M first, otherwise along N
    Value opndA = cast<DotOpInterface>(op).getA();
    Value opndB = cast<DotOpInterface>(op).getB();
    Value accumulator = cast<DotOpInterface>(op)->getResult(0);

    auto dotType = accumulator.getType();
    LLVM_DEBUG({
      LDBG("Computing partition scheme for");
      op->dump();
      LDBG("\n");
    });

    auto shapePerCTA = getShapePerCTA(dotType);
    if (shapePerCTA.size() != 2) {
      LDBG("partition not possible: shapePerCTA " << shapePerCTA.size());
      return false;
    }
    int sliceSizeM = shapePerCTA[0] / 2;
    int sliceSizeN = shapePerCTA[1] / 2;
    SmallVector<unsigned, 2> partitionDim, partitionSize;

    if (sliceSizeM >= 16) {
      partitionDim.push_back(0);
      partitionSize.push_back(sliceSizeM);
    }

    if (sliceSizeN >= 16) {
      partitionDim.push_back(1);
      partitionSize.push_back(sliceSizeN);
    }

    if (partitionDim.empty()) {
      LDBG("Partition not available: " << sliceSizeM << " " << sliceSizeN);
      return false;
    }

    if (partitionScheme.numPartitions == 0) {
      partitionScheme.numPartitions = 2;
    } else {
      if (partitionScheme.numPartitions != 2) {
        LDBG("partition not possible, in conflict with previous partition\n");
        return false;
      }
    }

    bool success = false;
    for (int i = 0; i < partitionDim.size(); ++i) {
      // Partition the slice closure
      auto trialPartitionScheme = partitionScheme;
      LLVM_DEBUG(
          { LDBG("Trying partition along " << partitionDim[i] << " \n"); });

      if (getSliceToPartition(accumulator, trialPartitionScheme,
                              partitionDim[i])) {
        success = true;
        partitionScheme = trialPartitionScheme;
      }

      LLVM_DEBUG({
        LDBG(" Trial slice:\n");
        trialPartitionScheme.dump();
        LDBG("\n");
      });

      if (success)
        break;
    }

    if (!success) {
      LDBG("partition not possible\n");
      return false;
    }
  }

  LLVM_DEBUG({
    LDBG("\n");
    LDBG(" Final slice:\n");
    partitionScheme.dump();
    LDBG("\n");
  });

  return !partitionScheme.ops.empty();
}

// For each op to be rematerialized, create a new op and replace its user with
// the new op.
static void rewriteRematerializedOps(triton::FuncOp &funcOp,
                                     DataPartitionScheme &partitionScheme) {
  if (partitionScheme.rematerializedOps.empty())
    return;

  hcu::OpBuilderWithAsyncTaskIds builder(funcOp.getContext());

  // For each rematerialized op, create a new op and replace its user with it.
  for (auto opDim : partitionScheme.rematerializedOps) {
    auto oldOp = opDim.first;
    builder.setInsertionPoint(oldOp);
    builder.setAsyncTaskIdsFromOp(oldOp);

    // Skip the first dim which will be using the original op.
    for (unsigned i = 1; i < opDim.second.size(); i++) {
      unsigned dim = opDim.second[i];
      LLVM_DEBUG({
        LDBG("rewriting op along dim " << dim << ":");
        oldOp->dump();
      });

      Operation *newOp = nullptr;
      if (auto allocOp = dyn_cast<LocalAllocOp>(oldOp)) {
        // create a memdesc view
        auto memdescType = allocOp.getType();
        SmallVector<int64_t> shape = getShape(memdescType);
        int sliceSize = shape[dim] / partitionScheme.numPartitions;
        shape[dim] = sliceSize;
        // 保持底层分配形状不变，只缩小逻辑 shape。这样 allocShape 仍然是
        // 原来的整块（例如 32x32），而当前 view 的 shape 是切分后的
        // 16x32，便于后端通过 allocShape 与 shape 的差异识别出这是
        // 一个 subview，并正确计算 LDS 偏移。
        SmallVector<int64_t> allocShape(memdescType.getAllocShape().begin(),
                                        memdescType.getAllocShape().end());
        auto slicedMemdescType = MemDescType::get(
            shape, memdescType.getElementType(), memdescType.getEncoding(),
            memdescType.getMemorySpace(), memdescType.getMutableMemory(),
            allocShape);
        SmallVector<int32_t> offsets(shape.size(), 0);
        auto viewOp = builder.createWithAsyncTaskIds<MemDescSubsliceOp>(
            allocOp.getLoc(), slicedMemdescType, allocOp.getResult(), offsets);
        newOp = viewOp;
      } else if (isa<arith::ConstantOp>(oldOp)) {
        newOp = builder.clone(*oldOp);
      } else {
        llvm_unreachable("Unexpected op");
      }

      LLVM_DEBUG({
        LDBG("new op:");
        newOp->dump();
      });

      hcu::setAsyncTaskIds(newOp, hcu::getAsyncTaskIds(oldOp));
      partitionScheme.ops.insert(newOp);
      partitionScheme.opPartitionDims[newOp] = dim;

      // replace the users that have same partition dim with the op.
      auto dimMatches = [&](OpOperand &operand) {
        auto user = operand.getOwner();
        assert(partitionScheme.opPartitionDims.contains(user) &&
               "user not partitioned");
        unsigned userDim = partitionScheme.opPartitionDims[user];
        if (isa<TransOp, MemDescTransOp>(user)) {
          // flip userDim for trans
          userDim = partitionScheme.flipPartitionDim(userDim);
        } else if (auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(user)) {
          // infer userDim for dot
          assert(partitionScheme.dotPartitionOperand.contains(user) &&
                 "no operand info");
          unsigned opndIndx = partitionScheme.dotPartitionOperand[user];
          if (userDim == 0 && opndIndx == 1 || userDim == 1 && opndIndx == 0)
            userDim = DataPartitionScheme::noOpPartitionDim;
        }

        if (userDim != dim)
          return false;
        LLVM_DEBUG({
          LDBG("replacing user with dim " << userDim << ":");
          user->dump();
        });
        return true;
      };

      oldOp->getResult(0).replaceUsesWithIf(newOp->getResult(0), dimMatches);
    }
  }
}

static Operation *sliceOp(Value v, int offset, IRMapping &mappings,
                          IRMapping &reverseMappings,
                          DataPartitionScheme &partitionScheme);

static Operation *sliceOp(Operation *op, int offset, IRMapping &mappings,
                          IRMapping &reverseMappings,
                          DataPartitionScheme &partitionScheme) {
  if (!partitionScheme.ops.contains(op))
    return op;
  if (mappings.contains(op))
    return mappings.lookupOrNull(op);
  if (reverseMappings.contains(op))
    return op;

  unsigned dim = partitionScheme.opPartitionDims[op];
  unsigned numOfPartitions = partitionScheme.numPartitions;

  LLVM_DEBUG({
    LDBG("slicing along dim " << dim << ":");
    op->dump();
  });

  SmallVector<AsyncTaskId, 3> asyncTaskIds = {0, 1};
  SmallVector<AsyncTaskId, 3> sliceTaskIds;
  // We are slicing the op for consumer only
  sliceTaskIds.push_back(asyncTaskIds[offset]);

  hcu::OpBuilderWithAsyncTaskIds builder(op->getContext());
  builder.setAsynTaskIdsFromArray(sliceTaskIds);
  auto cloneAndSetResultType = [&](Operation *op) {
    builder.setInsertionPoint(op);
    auto newOp = builder.clone(*op, mappings);
    hcu::setAsyncTaskIds(newOp, sliceTaskIds);
    mappings.map(op, newOp);
    reverseMappings.map(newOp, op);
    // set result shape
    if (!op->getResults().empty()) {
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      bool needRetype = true;
      if (dim == DataPartitionScheme::noOpPartitionDim) {
        // Just duplicate the op for noOpPartitionDim
        needRetype = false;
      } else if (isa<nvidia_gpu::WarpGroupDotOp, nvidia_gpu::TCGen5MMAOp>(op)) {
        assert(partitionScheme.dotPartitionOperand.contains(op) &&
               "no operand info");
        unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
        if (dim == 0 && opndIndx == 1 || dim == 1 && opndIndx == 0) {
          needRetype = false;
        }
      }

      if (needRetype) {
        if (auto type = dyn_cast<MemDescType>(v.getType())) {
          SmallVector<int64_t> shape{type.getShape().begin(),
                                     type.getShape().end()};
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          // change encoding for ttng.tensor_memory_encoding to match gen5.
          if (auto tmem = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(
                  type.getEncoding())) {
            Attribute accEncoding =
                triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
                    builder.getContext(),
                    dim == 0 ? tmem.getBlockM() / 2 : tmem.getBlockM(),
                    dim == 1 ? tmem.getBlockN() / 2 : tmem.getBlockN(),
                    tmem.getColStride(), tmem.getCTASplitM(),
                    tmem.getCTASplitN(), tmem.getTwoCTAs());
            auto newType = MemDescType::get(shape, type.getElementType(),
                                            accEncoding, type.getMemorySpace(),
                                            type.getMutableMemory());
            newV.setType(newType);
          } else {
            auto newType = MemDescType::get(
                shape, type.getElementType(), type.getEncoding(),
                type.getMemorySpace(), type.getMutableMemory());
            newV.setType(newType);
          }
        } else if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
          SmallVector<int64_t> shape{type.getShape().begin(),
                                     type.getShape().end()};
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          auto newType = RankedTensorType::get(shape, type.getElementType(),
                                               type.getEncoding());
          newV.setType(newType);
        } else if (auto type = dyn_cast<TensorDescType>(v.getType())) {
          auto blockType = type.getBlockType();
          SmallVector<int64_t> shape{blockType.getShape().begin(),
                                     blockType.getShape().end()};
          int sliceSize = shape[dim] / numOfPartitions;
          shape[dim] = sliceSize;
          auto newBlockType = RankedTensorType::get(
              shape, blockType.getElementType(), blockType.getEncoding());
          auto newType =
              TensorDescType::get(builder.getContext(), newBlockType);
          newV.setType(newType);
        }
      }
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
    return newOp;
  };

  // slice operands first
  Operation *newOp;
  if ((dim == DataPartitionScheme::noOpPartitionDim) ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<ConvertLayoutOp, BroadcastOp, SplatOp, ExpandDimsOp, FpToFpOp,
          AtomicRMWOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto allocOp = dyn_cast<triton::gpu::LocalAllocOp>(op)) {
    // Empty pipelined buffers (e.g. matmul A/B: local_alloc : () -> memdesc)
    // are shared by load (full tile) and MMA (M-sliced). Cloning them creates
    // extra LDS (192KB instead of 128KB). Keep the original alloc; LocalLoad
    // creates memdesc_subslice views. Only clone allocs that materialize a
    // tensor src (e.g. FA P staging: local_alloc %tensor).
    if (!allocOp.getSrc()) {
      mappings.map(op, op);
      reverseMappings.map(op, op);
      for (Value v : op->getResults()) {
        mappings.map(v, v);
        reverseMappings.map(v, v);
      }
      return op;
    }
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (isa<triton::gpu::LocalStoreOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto tmemLdOp = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    auto srcTy = mappings.lookupOrNull(tmemLdOp.getSrc()).getType();
    auto type = cast<MemDescType>(srcTy);
    auto tmem = cast<nvidia_gpu::TensorMemoryEncodingAttr>(type.getEncoding());

    RankedTensorType oldRetType = tmemLdOp.getType();
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = mlir::triton::gpu::lookupNumWarps(op);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    builder.setInsertionPoint(op);
    // The source op is already sliced at this point, so srcTy, type, tmem is
    // sliced. We use getTmemCompatibleLayout to get a block layout that is for
    // the sliced tmem here.
    auto newDistributedEncoding =
        nvidia_gpu::getDefaultLayoutForTmemLdSt(type, numWarps, CTALayout);

    // oldRetType is the desired output, we slice it and convert from the
    // compatible layout to the sliced desired output.
    SmallVector<int64_t> shape{oldRetType.getShape().begin(),
                               oldRetType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newAccType = RankedTensorType::get(shape, oldRetType.getElementType(),
                                            newDistributedEncoding);
    auto ld = builder.createWithAsyncTaskIds<triton::nvidia_gpu::TMEMLoadOp>(
        op->getLoc(), newAccType, mappings.lookupOrNull(tmemLdOp.getSrc()));

    auto newType = RankedTensorType::get(shape, oldRetType.getElementType(),
                                         oldRetType.getEncoding());
    auto cvtOp = builder.createWithAsyncTaskIds<ConvertLayoutOp>(op->getLoc(),
                                                                 newType, ld);
    auto v = tmemLdOp->getResult(0);
    auto newV = cvtOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
    newOp = cvtOp;
  } else if (auto tmemAllocOp = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    // Check for src.
    if (tmemAllocOp.getSrc()) {
      // src is blocked layout. apply convert layout on src
      auto srcTy = cast<RankedTensorType>(
          mappings.lookupOrNull(tmemAllocOp.getSrc()).getType());

      // convert from srcTy to a compatible blocked layout.
      auto retShapePerCTA = getShapePerCTA(srcTy);
      int numWarps = mlir::triton::gpu::lookupNumWarps(op);
      auto CTALayout = getCTALayout(srcTy.getEncoding());
      builder.setInsertionPoint(op);

      // calculate new tmem type.
      auto retType = cast<MemDescType>(tmemAllocOp.getType());
      SmallVector<int64_t> shape{retType.getShape().begin(),
                                 retType.getShape().end()};
      int sliceSize = shape[dim] / numOfPartitions;
      shape[dim] = sliceSize;
      auto tmem =
          cast<nvidia_gpu::TensorMemoryEncodingAttr>(retType.getEncoding());
      auto accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
          builder.getContext(),
          dim == 0 ? tmem.getBlockM() / 2 : tmem.getBlockM(),
          dim == 1 ? tmem.getBlockN() / 2 : tmem.getBlockN(),
          tmem.getColStride(), tmem.getCTASplitM(), tmem.getCTASplitN(),
          tmem.getTwoCTAs());
      auto newType = MemDescType::get(shape, retType.getElementType(),
                                      accEncoding, retType.getMemorySpace(),
                                      retType.getMutableMemory());

      auto newDistributedEncoding =
          nvidia_gpu::getDefaultLayoutForTmemLdSt(retType, numWarps, CTALayout);
      auto newAccType = RankedTensorType::get(
          srcTy.getShape(), srcTy.getElementType(), newDistributedEncoding);
      auto cvtOp = builder.createWithAsyncTaskIds<ConvertLayoutOp>(
          op->getLoc(), newAccType,
          mappings.lookupOrNull(tmemAllocOp.getSrc()));

      // replace tmemAllocOp with alloc, where the src is cvtOp.
      auto alloc =
          builder.createWithAsyncTaskIds<triton::nvidia_gpu::TMEMAllocOp>(
              op->getLoc(), newType, cvtOp);

      auto v = tmemAllocOp->getResult(0);
      auto newV = alloc->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
      newOp = alloc;
    } else
      newOp = cloneAndSetResultType(op);
  } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    builder.setInsertionPoint(op);
    auto valAttr = cast<DenseElementsAttr>(constOp.getValueAttr());
    auto valType = cast<ShapedType>(valAttr.getType());
    SmallVector<int64_t> shape{valType.getShape().begin(),
                               valType.getShape().end()};
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    auto newValType = valType.clone(shape);
    auto newValAttr = valAttr.resizeSplat(newValType);
    newOp = builder.createWithAsyncTaskIds<arith::ConstantOp>(op->getLoc(),
                                                              newValAttr);
    // Do not drop original task id as constant folding may lose one constant.
    hcu::setAsyncTaskIds(newOp, hcu::getAsyncTaskIds(op));
    auto v = op->getResult(0);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (auto makeRangeOp = dyn_cast<MakeRangeOp>(op)) {
    builder.setInsertionPoint(op);
    int newRangeStart = makeRangeOp.getStart();
    int newRangeEnd = makeRangeOp.getEnd();
    int sliceSize = (newRangeEnd - newRangeStart) / numOfPartitions;
    newRangeStart += offset * sliceSize;
    newRangeEnd = newRangeStart + sliceSize;
    auto v = op->getResult(0);
    auto type = cast<RankedTensorType>(v.getType());
    auto newType = RankedTensorType::get({sliceSize}, builder.getI32Type(),
                                         type.getEncoding());
    newOp = builder.createWithAsyncTaskIds<MakeRangeOp>(
        op->getLoc(), newType, newRangeStart, newRangeEnd);
    auto newV = newOp->getResult(0);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (isa<StoreOp, LoadOp, LocalLoadOp>(op)) {
    if (auto localLdOp = dyn_cast<LocalLoadOp>(op)) {
      // For pipelined empty buffers, do not clone LocalAlloc / shrink
      // MemDescIndex — view the shared full tile with memdesc_subslice.
      // For LocalAlloc with a tensor src (FA P), recurse so the producer
      // chain is sliced and the cloned alloc is used directly.
      Value src = localLdOp.getSrc();
      Operation *srcDef = src.getDefiningOp();
      auto srcAlloc = dyn_cast_or_null<LocalAllocOp>(srcDef);
      bool sliceSrcAlloc = srcAlloc && srcAlloc.getSrc();
      if (sliceSrcAlloc) {
        sliceOp(src, offset, mappings, reverseMappings, partitionScheme);
        if (Value mappedSrc = mappings.lookupOrNull(src))
          mappings.map(localLdOp.getSrc(), mappedSrc);
      } else if (dim != DataPartitionScheme::noOpPartitionDim) {
        Value base = mappings.lookupOrNull(src);
        if (!base)
          base = src;
        if (auto memTy = dyn_cast<MemDescType>(base.getType())) {
          SmallVector<int64_t> shape{memTy.getShape().begin(),
                                     memTy.getShape().end()};
          if (shape.size() > dim && shape[dim] > 0 &&
              shape[dim] % static_cast<int64_t>(numOfPartitions) == 0 &&
              shape[dim] / static_cast<int64_t>(numOfPartitions) < shape[dim]) {
            int64_t sliceSize =
                shape[dim] / static_cast<int64_t>(numOfPartitions);
            int32_t addrOffset =
                static_cast<int32_t>(offset * static_cast<int32_t>(sliceSize));
            SmallVector<int32_t> offsets(shape.size(), 0);
            offsets[dim] = addrOffset;
            shape[dim] = sliceSize;

            SmallVector<int64_t> allocShape(memTy.getAllocShape().begin(),
                                            memTy.getAllocShape().end());
            auto slicedTy =
                MemDescType::get(shape, memTy.getElementType(),
                                 memTy.getEncoding(), memTy.getMemorySpace(),
                                 memTy.getMutableMemory(), allocShape);

            builder.setInsertionPoint(op);
            auto viewOp =
                builder.createWithAsyncTaskIds<MemDescSubsliceOp>(
                    op->getLoc(), slicedTy, base, offsets);

            mappings.map(localLdOp.getSrc(), viewOp.getResult());
          }
        }
      }
      // Only slice non-address operands (e.g. token); do not recurse into
      // empty-buffer / memdesc_index src (handled above).
      for (Value operand : op->getOperands())
        if (operand != localLdOp.getSrc())
          sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    } else {
      // LoadOp / StoreOp: only slice operands. Do NOT also AddPtr by
      // offset*sliceSize — MakeRange / index chains already carry the
      // partition row offset after operand slicing. Adding both double-
      // offsets task1 loads (e.g. FA Q: make_range(32,64) + AddPtr(+32)).
      for (Value operand : op->getOperands())
        sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    }
    newOp = cloneAndSetResultType(op);
  } else if (isa<DescriptorLoadOp, DescriptorStoreOp>(op)) {
    SmallVector<int64_t> shape;
    Value coordVal;
    if (auto loadOp = dyn_cast<DescriptorLoadOp>(op)) {
      sliceOp(loadOp.getDesc(), offset, mappings, reverseMappings,
              partitionScheme);
      coordVal = loadOp.getIndices()[dim];
      shape = getShape(loadOp.getResult());
    } else if (auto storeOp = dyn_cast<DescriptorStoreOp>(op)) {
      sliceOp(storeOp.getDesc(), offset, mappings, reverseMappings,
              partitionScheme);
      coordVal = storeOp.getIndices()[dim];
      shape = getShape(storeOp.getSrc());
    }
    auto newCoordVal = coordVal;
    if (offset) {
      builder.setInsertionPointAfter(coordVal.getDefiningOp());
      Value offsetVal = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          op->getLoc(), offset * shape[dim] / numOfPartitions, 32);
      newCoordVal = builder.createWithAsyncTaskIds<arith::AddIOp>(
          op->getLoc(), coordVal, offsetVal);
      mappings.map(coordVal, newCoordVal);
      reverseMappings.map(newCoordVal, coordVal);
    }

    newOp = cloneAndSetResultType(op);
    if (isa<DescriptorLoadOp>(op)) {
      // map load result
      auto v = op->getResult(0);
      auto newV = newOp->getResult(0);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);
    }
  } else if (auto tensorDescOp = dyn_cast<MakeTensorDescOp>(op)) {
    newOp = cloneAndSetResultType(op);
  } else if (auto memDescIndexOp = dyn_cast<MemDescIndexOp>(op)) {
    // Keep the full stage tile (e.g. 128x128). Do not slice/clone the
    // underlying empty LocalAlloc — MMA takes an M-subslice later via
    // LocalLoad's memdesc_subslice, while load still stores the full tile.
    builder.setInsertionPoint(op);
    newOp = builder.clone(*op, mappings);
    hcu::setAsyncTaskIds(newOp, sliceTaskIds);
    mappings.map(op, newOp);
    reverseMappings.map(newOp, op);
    for (auto [oldV, newV] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      mappings.map(oldV, newV);
      reverseMappings.map(newV, oldV);
    }
    // Intentionally do not retype: result stays full MxK / KxN.
  } else if (auto tensorDescOp = dyn_cast<ttng::ReinterpretTensorDescOp>(op)) {
    newOp = cloneAndSetResultType(op);
  } else if (isa<TransOp, MemDescTransOp>(op)) {
    sliceOp(op->getOperand(0), offset, mappings, reverseMappings,
            partitionScheme);
    builder.setInsertionPoint(op);
    auto v = op->getResult(0);
    SmallVector<int64_t> shape = getShape(v.getType());
    int sliceSize = shape[dim] / numOfPartitions;
    shape[dim] = sliceSize;
    Type newType;
    if (auto descType = dyn_cast<MemDescType>(v.getType())) {
      newType = MemDescType::get(
          shape, descType.getElementType(), descType.getEncoding(),
          descType.getMemorySpace(), descType.getMutableMemory());
    } else if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      newType = RankedTensorType::get(shape, tensorType.getElementType(),
                                      tensorType.getEncoding());
    } else {
      llvm_unreachable("unsupported type");
    }
    builder.setInsertionPoint(op);
    newOp = builder.clone(*op, mappings);
    hcu::setAsyncTaskIds(newOp, sliceTaskIds);
    auto newV = newOp->getResult(0);
    newV.setType(newType);
    mappings.map(v, newV);
    reverseMappings.map(newV, v);
  } else if (isa<DotOpInterface>(op)) {
    assert(partitionScheme.dotPartitionOperand.contains(op) &&
           "no operand info");
    unsigned opndIndx = partitionScheme.dotPartitionOperand[op];
    LDBG("slicing operand " << opndIndx << "\n");
    sliceOp(op->getOperand(opndIndx), offset, mappings, reverseMappings,
            partitionScheme);
    if (dim == 0 && opndIndx == 1 || dim == 1 && opndIndx == 0) {
      // slice the other operand
      unsigned otherOpndIndx = 1 - opndIndx;
      LDBG("slicing operand " << otherOpndIndx << "\n");
      sliceOp(op->getOperand(otherOpndIndx), offset, mappings, reverseMappings,
              partitionScheme);
    }
    // Hanlde accumulator
    Value accumulator = cast<DotOpInterface>(op)->getOperand(2);
    sliceOp(accumulator, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // Add new loop arguments
    SmallVector<Value> newLoopArgs;
    for (auto initArg : forOp.getInitArgs())
      newLoopArgs.push_back(initArg);
    DenseMap<int, int> newArgIdices;
    for (unsigned i = 0; i < forOp.getInitArgs().size(); i++) {
      auto initArg = forOp.getInitArgs()[i];
      Value newInitArg;
      auto newInitArgOp =
          sliceOp(initArg, offset, mappings, reverseMappings, partitionScheme);
      if (auto bbArg = dyn_cast<BlockArgument>(initArg)) {
        // find the corresponding new block argument
        Block *parentBlock = bbArg.getOwner();
        unsigned argIndex = parentBlock->getNumArguments();
        for (unsigned i = 0; i < parentBlock->getNumArguments(); ++i) {
          if (parentBlock->getArgument(i) == bbArg) {
            argIndex = i;
            break;
          }
        }
        assert(argIndex < parentBlock->getNumArguments() &&
               "new init argment not found");
        Region *parentRegion = parentBlock->getParent();
        Region &newParentRegion =
            newInitArgOp->getRegion(parentRegion->getRegionNumber());
        newInitArg = parentRegion->getArgument(argIndex);
      } else {
        newInitArg = mappings.lookupOrNull(initArg);
      }

      if (newInitArg) {
        assert(newInitArg != initArg && "value not sliced");
        newLoopArgs.append({newInitArg});
        forOp.getBody()->insertArgument(forOp.getBody()->getNumArguments(),
                                        newInitArg.getType(), forOp.getLoc());
        newArgIdices[i] = newLoopArgs.size() - 1;
      }
    }

    // Create newForOp and take the region of forOp
    builder.setInsertionPoint(op);
    auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newLoopArgs);
    assert(newForOp.getRegionIterArgs().size() ==
           newForOp.getInitArgs().size());
    newForOp->setAttrs(forOp->getAttrs());
    partitionScheme.ops.insert(newForOp);
    newOp = newForOp;

    // Replace forOp with newForOp
    newForOp.getRegion().takeBody(forOp.getRegion());
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    op->setAttr("to_be_removed", builder.getUnitAttr());

    // Map new loop arguments
    for (auto argIndex : newArgIdices) {
      Value v = newForOp.getResult(argIndex.first);
      Value newV = newForOp.getResult(argIndex.second);
      mappings.map(v, newV);
      reverseMappings.map(newV, v);

      auto regionArg = newForOp.getRegionIterArg(argIndex.first);
      auto newRegionArg = newForOp.getRegionIterArg(argIndex.second);
      mappings.map(regionArg, newRegionArg);
      reverseMappings.map(newRegionArg, regionArg);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // Slice the yield op and update if results
    auto thenYieldOp = ifOp.thenYield();
    auto elseYieldOp = ifOp.elseYield();
    auto newThenYieldOp = sliceOp(thenYieldOp, offset, mappings,
                                  reverseMappings, partitionScheme);
    sliceOp(elseYieldOp, offset, mappings, reverseMappings, partitionScheme);
    assert(newThenYieldOp->getNumOperands() > ifOp->getNumResults() &&
           "no need to slice if op");
    // Clone ifOp with updated results but re-use the original regions.
    builder.setInsertionPoint(op);
    SmallVector<Type, 4> newResultTypes;
    for (auto thenResult : thenYieldOp.getResults()) {
      newResultTypes.push_back(thenResult.getType());
    }
    auto newIfOp = builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                             ifOp.getCondition());
    // Move the original regions to the cloned operation.
    newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    newOp = newIfOp;
    newIfOp->setAttrs(ifOp->getAttrs());
    partitionScheme.ops.insert(newIfOp);
    ifOp->setAttr("to_be_removed", builder.getUnitAttr());

    // Replace ifOp with newIfOp
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      ifOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));

    // Map if results based on the mapping for yield
    for (auto &v : thenYieldOp->getOpOperands()) {
      auto newV = mappings.lookupOrNull(v.get());
      if (newV) {
        int operandIndex = v.getOperandNumber();
        // find the corresponding operand index of newV in newYieldOp
        int newOperandIndex = -1;
        for (int i = 0; i < newThenYieldOp->getNumOperands(); ++i) {
          if (newThenYieldOp->getOperand(i) == newV) {
            newOperandIndex = i;
            break;
          }
        }
        assert(newOperandIndex >= 0 && "newV not found in newYieldOp");
        auto newResult = newIfOp.getResult(operandIndex);
        auto newSlicedResult = newIfOp.getResult(newOperandIndex);
        mappings.map(newResult, newSlicedResult);
        reverseMappings.map(newSlicedResult, newResult);
      }
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    int num = yieldOp.getNumOperands();
    for (int i = 0; i < num; i++) {
      auto operand = yieldOp.getOperand(i);
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
      if (auto newV = mappings.lookupOrNull(operand))
        yieldOp->insertOperands(op->getNumOperands(), newV);
    }
    newOp = op;
  } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    assert(reduceOp.getAxis() != dim &&
           "reduce should not happen on the partitioned dimension");
    for (Value operand : op->getOperands())
      sliceOp(operand, offset, mappings, reverseMappings, partitionScheme);
    newOp = cloneAndSetResultType(op);
    // recursively set async task ids for child ops
    newOp->walk(
        [&](Operation *childOp) { hcu::setAsyncTaskIds(childOp, sliceTaskIds); });
  } else {
    llvm_unreachable("unsupported op type");
  }

  LLVM_DEBUG({
    LDBG("resulting");
    newOp->dump();
  });
  mappings.map(op, newOp);
  reverseMappings.map(newOp, op);
  return newOp;
}

static Operation *sliceOp(Value v, int offset, IRMapping &mappings,
                          IRMapping &reverseMappings,
                          DataPartitionScheme &partitionScheme) {
  if (auto op = v.getDefiningOp()) {
    return sliceOp(op, offset, mappings, reverseMappings, partitionScheme);
  } else {
    assert(isa<BlockArgument>(v) && "value is not an operation or block ");
    auto bbArg = cast<BlockArgument>(v);
    Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
    return sliceOp(bbAargOwner, offset, mappings, reverseMappings,
                   partitionScheme);
  }
}

static bool doDeepCleanup(triton::FuncOp &funcOp,
                          DataPartitionScheme &partitionScheme) {
  SmallVector<Operation *> opsToDelete;
  DenseSet<Operation *> opsCanBeTriviallyDead;

  do {
    opsToDelete.clear();
    opsCanBeTriviallyDead.clear();

    // Identify root ops that are not used so to be deleted.
    funcOp.walk([&](Operation *op) {
      if (isa<scf::YieldOp>(op))
        return;
      if (!partitionScheme.ops.contains(op))
        return;

      // Ignore the side effect of ops that are already sliced. The
      // resulting ops preserve the side effect.
      if (!isMemoryEffectFree(op))
        opsCanBeTriviallyDead.insert(op);

      bool notUsed = true;
      for (auto result : op->getResults()) {
        if (!result.getUsers().empty()) {
          notUsed = false;
          break;
        }
      }
      if (notUsed)
        opsToDelete.push_back(op);
    });

    LLVM_DEBUG({
      LDBG("opsToDelete:\n");
      for (auto op : opsToDelete) {
        LDBG("op: ");
        op->dump();
      }
      LDBG("\n");
    });

    if (opsToDelete.empty())
      return true;

    // Delete root ops.
    for (auto op : opsToDelete) {
      partitionScheme.ops.remove(op);
      op->erase();
    }

    LLVM_DEBUG({
      LDBG("prior to loop arg deletion:");
      funcOp.dump();
    });

    // delete block arguments
    RewritePatternSet cleanUpPatterns(funcOp.getContext());
    populateForOpDeadArgumentElimination(cleanUpPatterns);
    scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns,
                                            funcOp.getContext());
    scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns,
                                           funcOp.getContext());
    if (applyPatternsGreedily(funcOp, std::move(cleanUpPatterns)).failed()) {
      return false;
    }
  } while (!opsToDelete.empty());
  return true;
}

static bool doDataPartition(triton::FuncOp &funcOp,
                            unsigned numConsumerGroups) {
  DataPartitionScheme partitionScheme;
  if (!computePartitionScheme(funcOp, partitionScheme)) {
    if (numConsumerGroups > 1) {
      LDBG("computePartitionScheme failed when requested");
      return false;
    }
    return true;
  }

  bool hasLocalStoreOp = false;
  for (auto *op : partitionScheme.ops) {
    if (isa<triton::gpu::LocalStoreOp>(op)) {
      hasLocalStoreOp = true;
      break;
    }
  }
  if (hasLocalStoreOp) {
    LDBG("local store op found, skipping data partition");
    return true;
  }

  // Rewrite the rematerialized ops.
  LDBG("Rewriting rematerialized Ops");
  rewriteRematerializedOps(funcOp, partitionScheme);
  LLVM_DEBUG({
    LDBG("After rewriting rematerialized Ops:");
    funcOp.dump();
    LDBG("\n");
    LDBG(" Final parition scheme:\n");
    partitionScheme.dump();
  });

  // Slice the ops.
  for (int i = 0; i < partitionScheme.numPartitions; i++) {
    IRMapping mappings, reverseMappings;
    LDBG("partitioning op for task " << i + 1 << ":\n");
    int numOps = partitionScheme.ops.size();
    for (int j = 0; j < numOps; j++) {
      auto op = partitionScheme.ops[j];
      sliceOp(op, i, mappings, reverseMappings, partitionScheme);
    }

    // clean up
    LLVM_DEBUG({
      LDBG("prior to clean up:");
      funcOp.dump();
    });
    SmallVector<Operation *> opsToDelete;
    for (auto op : partitionScheme.ops) {
      if (op->hasAttr("to_be_removed"))
        opsToDelete.push_back(op);
    }
    for (auto op : opsToDelete) {
      partitionScheme.ops.remove(op);
      op->erase();
    }
  }

  LLVM_DEBUG({
    LDBG("prior to final cleanup:");
    funcOp.dump();
  });

  // Make sure original ops are not used
  if (!doDeepCleanup(funcOp, partitionScheme)) {
    LDBG("final cleanup failed");
    return false;
  }

  // Make sure original ops are not used
  LLVM_DEBUG({
    LDBG("after partition");
    funcOp.dump();
    LDBG("\n");
  });

  fixTaskId(funcOp);
  return true;
}

} // namespace mlir
namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUDATAPARTITION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
// HCU-specific data partition pass that reuses the generic data partition
// algorithm above to split work along the data dimension before lowering
// warp-specialized loops into warp groups.
struct TritonGPUDataPartition
    : mlir::triton::gpu::impl::TritonGPUDataPartitionBase<
          TritonGPUDataPartition> {
  using mlir::triton::gpu::impl::TritonGPUDataPartitionBase<
      TritonGPUDataPartition>::TritonGPUDataPartitionBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // For the HCU pipeline we currently have 1 load partition (4 warps) and
    // 1 MMA partition (8 warps). We split the MMA work into two consumer
    // groups so that each group only handles 4 warps, which matches the
    // hardware restriction when using set_vgpr.
    constexpr unsigned numConsumerGroups = 2;

    WalkResult result = mod.walk([&](triton::FuncOp funcOp) {
      if (!doDataPartition(funcOp, numConsumerGroups))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
