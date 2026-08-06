#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/WdraSplitPlan.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/AbarrierIdManager.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace tta = triton::amdgpu;
namespace hcu = triton::HCU;

//===----------------------------------------------------------------------===//
// getPartitionScheme
//===----------------------------------------------------------------------===//

namespace {
struct PipelinedLoad {
  PipelinedLoad(Operation *loadOp);

  TypedValue<RankedTensorType> getResult() const {
    return cast<TypedValue<RankedTensorType>>(loadOp->getResult(0));
  }
  unsigned getLoadSizeInBytes() const {
    return type.getNumElements() * type.getElementTypeBitWidth() / 8;
  }
  LogicalResult determineLiveRange(Block &container, DominanceInfo &domInfo,
                                   PostDominanceInfo &postDomInfo,
                                   WarpSchedule &schedule);

  Operation *loadOp;
  /// True when loadOp is amdgpu.matrix_load_to_local (global→LDS, no local_store).
  bool isMls = false;
  RankedTensorType type;
  SharedEncodingTrait sharedEnc;

  SmallVector<Operation *, 1> allocOps;
  SmallVector<Operation *, 1> liveBeforeOps;
  SmallVector<std::pair<Operation *, bool>, 0> liveUntilOps;
  SmallVector<Operation *, 1> asyncUsers;
};

PipelinedLoad::PipelinedLoad(Operation *loadOp) : loadOp(loadOp) {
  if (auto mls = dyn_cast<tta::MatrixLoadToLocalOp>(loadOp)) {
    isMls = true;
    auto memDesc = cast<MemDescType>(mls.getDest().getType());
    sharedEnc = cast<SharedEncodingTrait>(memDesc.getEncoding());
    if (auto alloc = mls.getDest().getDefiningOp<LocalAllocOp>())
      allocOps.push_back(alloc);
    // Prefer the LocalLoad result type for createAlloc shape/element info.
    for (Operation *allocOp : allocOps) {
      for (Operation *user : allocOp->getUsers()) {
        if (auto localLoad = dyn_cast<LocalLoadOp>(user)) {
          type = localLoad.getType();
          return;
        }
      }
    }
    // No LocalLoad yet; synthesize a ranked tensor type from the memdesc.
    // createAlloc only needs shape + element type from this.
    auto ctx = loadOp->getContext();
    auto order = getOrder(memDesc);
    auto ctaLayout = getCTALayout(sharedEnc);
    SmallVector<unsigned> sizePerThread(order.size(), 1);
    SmallVector<unsigned> threadsPerWarp(order.size(), 1);
    SmallVector<unsigned> warpsPerCTA(order.size(), 1);
    if (!threadsPerWarp.empty())
      threadsPerWarp.back() = 1;
    type = RankedTensorType::get(
        memDesc.getShape(), memDesc.getElementType(),
        BlockedEncodingAttr::get(ctx, sizePerThread, threadsPerWarp,
                                 warpsPerCTA, order, ctaLayout));
    return;
  }

  type = cast<RankedTensorType>(loadOp->getResult(0).getType());
  sharedEnc = getSharedEncoding(loadOp);
}

struct PipelinedMMA {
  PipelinedMMA(DotOpInterface mmaOp) : mmaOp(mmaOp) {}

  DotOpInterface mmaOp;
};
} // namespace

static std::pair<SmallVector<PipelinedLoad>, SmallVector<PipelinedMMA>>
getPartitionScheme(scf::ForOp loop, const WarpSchedule &schedule) {
  SmallVector<PipelinedLoad> loads;
  SmallVector<PipelinedMMA> mmas;

  for (Operation &op : loop.getOps()) {
    if (isa<LoadOp>(op)) {
      auto &load = loads.emplace_back(&op);
      for (Operation *user : op.getUsers()) {
        if (schedule.getPartition(user) == schedule.getPartition(&op) &&
            isa<LocalAllocOp>(user))
          load.allocOps.push_back(user);
      }
      continue;
    }
    if (auto mls = dyn_cast<tta::MatrixLoadToLocalOp>(op)) {
      auto &load = loads.emplace_back(&op);
      // allocOps already populated in PipelinedLoad ctor for MLS; keep only
      // those scheduled in the load partition.
      load.allocOps.erase(
          llvm::remove_if(load.allocOps,
                          [&](Operation *alloc) {
                            return schedule.getPartition(alloc) !=
                                   schedule.getPartition(&op);
                          }),
          load.allocOps.end());
      (void)mls;
      continue;
    }
  }

  for (auto mmaOp : loop.getOps<DotOpInterface>()) {
    mmas.emplace_back(mmaOp);
  }

  return {std::move(loads), std::move(mmas)};
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static std::pair<Value, Value> postIncrementModulo(ImplicitLocOpBuilder &b,
                                                   Value index, Value phase,
                                                   unsigned numStages) {
  auto intCst = [&](int value) {
    return b.create<arith::ConstantIntOp>(value, 32);
  };
  Value nextIndex = b.create<arith::AddIOp>(index, intCst(1));
  Value nextPhase = b.create<arith::XOrIOp>(phase, intCst(1));

  Value rollover = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextIndex,
                                           intCst(numStages));
  nextIndex = b.create<arith::SelectOp>(rollover, intCst(0), nextIndex);
  nextPhase = b.create<arith::SelectOp>(rollover, nextPhase, phase);

  return {nextIndex, nextPhase};
}

static std::pair<BlockArgument, BlockArgument>
addIndexAndPhase(PartitionBuilder &b, scf::ForOp &loop, unsigned numStages,
                 Value epilogue = {}) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);

  // Index and phase both start at 0.
  loop = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  auto newArgs = loop.getRegionIterArgs().take_back(2);
  BlockArgument index = newArgs[0];
  BlockArgument phase = newArgs[1];

  // Post-increment the index and phase.
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(yield);

  auto [nextIndex, nextPhase] = postIncrementModulo(b, index, phase, numStages);
  if (epilogue) {
    nextIndex = b.create<arith::SelectOp>(epilogue, nextIndex, index);
    nextPhase = b.create<arith::SelectOp>(epilogue, nextPhase, phase);
  }
  yield->insertOperands(yield.getNumOperands(), {nextIndex, nextPhase});

  return {index, phase};
}

static Value getUserPrecondition(ImplicitLocOpBuilder &b, scf::ForOp loop,
                                 Operation *domOp) {
  // If the use is inside a loop besides the actual loop being pipelined, we
  // have to hoist the use up to that loop, otherwise the barriers will be
  // inserted in the loop.
  for (Operation *userLoop;
       loop != (userLoop = domOp->getParentOfType<LoopLikeOpInterface>());)
    domOp = userLoop;
  assert(loop->isProperAncestor(domOp));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  Value trueVal = b.create<arith::ConstantOp>(b.getBoolAttr(true));

  Value userPred = trueVal;
  Operation *parentOp = domOp;
  b.setInsertionPoint(loop.getBody()->findAncestorOpInBlock(*domOp));
  while (loop != (parentOp = parentOp->getParentOp())) {
    assert(!isa<LoopLikeOpInterface>(parentOp));
    auto ifOp = dyn_cast<scf::IfOp>(parentOp);
    if (!ifOp) {
      llvm::report_fatal_error(
          "FIXME: unsupported parent operation for MMA user");
    }
    Value cond = ifOp.getCondition();
    if (domOp->getParentRegion() == &ifOp.getElseRegion())
      cond = b.create<arith::XOrIOp>(cond, trueVal);
    userPred = b.create<arith::AndIOp>(userPred, cond);
  }

  return userPred;
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true);
}

//===----------------------------------------------------------------------===//
// Load Pipelining
//===----------------------------------------------------------------------===//

// Find the last operation that consumes the in-memory result of a load. This
// only looks at the current loop iteration.
static LogicalResult
findSharedMemorySinkOps(Value value, SmallVectorImpl<Operation *> &sinkOps) {
  for (Operation *user : value.getUsers()) {
    if (isa<LocalLoadOp>(user)) {
      sinkOps.push_back(user);
    } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      if (failed(findSharedMemorySinkOps(user->getResult(0), sinkOps)))
        return failure();
    } else if (isa<tta::MatrixLoadToLocalOp>(user)) {
      // MLS producer writing into this buffer; not a consumer sink.
      continue;
    } else {
      return mlir::emitWarning(user->getLoc(),
                               "failed to warp specialize: cannot handle sink "
                               "of in-memory load operation");
    }
  }
  return success();
}

LogicalResult PipelinedLoad::determineLiveRange(Block &container,
                                                DominanceInfo &domInfo,
                                                PostDominanceInfo &postDomInfo,
                                                WarpSchedule &schedule) {
  // Find the liveBefore and liveUntil operations of the load.
  llvm::MapVector<Partition *, SmallVector<Operation *>> regSinks, shmemSinks;
  if (isMls) {
    // MLS: tensor never lands in registers from the load op; sinks are
    // LocalLoads of the destination alloc.
    for (Operation *allocOp : allocOps) {
      SmallVector<Operation *> sinkOps;
      if (failed(findSharedMemorySinkOps(allocOp->getResult(0), sinkOps)))
        return failure();
      for (Operation *sinkOp : sinkOps)
        shmemSinks[schedule.getPartition(sinkOp)].push_back(sinkOp);
    }
  } else {
    for (Operation *user : loadOp->getUsers()) {
      auto it = llvm::find(allocOps, user);
      if (it == allocOps.end()) {
        // This is an in-register use of the load. The result must be live before
        // the op. Since it will be loaded out of shared memory, it only needs to
        // be live until the op as well.
        regSinks[schedule.getPartition(user)].push_back(user);
        continue;
      }
      SmallVector<Operation *> sinkOps;
      if (failed(findSharedMemorySinkOps((*it)->getResult(0), sinkOps)))
        return failure();
      for (Operation *sinkOp : sinkOps)
        shmemSinks[schedule.getPartition(sinkOp)].push_back(sinkOp);
    }
  }
  SetVector<Partition *> userPartitions;
  userPartitions.insert_range(llvm::make_first_range(regSinks));
  userPartitions.insert_range(llvm::make_first_range(shmemSinks));

  // The result must be live before all the sinks in each partition.
  for (Partition *userPartition : userPartitions) {
    SmallVector<Operation *> regSink = regSinks.lookup(userPartition);
    SmallVector<Operation *> shmemSink = shmemSinks.lookup(userPartition);

    auto sinks = llvm::to_vector(llvm::concat<Operation *>(regSink, shmemSink));
    Operation *liveBeforeOp = findNearestCommonDominator(sinks, domInfo);
    liveBeforeOp = container.findAncestorOpInBlock(*liveBeforeOp);
    liveBeforeOps.push_back(liveBeforeOp);

    SmallVector<Operation *> shmemTerminals;
    // Prefer a live op in this partition when reading module attrs.
    Operation *attrAnchor =
        !shmemSink.empty()
            ? shmemSink.front()
            : (!container.empty() ? &container.front() : nullptr);
    const bool delayEmptyArrive =
        attrAnchor ? hcu::getEmptyArriveAfterMmacAttr(attrAnchor,
                                                      /*defaultVal=*/true)
                   : true;
    for (Operation *sinkOp : shmemSink) {
      sinkOp = container.findAncestorOpInBlock(*sinkOp);
      // When empty_arrive_after_mmac is set, release the LDS buffer after
      // consumers of the LocalLoad (typically tt.dot), not immediately after
      // local_load. A matching SchedBarrier after mmac is inserted in MMAC.cpp.
      if (delayEmptyArrive) {
        SmallVector<Operation *> consumers;
        if (auto localLoad = dyn_cast<LocalLoadOp>(sinkOp)) {
          for (Operation *user : localLoad.getResult().getUsers()) {
            if (Operation *ancestor = container.findAncestorOpInBlock(*user))
              consumers.push_back(ancestor);
          }
        }
        if (!consumers.empty()) {
          Operation *lastConsumer =
              findNearestCommonPostDominator(consumers, postDomInfo);
          shmemTerminals.push_back(lastConsumer ? lastConsumer : sinkOp);
          continue;
        }
      }
      shmemTerminals.push_back(sinkOp);
    }

    // Normalize the sink op to be one immediately under the loop. Then, the
    // memory must be live until after this operation.
    Operation *lastShmemSink =
        findNearestCommonPostDominator(shmemTerminals, postDomInfo);

    // The memory only needs to be live until before the first register user.
    Operation *liveUntilReg = findNearestCommonDominator(regSink, domInfo);
    if (liveUntilReg)
      liveUntilReg = container.findAncestorOpInBlock(*liveUntilReg);

    // The memory is live until before the first register user or after the last
    // shmem terminal, whichever is later.
    std::pair<Operation *, bool> liveUntilOp = {nullptr, false};
    if (lastShmemSink && liveUntilReg) {
      if (liveUntilReg->isBeforeInBlock(lastShmemSink)) {
        liveUntilOp = {lastShmemSink, true};
      } else {
        liveUntilOp = {liveUntilReg, false};
      }
    } else if (liveUntilReg) {
      liveUntilOp = {liveUntilReg, false};
    } else {
      liveUntilOp = {lastShmemSink, true};
    }
    liveUntilOps.push_back(liveUntilOp);
  }

  return success();
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

namespace {

struct PipelinedLoadGroup {
  Location getLoc();
  void allocateAref(scf::ForOp &loop, int numStages,
                    SmallVector<int> &availableAbarrierIds,
                    const hcu::WdraTopology &topo, int waspNumLoadWarps,
                    int waspNumMmaWarps);
  LogicalResult lowerLoads(WarpSchedule &schedule, DominanceInfo &domInfo,
                           PostDominanceInfo &postDomInfo);

  SmallVector<PipelinedLoad> loads;

  SmallVector<Value> loadBuffers;
  // Barrier IDs laid out as [pair0_stage0, pair0_stage1, ..., pair1_stage0, ...]
  // For OnePTwoC / WASP, numBarrierPairs==1 (shared fan-out to all consumers).
  SmallVector<int, 8> emptyBarIds;
  SmallVector<int, 8> readyBarIds;
  unsigned numBarrierPairs = 1;
  unsigned numStagesStored = 0;
  BlockArgument index;
  BlockArgument phase;
};
} // namespace

namespace {
static void maybeSetAsyncTaskId(Operation *op, unsigned pair,
                                unsigned numPairs) {
  if (numPairs <= 1)
    return;
  op->setAttr("async_task_id",
              DenseI32ArrayAttr::get(op->getContext(),
                                     {static_cast<int32_t>(pair)}));
}
} // namespace

Location PipelinedLoadGroup::getLoc() {
  SmallVector<Location> locs = llvm::map_to_vector(
      loads, [](PipelinedLoad &load) { return load.loadOp->getLoc(); });
  return FusedLoc::get(locs.front().getContext(), locs);
}

void PipelinedLoadGroup::allocateAref(scf::ForOp &loop, int numStages,
                                      SmallVector<int> &availableAbarrierIds,
                                      const hcu::WdraTopology &topo,
                                      int waspNumLoadWarps,
                                      int waspNumMmaWarps) {
  assert(loadBuffers.empty() && "already allocated");

  // Create buffers for each the loads.
  for (PipelinedLoad &load : loads) {
    loadBuffers.push_back(createAlloc(loop, load.type, load.loadOp->getLoc(),
                                      load.sharedEnc, numStages));
  }

  numBarrierPairs = std::max(1u, topo.numBarrierPairs());
  numStagesStored = numStages;

  // Empty-bar participants / ready-bar participants.
  // OnePTwoC keeps the historical init counts (all MMA / load warps on one
  // shared barrier). TwoPTwoC uses per-pair barriers sized to one partition.
  int emptyInitCount =
      topo.kind == hcu::WdraTopoKind::TwoPTwoC
          ? static_cast<int>(topo.warpsPerPartition)
          : waspNumMmaWarps;
  int readyInitCount =
      topo.kind == hcu::WdraTopoKind::TwoPTwoC
          ? static_cast<int>(topo.warpsPerPartition)
          : waspNumLoadWarps;
  int preArrive = topo.enabled() ? static_cast<int>(topo.emptyPreArriveCount())
                                 : 1;

  PartitionBuilder b(getLoc(), loop);
  for (unsigned pair = 0; pair < numBarrierPairs; ++pair) {
    for (auto i : llvm::seq(numStages)) {
      int barId = availableAbarrierIds[pair * numStages * 2 + i * 2];
      createAbarrier(loop, barId, emptyInitCount);
      emptyBarIds.push_back(barId);
      b.create<ROCDL::HCUAbarrierArriveOp>(barId, preArrive);

      barId = availableAbarrierIds[pair * numStages * 2 + i * 2 + 1];
      createAbarrier(loop, barId, readyInitCount);
      readyBarIds.push_back(barId);
    }
  }

  std::tie(index, phase) = addIndexAndPhase(b, loop, numStages);
}

static void lowerTMACopy(PartitionBuilder &b, Partition &loadPartition,
                         StageCluster stageCluster, Operation *op,
                         Value barrier, Value view) {
  Value truePred = b.boolCst(true);
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    auto indices = ttng::translateTMAIndices(
        b, load.getLoc(), load.getDesc().getType().getBlockType().getEncoding(),
        load.getIndices());
    b.createInto<ttng::AsyncTMACopyGlobalToLocalOp>(loadPartition, stageCluster,
                                                    load.getDesc(), indices,
                                                    barrier, view, truePred);
  } else {
    auto gather = cast<DescriptorGatherOp>(op);
    b.createInto<ttng::AsyncTMAGatherOp>(
        loadPartition, stageCluster, gather.getDesc(), gather.getXOffsets(),
        gather.getYOffset(), barrier, view, truePred);
  }
}

LogicalResult PipelinedLoadGroup::lowerLoads(WarpSchedule &schedule,
                                             DominanceInfo &domInfo,
                                             PostDominanceInfo &postDomInfo) {
  // Insert before the group of loads.
  auto firstLoad = llvm::min_element(loads, [&](auto &lhs, auto &rhs) {
    return domInfo.properlyDominates(lhs.loadOp, rhs.loadOp);
  });
  // 在所有 load 的 localAllocs 中找到最前面的一个 op。
  Operation *firstAllocOp = nullptr;
  for (auto &load : loads) {
    for (Operation *allocOp : load.allocOps) {
      if (!firstAllocOp ||
          domInfo.properlyDominates(allocOp, firstAllocOp)) {
        firstAllocOp = allocOp;
      }
    }
  }
  Partition &loadPartition = *schedule.getPartition(firstLoad->loadOp);
  PartitionBuilder b(getLoc(), firstLoad->loadOp);

  // Producer acquire (empty-buffer try-wait).
  //
  // Insertion point must be the earlier of {load, alloc} that still keeps:
  //   - tt.load path: tryWait AFTER the load (alloc consumes load result, so
  //     alloc is after load). local_store is placed after tryWait and must see
  //     the load result.
  //   - MLS path: tryWait BEFORE matrix_load_to_local (empty alloc is the dest
  //     and sits before the load). Index selects must be created at the same
  //     point so they dominate tryWait.
  Operation *tryWaitInsertPt =
      firstAllocOp ? firstAllocOp : firstLoad->loadOp;
  b.setInsertionPoint(tryWaitInsertPt);

  StageCluster stageCluster = getStageCluster(firstLoad->loadOp);
  auto intCst = [&](int value) { return b.create<arith::ConstantIntOp>(value, 32); };

  // Helper: pick stage bar id from a per-pair contiguous [s0, s1, ...] slice.
  auto selectStageBarId = [&](ArrayRef<int> pairBarIds) -> Value {
    Value cur = intCst(pairBarIds.back());
    Value isFirstStage =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, index, intCst(0));
    cur = b.create<arith::SelectOp>(isFirstStage, intCst(pairBarIds.front()),
                                    cur);
    return cur;
  };

  // Per-pair empty try-waits (TwoPTwoC tags async_task_id so SplitMma can
  // move each pair into its Load/MMA partitions).
  SmallVector<ROCDL::HCUAbarrierTryWaitOp> tryWaitOps;
  for (unsigned pair = 0; pair < numBarrierPairs; ++pair) {
    ArrayRef<int> pairEmpty(emptyBarIds.data() + pair * numStagesStored,
                            numStagesStored);
    if (pair == 0)
      b.setInsertionPoint(tryWaitInsertPt);
    else
      b.setInsertionPointAfter(tryWaitOps.back());
    Value curEmptyBarId = selectStageBarId(pairEmpty);
    auto tryWaitOp = b.createInto<ROCDL::HCUAbarrierTryWaitOp>(
        loadPartition, stageCluster, curEmptyBarId, phase);
    maybeSetAsyncTaskId(tryWaitOp, pair, numBarrierPairs);
    tryWaitOps.push_back(tryWaitOp);
  }
  // All local stores must be placed after every producer acquire.  In
  // TwoPTwoC, anchoring them after only the first wait leaves the task-1
  // stores before their own empty-buffer wait after partition splitting.
  Operation *lastProducerAcquire = tryWaitOps.back();

  // Set up the consumer wait. We know the live before ops are the same for all
  // loads since that's how they were grouped.
  DenseMap<Partition *, ROCDL::HCUAbarrierArriveOp> arriveOps;
  for (auto [i, liveBeforeOp] : llvm::enumerate(firstLoad->liveBeforeOps)) {
    // 每一个 userPartition 贡献一个 liveBeforeOp，所以遍历 liveBeforeOp 也就是遍历 userPartition
    Partition &userPartition = *schedule.getPartition(liveBeforeOp);
    StageCluster userStageCluster = getStageCluster(liveBeforeOp);

    for (unsigned pair = 0; pair < numBarrierPairs; ++pair) {
      ArrayRef<int> pairReady(readyBarIds.data() + pair * numStagesStored,
                              numStagesStored);
      ArrayRef<int> pairEmpty(emptyBarIds.data() + pair * numStagesStored,
                              numStagesStored);
      b.setInsertionPoint(liveBeforeOp);
      Value curReadyBarId = selectStageBarId(pairReady);
      auto consumerWait = b.createInto<ROCDL::HCUAbarrierTryWaitOp>(
          userPartition, userStageCluster, curReadyBarId, phase);
      maybeSetAsyncTaskId(consumerWait, pair, numBarrierPairs);

      // 所有的 load 在当前这个 userPartition 中的 liveUntilOp 的集合
      SmallVector<Operation *> liveUntilOps;
      for (PipelinedLoad &load : loads) {
        auto [liveUntilOp, after] = load.liveUntilOps[i];
        if (liveUntilOp)
          liveUntilOps.push_back(after ? liveUntilOp->getNextNode()
                                       : liveUntilOp);
      }
      if (!liveUntilOps.empty()) {
        Operation *liveUntilOp =
            findNearestCommonPostDominator(liveUntilOps, postDomInfo);
        b.setInsertionPoint(liveUntilOp);
        Value curEmptyBarId = selectStageBarId(pairEmpty);
        auto arriveOp = b.createInto<ROCDL::HCUAbarrierArriveOp>(
            userPartition, userStageCluster, curEmptyBarId, 1);
        maybeSetAsyncTaskId(arriveOp, pair, numBarrierPairs);
        // Keep the last arrive for reg-use placement (same as before for 1 pair).
        arriveOps[schedule.getPartition(liveUntilOp)] = arriveOp;
      }
    }
  }

  // 将 Load 的结果放到 loadBuffers 中
  // - tt.load path: local_store(load_result, view)
  // - MLS path: retarget matrix_load_to_local dest to view (no local_store)
  SmallVector<Operation *> storeOps;
  for (auto [load, buffer] : llvm::zip(loads, loadBuffers)) {
    b.setInsertionPoint(load.loadOp);
    Value view = createSingleBufferView(b, buffer, index);
    if (load.isMls) {
      auto mlsOp = cast<tta::MatrixLoadToLocalOp>(load.loadOp);
      // Point matrix_load_to_local at the multi-buffered slice, then retire the
      // per-iteration empty local_alloc.
      mlsOp.getDestMutable().assign(view);
      for (Operation *allocOp : load.allocOps) {
        replaceUsesAndPropagateType(b, allocOp, view);
        allocOp->erase();
      }
      // Treat the async wait (or the MLS op itself) as the end of the "store"
      // so the ready-barrier arrive is placed after LDS write completion.
      Operation *completionOp = mlsOp;
      for (Operation *user : llvm::make_early_inc_range(mlsOp->getUsers())) {
        if (auto commit = dyn_cast<AsyncCommitGroupOp>(user)) {
          for (Operation *waitUser : commit->getUsers()) {
            if (isa<AsyncWaitOp>(waitUser))
              completionOp = waitUser;
          }
        }
      }
      storeOps.push_back(completionOp);
      continue;
    }

    b.setInsertionPointAfter(lastProducerAcquire);
    auto storeOp = b.createInto<LocalStoreOp>(loadPartition, stageCluster,
                                              load.getResult(), view);
    storeOps.push_back(storeOp);
    for (Operation *allocOp : load.allocOps) {
      replaceUsesAndPropagateType(b, allocOp, view);
      allocOp->erase();
    }
    // If there are remaining users, they must be in-register.
    llvm::MapVector<Partition *, SmallVector<OpOperand *>> regUses;
    for (OpOperand &use : load.loadOp->getUses()) {
      auto owner = use.getOwner();
      if (llvm::find(storeOps, use.getOwner()) != storeOps.end())
        continue;
      regUses[schedule.getPartition(use.getOwner())].push_back(&use);
    }
    for (auto &[partition, uses] : regUses) {
      auto users = llvm::to_vector(llvm::map_range(
          uses, [](OpOperand *use) { return use->getOwner(); }));
      if (Operation *arriveOp = arriveOps.lookup(partition))
        users.push_back(arriveOp);
      Operation *loadBeforeOp = findNearestCommonDominator(users, domInfo);
      b.setInsertionPoint(loadBeforeOp);
      StageCluster userStageCluster = getStageCluster(loadBeforeOp);
      Value loaded = b.createInto<LocalLoadOp>(*partition, userStageCluster,
                                               load.type, view);
      b.createInto<ttng::FenceAsyncSharedOp>(*partition, userStageCluster,
                                             /*bCluster=*/false);
      for (OpOperand *use : uses)
        use->set(loaded);
    }
  }

  // 标记 loadGroup 的结束 — per-pair ready arrives for TwoPTwoC.
  auto loadAndStoreOps = llvm::to_vector(llvm::concat<Operation *>(storeOps, 
                                         map_range(loads, [](PipelinedLoad &load) { return load.loadOp; })));
  Operation *lastLoadAndStore = findNearestCommonPostDominator(loadAndStoreOps, postDomInfo);
  for (unsigned pair = 0; pair < numBarrierPairs; ++pair) {
    ArrayRef<int> pairReady(readyBarIds.data() + pair * numStagesStored,
                            numStagesStored);
    b.setInsertionPoint(lastLoadAndStore);
    Value curReadyBarId = intCst(pairReady.back());
    for (int i = static_cast<int>(pairReady.size()) - 2; i >= 0; --i) {
      Value cond = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, index,
                                           intCst(i));
      curReadyBarId =
          b.create<arith::SelectOp>(cond, intCst(pairReady[i]), curReadyBarId);
    }
    b.setInsertionPointAfter(lastLoadAndStore);
    auto readyArrive = b.createInto<ROCDL::HCUAbarrierArriveOp>(
        loadPartition, stageCluster, curReadyBarId, 1);
    maybeSetAsyncTaskId(readyArrive, pair, numBarrierPairs);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// lowerLoops
//===----------------------------------------------------------------------===//

LogicalResult lowerLoops(scf::ForOp &loop, MutableArrayRef<PipelinedLoad> loads,
                         MutableArrayRef<PipelinedMMA> mmas,
                         WarpSchedule &schedule, int numLoadStages,
                         bool wdraEnabled, int waspNumLoadWarps,
                         int waspNumMmaWarps) {
  Block &body = *loop.getBody();
  DominanceInfo domInfo(loop);
  PostDominanceInfo postDomInfo(loop);

  hcu::WdraTopology topo =
      hcu::deriveWdraTopology(wdraEnabled, waspNumLoadWarps, waspNumMmaWarps);

  // Group loads by common first user operations. This ensures, for example,
  // that multiple loads feeding into the same MMA op are placed together.
  llvm::MapVector<ArrayRef<Operation *>, SmallVector<PipelinedLoad>>
      liveBeforeGroups;
  for (PipelinedLoad &load : loads) {
    if (failed(load.determineLiveRange(body, domInfo, postDomInfo, schedule)))
      return failure();
    liveBeforeGroups[load.liveBeforeOps].push_back(std::move(load));
  }

  SmallVector<PipelinedLoadGroup, 0> loadGroups;
  for (auto &loads : llvm::make_second_range(liveBeforeGroups))
    loadGroups.push_back({std::move(loads)});

  // Multi-buffer and lower the loads.
  auto numLoadGroups = loadGroups.size();
  unsigned numPairs = std::max(1u, topo.numBarrierPairs());
  auto neededAbarrierNum =
      static_cast<int>(2 * numLoadGroups * numLoadStages * numPairs);
  llvm::SmallVector<int> availableAbarrierIds =
      getAvailableAbarrierIds(loop, neededAbarrierNum);
  for (PipelinedLoadGroup &group : loadGroups) {
    group.allocateAref(loop, numLoadStages, availableAbarrierIds, topo,
                       waspNumLoadWarps, waspNumMmaWarps);
    availableAbarrierIds.erase(availableAbarrierIds.begin(),
                               availableAbarrierIds.begin() +
                                   2 * numLoadStages * numPairs);
  }

  for (PipelinedLoadGroup &group : loadGroups) {
    if (failed(group.lowerLoads(schedule, domInfo, postDomInfo)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOADMMASPECIALIZATIONHCU
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct LoadMMASpecialization
    : public triton::gpu::impl::TritonGPULoadMMASpecializationHCUBase<
          LoadMMASpecialization> {
  using TritonGPULoadMMASpecializationHCUBase::TritonGPULoadMMASpecializationHCUBase;

  void runOnOperation() override;
};
} // namespace

void LoadMMASpecialization::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  auto mod = getOperation();
  for (scf::ForOp loop : loops) {
    FailureOr<WarpSchedule> schedule = WarpSchedule::deserialize(loop);
    if (failed(schedule))
      continue;
    auto [loads, mmas] = getPartitionScheme(loop, *schedule);
    if (loads.empty() && mmas.empty())
      continue;
    int loopNumStages = getNumStagesOrDefault(loop, numStages);
    if (failed(lowerLoops(loop, loads, mmas, *schedule, loopNumStages,
                          wdraEnabled, waspNumLoadWarps, waspNumMmaWarps)))
      continue;
  }
}
