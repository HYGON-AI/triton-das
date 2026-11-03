// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/BufferLdsEncoding.h"
#include "TritonHCU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace ttg = mlir::triton::gpu;
namespace mlir {
#define GEN_PASS_DEF_TRITONHCUCONVERTBUFFERLDSCOPIES

#include "TritonHCU/Passes.h.inc"

namespace {
// A/B share the same producer rewrite; only their physical layout proof
// differs.
static bool isBufferLdsTensorAndMemDesc(RankedTensorType srcTy,
                                        ttg::MemDescType dstTy) {
  if (!srcTy || !dstTy || srcTy.getShape() != dstTy.getShape())
    return false;
  return isa<ttg::HCUBufferLdsSharedEncodingAttr>(dstTy.getEncoding()) &&
         srcTy.getElementType() == dstTy.getElementType();
}

// The shared encoding includes wrap; generic stores cannot implement it.
// Recognize the destination independently of the producer type/layout: a
// failed producer match must not make the physical mismatch invisible.
static bool requiresWrappedProducer(ttg::MemDescType type) {
  return isa<ttg::HCUBufferLdsSharedEncodingAttr>(type.getEncoding());
}

// Pipelined loop copies are rewritten in place, at the original load
// position: the copy issues right after the reads that free its slot, so
// its DMA completes under the MMA while the synchronize pass publishes each
// generation at its consume point (wait for the consumed generation + block
// barrier before each read) instead of draining every copy. In-place
// emission puts the copy before the same-iteration reads of the slot it
// overwrites, so it additionally proves every such read can move before the
// copy, and inserts a local barrier between the reads and the copy: in
// every warp the reads of the written slot precede the barrier and the copy
// follows it, so all warps' reads complete (at the barrier) before any warp
// issues its copy. That closes the cross-warp read/write race formally
// instead of relying on the HBM latency margin. Retimed emission (copy at
// the store, after the dot) is the fallback when the read-before-copy proof
// fails or the loop is WASP-specialized. The
// initialized-allocation fallback below still retimes its producer; future
// work should make direct placement a shared primitive for other async-copy
// users. Retiming a global read is legal only while its source is unchanged.
// This deliberately conservative check admits shared-memory effects, but
// rejects global writes, atomics, calls and unknown effects in the enclosing
// prefix (including the entire loop containing the new read).
static bool preservesGlobalSource(Operation *op) {
  // UTC warmup is modeled as a GlobalMemory write solely to keep ordinary
  // payload loads below the cache-priming request. Converting a later payload
  // load and its adjacent local allocation into one buffer-to-LDS producer
  // does not move it across the warmup, so it is safe for this retiming proof.
  if (isa<ttg::LocalStoreOp, ttg::LocalAllocOp, ttg::LocalDeallocOp,
          ttg::LocalBarrierOp, gpu::BarrierOp, ROCDL::SchedBarrier,
          LLVM::AssumeOp, ROCDL::SetPrioOp, triton::amdgpu::CondBarrierOp,
          ROCDL::SBarrierOp, triton::amdgpu::MemoryCounterWaitOp,
          ROCDL::HCUAbarrierTryWaitOp, ROCDL::HCUAbarrierArriveOp,
          ROCDL::HCUAbarrierSeqOp, ROCDL::HCUAbarrierInitOp,
          ROCDL::HCUAbarrierInvOp, triton::amdgpu::BufferLoadToLocalOp,
          triton::amdgpu::UTCWarmupOp>(op))
    return true;
  if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin() == "llvm.amdgcn.readfirstlane";
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!preservesGlobalSource(&nested))
            return false;
    return true;
  }
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    return llvm::all_of(instances, [](auto effect) {
      return isa<MemoryEffects::Read>(effect.getEffect());
    });
  }
  return isMemoryEffectFree(op);
}

// Check the executed prefix at each nesting level. An output store after the
// enclosing loop/if is not crossed by the retimed read. Cyclic regions still
// require the entire region to preserve the source: a write after the new read
// can invalidate a tensor carried from the preceding iteration.
static bool canRetimeRead(Operation *destination) {
  auto function = destination->getParentOfType<triton::FuncOp>();
  if (!function || !llvm::hasSingleElement(function.getBody()))
    return false;
  Operation *anchor = destination;
  while (true) {
    for (Operation &op : *anchor->getBlock()) {
      if (&op == anchor)
        break;
      if (!preservesGlobalSource(&op))
        return false;
    }
    Operation *parent = anchor->getParentOp();
    if (parent == function)
      return true;
    // SCF if executes only the selected branch. For loops and other region
    // operations, retain the conservative whole-region proof.
    if (!isa<scf::IfOp>(parent) && !preservesGlobalSource(parent))
      return false;
    anchor = parent;
  }
}

// Warp specialization puts the producer load and its LDS store in the same
// load-partition block, while unrelated consumer-side global stores live in a
// sibling region.  In that case only the operations crossed inside the
// producer block are relevant to the retiming proof.
static bool canRetimeRead(Operation *source, Operation *destination) {
  if (source->getBlock() == destination->getBlock() &&
      source->isBeforeInBlock(destination)) {
    for (Operation *op = source->getNextNode(); op && op != destination;
         op = op->getNextNode())
      if (!preservesGlobalSource(op))
        return false;
    return true;
  }
  return canRetimeRead(destination);
}

// Follow memdesc views and loop-carried slot arguments back to the physical
// allocation they address.
static Value rootSharedAlloc(Value value) {
  for (;;) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (!loop || argument.getOwner() != loop.getBody())
        break;
      value = loop.getInitArgs()[argument.getArgNumber() - 1];
      continue;
    }
    auto definingOp = value.getDefiningOp();
    if (!definingOp)
      break;
    Value source;
    if (auto view = dyn_cast<ttg::MemDescIndexOp>(definingOp))
      source = view.getSrc();
    else if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(definingOp))
      source = reshape.getSrc();
    else if (auto slice = dyn_cast<ttg::MemDescSubsliceOp>(definingOp))
      source = slice.getSrc();
    else if (auto reinterpret = dyn_cast<ttg::MemDescReinterpretOp>(definingOp))
      source = reinterpret.getSrc();
    if (!source)
      break;
    value = source;
  }
  return value;
}

// Moving a local_load earlier can only expose it to operations that write
// shared memory or have unknown side effects. Global writes do not matter:
// the read stays in SSA order relative to its users and reads the same LDS
// slot. Region ops are proven recursively over their whole content, which
// is conservative for cyclic regions but keeps a nested local_store from
// being treated as harmless just because the region only advertises the
// generic Memory resource.
static bool preservesLdsReadSource(Operation *op, Value readRoot) {
  // A local barrier only orders warps; it never writes shared memory. A read
  // moved across it (from after to before) still precedes the other pattern's
  // copy in every warp, and the barrier only reinforces that protocol, so it
  // is safe to cross. This keeps the A/B pattern applications
  // order-independent when each has already inserted its own barrier.
  if (isa<ttg::LocalBarrierOp>(op))
    return true;
  if (isa<ttg::LocalAllocOp, ttg::LocalStoreOp, ttg::LocalDeallocOp,
          gpu::BarrierOp>(op))
    return false;
  if (op->getNumRegions() > 0) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (!preservesLdsReadSource(&nested, readRoot))
            return false;
    return true;
  }
  if (isMemoryEffectFree(op))
    return true;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffectsOnResource(triton::gpu::SharedMemory::get(), instances);
  if (!llvm::all_of(instances, [](auto &instance) {
        return isa<MemoryEffects::Read>(instance.getEffect());
      }))
  {
    if (auto copy = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op);
        copy && !instances.empty() &&
        rootSharedAlloc(copy.getDest()) != rootSharedAlloc(readRoot))
      return true;
    return false;
  }
  return true;
}

static bool canMoveSharedReadBefore(ttg::LocalLoadOp read, Operation *anchor) {
  if (read->getBlock() != anchor->getBlock() || !anchor->isBeforeInBlock(read))
    return false;
  // Every operand must dominate the new position, or the move is illegal.
  for (Value operand : read->getOperands()) {
    if (auto definingOp = operand.getDefiningOp()) {
      if (definingOp->getBlock() != read->getBlock() ||
          !definingOp->isBeforeInBlock(anchor))
        return false;
    }
  }
  for (Operation *op = anchor->getNextNode(); op && op != read;
       op = op->getNextNode())
    if (!preservesLdsReadSource(op, read.getSrc()))
      return false;
  return true;
}

// The pipelined ring writes the next generation into the very slot the
// current iteration reads: with S slots, the generation-N+1 copy overwrites
// the slot holding the data generation N is about to consume, and program
// order is the only protection. Emission at the original load puts the copy
// before those reads, so every shared read of the written allocation that
// follows the copy point must be proven movable before it. This covers every
// ring depth, not just a single physical slot. Returns nullopt when the
// proof fails; the caller must fall back (retiming or the original error).
static std::optional<SmallVector<ttg::LocalLoadOp, 2>>
inPlaceCopyReads(Operation *copyPoint, scf::ForOp loop, Value writtenSlot) {
  Value writtenRoot = rootSharedAlloc(writtenSlot);
  SmallVector<ttg::LocalLoadOp, 2> reads;
  for (Operation *op = copyPoint->getNextNode(); op;
       op = op->getNextNode()) {
    // A nested region can hide reads of the written slot that this scan
    // cannot reach or move; give up on the in-place fallback rather than
    // prove incomplete.
    if (op->getNumRegions() > 0)
      return std::nullopt;
    auto read = dyn_cast<ttg::LocalLoadOp>(op);
    if (!read || rootSharedAlloc(read.getSrc()) != writtenRoot)
      continue;
    if (!canMoveSharedReadBefore(read, copyPoint))
      return std::nullopt;
    reads.push_back(read);
  }
  llvm::sort(reads, [](ttg::LocalLoadOp lhs, ttg::LocalLoadOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return reads;
}

static void moveReadsBeforeCopy(ArrayRef<ttg::LocalLoadOp> reads,
                                Operation *copyPoint) {
  for (auto read : llvm::reverse(reads))
    read->moveBefore(copyPoint);
}

static bool movePureOperationBefore(Operation *operation, Operation *point) {
  if (operation->getBlock() != point->getBlock())
    return true;
  if (!isMemoryEffectFree(operation) || !operation->getRegions().empty())
    return false;
  for (Value operand : operation->getOperands())
    if (auto definingOp = operand.getDefiningOp();
        definingOp && definingOp->getBlock() == point->getBlock() &&
        !movePureOperationBefore(definingOp, point))
      return false;
  if (point->isBeforeInBlock(operation))
    operation->moveBefore(point);
  return true;
}

static Value materializeNextSlotBefore(PatternRewriter &rewriter,
                                       Operation *point, Value nextSlot) {
  auto view = nextSlot.getDefiningOp<ttg::MemDescIndexOp>();
  if (!view)
    return {};
  if (auto index = view.getIndex().getDefiningOp();
      index && !movePureOperationBefore(index, point))
    return {};
  rewriter.setInsertionPoint(point);
  return rewriter.create<ttg::MemDescIndexOp>(
      view.getLoc(), view.getType(), view.getSrc(), view.getIndex());
}

// A carried tensor and carried offsets must describe the same load on BOTH
// entry and backedge, including zero-trip nested loops. A future load alone
// is not a proof of the entry value. Backedges are currently unmasked; retain
// the initial mask until at least one iteration of its defining loop ran.
struct ProducerProof {
  Value initialMask;
  SmallVector<scf::ForOp> completedLoops;
};

static LogicalResult proveProducer(Value tensor, Value offsets,
                                   triton::amdgpu::BufferLoadOp model,
                                   ProducerProof &proof) {
  if (auto load = tensor.getDefiningOp<triton::amdgpu::BufferLoadOp>()) {
    if (load.getPtr() != model.getPtr() ||
        load.getStride() != model.getStride() ||
        load.getCache() != model.getCache() || load.getOther() ||
        load.getOffsets() != offsets)
      return failure();
    proof.initialMask = load.getMask();
    return success();
  }
  auto tensorResult = dyn_cast<OpResult>(tensor);
  auto offsetResult = dyn_cast<OpResult>(offsets);
  if (!tensorResult || !offsetResult ||
      tensorResult.getOwner() != offsetResult.getOwner())
    return failure();
  auto loop = dyn_cast<scf::ForOp>(tensorResult.getOwner());
  if (!loop || !loop.isDefinedOutsideOfLoop(model.getPtr()) ||
      (model.getStride() && !loop.isDefinedOutsideOfLoop(model.getStride())))
    return failure();
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  unsigned ti = tensorResult.getResultNumber();
  unsigned oi = offsetResult.getResultNumber();
  ProducerProof backedge;
  if (failed(proveProducer(yield.getOperand(ti), yield.getOperand(oi), model,
                           backedge)) ||
      backedge.initialMask || !backedge.completedLoops.empty() ||
      failed(proveProducer(loop.getInitArgs()[ti], loop.getInitArgs()[oi],
                           model, proof)))
    return failure();
  proof.completedLoops.push_back(loop);
  return success();
}

static Value materializeMask(PatternRewriter &rewriter, Location loc,
                             const ProducerProof &proof,
                             scf::ForOp currentLoop = {}) {
  if (!proof.initialMask)
    return {};
  auto type = cast<RankedTensorType>(proof.initialMask.getType());
  Value all = arith::ConstantOp::create(rewriter, loc, type,
                                        DenseElementsAttr::get(type, true));
  Value mask = proof.initialMask;
  for (scf::ForOp loop : proof.completedLoops) {
    Value ran =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                              loop.getLowerBound(), loop.getUpperBound());
    mask = arith::SelectOp::create(rewriter, loc, ran, all, mask);
  }
  if (currentLoop) {
    Value first = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        currentLoop.getInductionVar(),
                                        currentLoop.getLowerBound());
    mask = arith::SelectOp::create(rewriter, loc, first, mask, all);
  }
  return mask;
}

// Carry pointer-offset tensors in the producer layout through SCF. Casts at
// the old users preserve their original contract; ordinary layout propagation
// folds those casts into address arithmetic. This avoids using LDS scratch to
// transpose a full offset tensor in the remainder/drain while four slots are
// resident. No memory operation or control-flow edge is moved.
static void transportOffsetEncoding(Value value, Attribute encoding,
                                    llvm::DenseSet<Value> &visited) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.getElementType().isInteger(32) ||
      type.getEncoding() == encoding || !visited.insert(value).second)
    return;
  scf::ForOp loop;
  unsigned index = 0;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument.getOwner() != loop.getBody() ||
        argument.getArgNumber() == 0)
      return;
    index = argument.getArgNumber() - 1;
  } else if (auto result = dyn_cast<OpResult>(value)) {
    loop = dyn_cast<scf::ForOp>(result.getOwner());
    if (loop)
      index = result.getResultNumber();
  }
  if (!loop) {
    if (Operation *def = value.getDefiningOp())
      for (Value operand : def->getOperands())
        if (auto operandType = dyn_cast<RankedTensorType>(operand.getType());
            operandType && operandType.getShape() == type.getShape())
          transportOffsetEncoding(operand, encoding, visited);
    return;
  }

  Value initial = loop.getInitArgs()[index];
  transportOffsetEncoding(initial, encoding, visited);
  auto newType = type.cloneWithEncoding(encoding);
  OpBuilder builder(loop);
  Value newInitial = initial;
  if (initial.getType() != newType)
    newInitial =
        ttg::ConvertLayoutOp::create(builder, loop.getLoc(), newType, initial);
  loop->setOperand(3 + index, newInitial);
  BlockArgument argument = loop.getRegionIterArgs()[index];
  auto result = loop.getResult(index);
  argument.setType(newType);
  result.setType(newType);
  builder.setInsertionPointToStart(loop.getBody());
  auto entryCast =
      ttg::ConvertLayoutOp::create(builder, loop.getLoc(), type, argument);
  argument.replaceAllUsesExcept(entryCast.getResult(), entryCast);
  builder.setInsertionPointAfter(loop);
  auto exitCast =
      ttg::ConvertLayoutOp::create(builder, loop.getLoc(), type, result);
  result.replaceAllUsesExcept(exitCast.getResult(), exitCast);
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  builder.setInsertionPoint(yield);
  Value next = yield.getOperand(index);
  auto nextCast =
      ttg::ConvertLayoutOp::create(builder, loop.getLoc(), newType, next);
  yield->setOperand(index, nextCast);
}

struct CombineBufferLdsBufferLoadToLocal
    : public mlir::OpRewritePattern<ttg::LocalStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttg::LocalStoreOp store,
                  PatternRewriter &rewriter) const override {
    auto load = store.getSrc().getDefiningOp<triton::amdgpu::BufferLoadOp>();
    if (load && load->getBlock() == store->getBlock() &&
        load->isBeforeInBlock(store) && !load.getOther() &&
        load.getResult().hasOneUse() &&
        isBufferLdsTensorAndMemDesc(cast<RankedTensorType>(load.getType()),
                                    store.getDst().getType())) {
      auto loop = dyn_cast<scf::ForOp>(store->getParentOp());
      // In-place is the primary schedule for plain pipelined loops: the
      // early issue lets the DMA fly under the MMA while the synchronize
      // pass publishes per generation. A global write between the load and
      // the store no longer forces a different schedule: the reads simply
      // move ahead of the copy. Retiming stays as the fallback when the
      // read-before-copy proof fails (or the loop is WASP-specialized).
      bool inWasp = load->getParentOfType<ttg::WarpSpecializeOp>() != nullptr;
      if (loop && !inWasp) {
        auto reads = inPlaceCopyReads(load, loop, store.getDst());
        if (reads) {
          Value slot =
              materializeNextSlotBefore(rewriter, load, store.getDst());
          if (slot) {
            auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
            unsigned slotIndex = 0;
            bool carried = false;
            for (auto [index, operand] : llvm::enumerate(yield.getOperands()))
              if (operand == store.getDst()) {
                slotIndex = index;
                carried = true;
              }
            if (carried) {
              yield->setOperand(slotIndex, slot);
              auto tokenType = rewriter.getType<ttg::AsyncTokenType>();
              rewriter.setInsertionPoint(load);
              auto barrier = ttg::LocalBarrierOp::create(rewriter, load.getLoc());
              auto copy = triton::amdgpu::BufferLoadToLocalOp::create(
                  rewriter, load.getLoc(), tokenType, slot,
                  load.getPtr(), load.getOffsets(), load.getMask(), Value(),
                  load.getStride(), load.getCache(),
                  /*contiguity=*/8);
              moveReadsBeforeCopy(*reads, barrier);
              // The barrier above closes the cross-warp race on the written
              // slot (reads complete in every warp before any copy issues);
              // the synchronize pass then publishes the copy (wait + barrier)
              // right after it, matching the retimed schedule's contract that
              // the next generation is only visible after the fence.
              rewriter.eraseOp(store);
              rewriter.eraseOp(load);
              return success();
            }
          }
        }
      }
    }

    if (!load || !canRetimeRead(load, store) || load.getOther() ||
        !load.getResult().hasOneUse() ||
        !isBufferLdsTensorAndMemDesc(cast<RankedTensorType>(load.getType()),
                                     store.getDst().getType()))
      return failure();

    // The in-place attempt may have moved the insertion point (a slot view
    // materialized at the load); pin the retimed copy to the store.
    rewriter.setInsertionPoint(store);
    // The in-place schedule inserts a barrier before the copy so that no
    // warp overwrites a slot until every warp has completed all earlier
    // reads of it. The retimed copy at the store has no such barrier, and
    // the per-generation publication commits it without a post-copy drain,
    // so restore the same protocol here. BPP-applied loops own their own
    // publication schedule; WASP copies keep the legacy post-copy barrier.
    if (auto copyLoop = dyn_cast<scf::ForOp>(store->getParentOp()))
      if (!store->getParentOfType<ttg::WarpSpecializeOp>() &&
          !copyLoop->hasAttr("hcu.block_pingpong.applied"))
        ttg::LocalBarrierOp::create(rewriter, store.getLoc());
    auto tokenType = rewriter.getType<ttg::AsyncTokenType>();
    triton::amdgpu::BufferLoadToLocalOp::create(
        rewriter, store.getLoc(), tokenType, store.getDst(), load.getPtr(),
        load.getOffsets(), load.getMask(), load.getOther(), load.getStride(),
        load.getCache(), /*contiguity=*/8);
    rewriter.eraseOp(store);
    if (load->use_empty())
      rewriter.eraseOp(load);
    return success();
  }
};

struct CombineBufferLdsLoopBufferLoadToLocal
    : public OpRewritePattern<ttg::LocalStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalStoreOp store,
                                PatternRewriter &rewriter) const override {
    if (auto sourceResult = dyn_cast<OpResult>(store.getSrc())) {
      if (auto loop = dyn_cast<scf::ForOp>(sourceResult.getOwner())) {
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        if (auto load = yield.getOperand(sourceResult.getResultNumber())
                            .getDefiningOp<triton::amdgpu::BufferLoadOp>();
            load && load->getBlock() == store->getBlock() &&
            load->isBeforeInBlock(store)) {
          auto srcTy = cast<RankedTensorType>(load.getType());
          if (!load.getOther() && load.getResult().hasOneUse() &&
              isBufferLdsTensorAndMemDesc(srcTy, store.getDst().getType())) {
            // Same contract as the same-block pattern above: in-place is
            // primary, gated only by the proof that every read of the
            // written allocation can precede the copy.
            if (!load->getParentOfType<ttg::WarpSpecializeOp>()) {
              auto reads = inPlaceCopyReads(load, loop, store.getDst());
              if (reads) {
                Value slot =
                    materializeNextSlotBefore(rewriter, load, store.getDst());
                if (slot) {
                  unsigned slotIndex = 0;
                  bool carried = false;
                  for (auto [index, operand] :
                       llvm::enumerate(yield.getOperands()))
                    if (operand == store.getDst()) {
                      slotIndex = index;
                      carried = true;
                    }
                  if (carried) {
                    yield->setOperand(slotIndex, slot);
                    auto tokenType = rewriter.getType<ttg::AsyncTokenType>();
                    rewriter.setInsertionPoint(load);
                    auto barrier =
                        ttg::LocalBarrierOp::create(rewriter, load.getLoc());
                    auto copy = triton::amdgpu::BufferLoadToLocalOp::create(
                        rewriter, load.getLoc(), tokenType, slot,
                        load.getPtr(), load.getOffsets(), load.getMask(),
                        Value(), load.getStride(), load.getCache(),
                        /*contiguity=*/8);
                    moveReadsBeforeCopy(*reads, barrier);
                    // Same race closure and publication contract as the
                    // same-block pattern above.
                    rewriter.eraseOp(store);
                    rewriter.eraseOp(load);
                    return success();
                  }
                }
              }
            }
          }
        }
      }
    }

    auto srcTy = dyn_cast<RankedTensorType>(store.getSrc().getType());
    if (!isBufferLdsTensorAndMemDesc(srcTy, store.getDst().getType()) ||
        !canRetimeRead(store))
      return failure();

    scf::ForOp loop;
    unsigned tensorIndex;
    bool carried = false;
    if (auto argument = dyn_cast<BlockArgument>(store.getSrc())) {
      loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (!loop || argument.getOwner() != loop.getBody() ||
          argument.getArgNumber() == 0)
        return failure();
      tensorIndex = argument.getArgNumber() - 1;
      carried = true;
    } else if (auto result = dyn_cast<OpResult>(store.getSrc())) {
      loop = dyn_cast<scf::ForOp>(result.getOwner());
      if (!loop)
        return failure();
      tensorIndex = result.getResultNumber();
    } else {
      return failure();
    }

    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    auto load = yield.getOperand(tensorIndex)
                    .getDefiningOp<triton::amdgpu::BufferLoadOp>();
    if (!load || load.getOther() || load.getMask() || load.getType() != srcTy ||
        !loop.isDefinedOutsideOfLoop(load.getPtr()) ||
        (load.getStride() && !loop.isDefinedOutsideOfLoop(load.getStride())))
      return failure();

    std::optional<unsigned> offsetIndex;
    for (auto [i, value] : llvm::enumerate(yield.getOperands()))
      if (value == load.getOffsets()) {
        offsetIndex = i;
        break;
      }
    if (!offsetIndex)
      return failure();

    // The in-place attempt may have moved the insertion point (a slot view
    // materialized at the load); pin the retimed copy to the store.
    rewriter.setInsertionPoint(store);
    Value offsets = carried ? Value(loop.getRegionIterArgs()[*offsetIndex])
                            : loop.getResult(*offsetIndex);
    ProducerProof proof;
    if (carried) {
      if (failed(proveProducer(loop.getInitArgs()[tensorIndex],
                               loop.getInitArgs()[*offsetIndex], load, proof)))
        return failure();
    } else if (failed(proveProducer(store.getSrc(), offsets, load, proof))) {
      return failure();
    }
    Value mask = materializeMask(rewriter, store.getLoc(), proof,
                                 carried ? loop : scf::ForOp());
    // Same pre-copy barrier protocol as the same-block pattern above: the
    // per-generation publication commits this copy without a post-copy
    // drain, so the slot overwrite must wait for every warp's earlier
    // reads. Created only after the producer proof succeeded so that a
    // failed pattern leaves no barrier behind. BPP-applied loops own their
    // own publication schedule; WASP copies keep the legacy post-copy
    // barrier.
    if (auto copyLoop = dyn_cast<scf::ForOp>(store->getParentOp()))
      if (!store->getParentOfType<ttg::WarpSpecializeOp>() &&
          !copyLoop->hasAttr("hcu.block_pingpong.applied"))
        ttg::LocalBarrierOp::create(rewriter, store.getLoc());
    triton::amdgpu::BufferLoadToLocalOp::create(
        rewriter, store.getLoc(), rewriter.getType<ttg::AsyncTokenType>(),
        store.getDst(), load.getPtr(), offsets, mask, Value(), load.getStride(),
        load.getCache(), /*contiguity=*/8);
    rewriter.eraseOp(store);
    return success();
  }
};

struct CombineBufferLdsBufferLoadAllocToLocal
    : public mlir::OpRewritePattern<ttg::LocalAllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttg::LocalAllocOp alloc,
                  PatternRewriter &rewriter) const override {
    auto src = alloc.getSrc();
    if (!src)
      return failure();
    auto load = src.getDefiningOp<triton::amdgpu::BufferLoadOp>();
    if (!load)
      return failure();

    if (!canRetimeRead(load, alloc) || load.getOther() ||
        !load.getResult().hasOneUse() ||
        !isBufferLdsTensorAndMemDesc(cast<RankedTensorType>(load.getType()),
                                     alloc.getType()))
      return failure();

    auto emptyAlloc = ttg::LocalAllocOp::create(rewriter, alloc.getLoc(),
                                                alloc.getType(), Value());
    auto tokenType = rewriter.getType<ttg::AsyncTokenType>();
    triton::amdgpu::BufferLoadToLocalOp::create(
        rewriter, alloc.getLoc(), tokenType, emptyAlloc, load.getPtr(),
        load.getOffsets(), load.getMask(), load.getOther(), load.getStride(),
        load.getCache(), /*contiguity=*/8);
    rewriter.replaceOp(alloc, emptyAlloc.getResult());
    rewriter.eraseOp(load);
    return success();
  }
};

} // namespace

struct TritonHCUConvertBufferLdsCopiesPass
    : impl::TritonHCUConvertBufferLdsCopiesBase<
          TritonHCUConvertBufferLdsCopiesPass> {
  using Base::Base;
  void runOnOperation() override {
    auto target = getOperation()->getAttrOfType<StringAttr>("ttg.target");
    if (!target || !target.getValue().starts_with("hip:"))
      return;
    StringRef arch = target.getValue().drop_front(4);
    if (!triton::HCU::getBufferLdsTargetProfile(
            std::string_view(arch.data(), arch.size())))
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<CombineBufferLdsLoopBufferLoadToLocal,
                 CombineBufferLdsBufferLoadAllocToLocal,
                 CombineBufferLdsBufferLoadToLocal>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    SmallVector<triton::amdgpu::BufferLoadToLocalOp> copies;
    getOperation().walk([&](triton::amdgpu::BufferLoadToLocalOp copy) {
      if (requiresWrappedProducer(copy.getDest().getType()))
        copies.push_back(copy);
    });
    for (auto copy : copies) {
      auto sharedEncoding = cast<ttg::HCUBufferLdsSharedEncodingAttr>(
          copy.getDest().getType().getEncoding());
      auto profile = triton::HCU::getBufferLdsTargetProfile(
          std::string_view(arch.data(), arch.size()));
      if (!profile ||
          !triton::HCU::isBufferLdsConfigEncodable(
              triton::HCU::getBufferLdsConfig(sharedEncoding), *profile)) {
        copy.emitError("buffer-to-LDS layout is not encodable on this target");
        signalPassFailure();
        return;
      }
      auto encoding = triton::HCU::getBufferLdsProducerEncoding(sharedEncoding);
      llvm::DenseSet<Value> visited;
      transportOffsetEncoding(copy.getOffsets(), encoding, visited);
      OpBuilder builder(copy);
      for (OpOperand &operand : copy->getOpOperands()) {
        if (auto type = dyn_cast<RankedTensorType>(operand.get().getType())) {
          auto convertedType = type.cloneWithEncoding(encoding);
          Value converted = ttg::ConvertLayoutOp::create(
              builder, copy.getLoc(), convertedType, operand.get());
          operand.set(converted);
        }
      }
      // Vector width must be established from the actual converted pointers
      // and mask, not forced merely because the tile has the expected shape.
      copy.setContiguity(0);
    }
    // Failing closed is intentional for the opt-in path. Do not leave an
    // unwrapped generic writer feeding a consumer that interprets M0 wrap.
    auto result = getOperation().walk([&](Operation *op) -> WalkResult {
      bool invalid = false;
      if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
        invalid = requiresWrappedProducer(store.getDst().getType());
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
        invalid = alloc.getSrc() && requiresWrappedProducer(alloc.getType());
      if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
        invalid = requiresWrappedProducer(copy.getResult().getType());
      if (!invalid)
        return WalkResult::advance();
      op->emitError()
          << "buffer-to-LDS wrap requires a buffer_load_to_local producer ("
          << op->getName() << " writes a wrapped tile); "
          << "generic LDS writes are not a safe fallback (disable "
          << "use_async_copy for this kernel)";
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace mlir
