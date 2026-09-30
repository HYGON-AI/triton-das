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

// TODO(hcu): Split an initialized local_alloc into an empty allocation before
// the original global load and emit buffer_load_to_local at that load's exact
// program point. That avoids retiming the read and should become a shared
// primitive for BufferLds and generic async copy. The generic path can then
// diagnose or fall back for unsupported masks/other values without leaving an
// unlowered buffer_load_to_local operation. Until that refactor is complete,
// BufferLds selection rejects sources that would cross a global write.
// Retiming a global read is legal only while its source is unchanged. This
// deliberately conservative check admits shared-memory effects, but rejects
// global writes, atomics, calls and unknown effects in the enclosing prefix
// (including the entire loop containing the new read).
static bool preservesGlobalSource(Operation *op) {
  // UTC warmup is modeled as a GlobalMemory write solely to keep ordinary
  // payload loads below the cache-priming request. Converting a later payload
  // load and its adjacent local allocation into one buffer-to-LDS producer
  // does not move it across the warmup, so it is safe for this retiming proof.
  if (isa<ttg::LocalStoreOp, ttg::LocalAllocOp, ttg::LocalDeallocOp,
          ttg::LocalBarrierOp, gpu::BarrierOp, ROCDL::SchedBarrier,
          LLVM::AssumeOp, ROCDL::SetPrioOp, triton::amdgpu::CondBarrierOp,
          ROCDL::SBarrierOp, triton::amdgpu::MemoryCounterWaitOp,
          triton::amdgpu::BufferLoadToLocalOp, triton::amdgpu::UTCWarmupOp>(op))
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

static bool canRetimeRead(Operation *destination) {
  auto function = destination->getParentOfType<triton::FuncOp>();
  if (!function || !llvm::hasSingleElement(function.getBody()))
    return false;
  Operation *anchor = destination;
  while (anchor->getParentOp() != function)
    anchor = anchor->getParentOp();
  for (Operation &op : function.getBody().front()) {
    if (&op == anchor)
      return anchor == destination || preservesGlobalSource(anchor);
    if (!preservesGlobalSource(&op))
      return false;
  }
  return false;
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
    if (!load || !canRetimeRead(store) || load.getOther() ||
        !load.getResult().hasOneUse() ||
        !isBufferLdsTensorAndMemDesc(cast<RankedTensorType>(load.getType()),
                                     store.getDst().getType()))
      return failure();

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

    if (!canRetimeRead(alloc) || load.getOther() ||
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
      op->emitError(
          "buffer-to-LDS wrap requires a buffer_load_to_local producer; "
          "generic LDS writes are not a safe fallback (disable "
          "use_async_copy for this kernel)");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    for (auto copy : copies) {
      OpBuilder builder(copy);
      builder.setInsertionPointAfter(copy);
      auto commit = ttg::AsyncCommitGroupOp::create(
          builder, copy.getLoc(), ValueRange{copy.getResult()});
      ttg::AsyncWaitOp::create(builder, copy.getLoc(),
                               ValueRange{commit.getResult()}, 0);
      ttg::LocalBarrierOp::create(builder, copy.getLoc());
    }
  }
};
} // namespace mlir
