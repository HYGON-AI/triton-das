// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonHCU/BufferLdsPipeline.h"
#include "TritonHCU/WaspSplitPlan.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace mlir::triton::HCU {
namespace {

// Scale publication needs lgkmcnt(0) as well as CTA rendezvous. gpu.barrier
// and deferred publication barriers only rendezvous; they cannot retire ds2buf.
bool isScalePublicationBarrier(Operation *op) {
  return isa<gpu::LocalBarrierOp>(op) &&
         !op->hasAttr(kDropPublicationLgkmcnt);
}

// A descriptor directly references the complete allocation of its defining
// scale_write. Scale Buffer intervals are in rows, including alignment gaps;
// this analysis never mixes them with LDS byte intervals.
std::optional<Interval<size_t>>
getScaleMemoryRange(const MemoryEffects::EffectInstance &effect) {
  if (effect.getResource() != amdgpu::ScaledMemory::get())
    return std::nullopt;
  Value storage = effect.getValue();
  auto write = storage ? storage.getDefiningOp<amdgpu::ScaleWriteOp>()
                       : amdgpu::ScaleWriteOp{};
  if (!write || !write.getRowBaseAttr())
    return std::nullopt;
  auto type = cast<gpu::MemDescType>(storage.getType());
  auto layout = cast<amdgpu::ScaleEncodingAttr>(type.getEncoding());
  auto block = layout.getBlockShape();
  size_t rows = llvm::divideCeil(type.getShape()[0], block[0]) *
                llvm::divideCeil(type.getShape()[1], block[1]) *
                layout.getRowStride();
  size_t base = write.getRowBaseAttr().getInt();
  return Interval<size_t>(base, base + rows);
}

// Reuse native CFG traversal, function summaries and RAW/WAR/WAW checks.
// Existing wait-and-CTA barriers publish Scale Buffer writes. Shared membar runs
// afterwards and reuses Scale barriers for pending LDS dependencies.
class ScaleMembarAnalysis : public MembarOrFenceAnalysis {
public:
  using MembarOrFenceAnalysis::MembarOrFenceAnalysis;

private:
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder) override {
    if (isScalePublicationBarrier(op)) {
      blockInfo->sync();
      return;
    }
    if (isa<ROCDL::SBarrierOp>(op) &&
        op->hasAttr(kConsumerPingpongBarrierAttrName)) {
      // This rendezvous already joins the consuming WASP partitions. Drain
      // ds2buf before it rather than adding another partition-local barrier.
      // A bare rendezvous only retires MMAC reads, never pending scale writes.
      auto wait =
          dyn_cast_or_null<amdgpu::MemoryCounterWaitOp>(op->getPrevNode());
      if (!blockInfo->syncWriteIntervals.empty() &&
          (!wait || !wait.getDsAttr() || wait.getDsAttr().getInt() != 0)) {
        builder->setInsertionPoint(op);
        amdgpu::MemoryCounterWaitOp::create(
            *builder, op->getLoc(), /*load=*/nullptr, /*store=*/nullptr,
            /*ds=*/builder->getI32IntegerAttr(0));
      }
      blockInfo->sync();
      return;
    }
    if (isa<mlir::gpu::BarrierOp, gpu::LocalBarrierOp>(op)) {
      // MMAC consumes scale as an instruction input. Joining its consuming
      // waves retires reads, but does not drain asynchronous ds2buf writes.
      blockInfo->syncReadIntervals.clear();
      return;
    }

    BlockInfo current;
    if (auto call = dyn_cast<triton::CallOp>(op)) {
      if (auto callee = dyn_cast<FunctionOpInterface>(call.resolveCallable()))
        current = funcBlockInfoMap->lookup(callee);
    } else if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const auto &effect : instances) {
        auto range = getScaleMemoryRange(effect);
        if (!range)
          continue;
        if (isa<MemoryEffects::Read>(effect.getEffect()))
          current.syncReadIntervals[*range].insert(op);
        else if (isa<MemoryEffects::Write>(effect.getEffect()))
          current.syncWriteIntervals[*range].insert(op);
      }
    }

    if (blockInfo->isIntersected(current, filter)) {
      builder->setInsertionPoint(op);
      gpu::LocalBarrierOp::create(*builder, op->getLoc());
      blockInfo->sync();
    }
    blockInfo->join(current);
  }
};

} // namespace

void addLocalBarrierForScaleMemory(ModuleAllocation *allocation) {
  // gfx946 keeps ttg.async_wait through asyncmark lowering. Expose its
  // rendezvous before Scale analysis, rather than waiting for Shared membar.
  allocation->getModuleOp().walk([&](gpu::AsyncWaitOp wait) {
    if (isa_and_nonnull<gpu::LocalBarrierOp, mlir::gpu::BarrierOp>(
            wait->getNextNode()))
      return;
    OpBuilder builder(wait);
    builder.setInsertionPointAfter(wait);
    gpu::LocalBarrierOp::create(builder, wait.getLoc());
  });
  ModuleMembarOrFenceAnalysis<ScaleMembarAnalysis> analysis(allocation);
  analysis.run();
}

} // namespace mlir::triton::HCU
