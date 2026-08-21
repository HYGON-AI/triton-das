// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "TritonHCU/WaitCntHCUUtility.h"

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::HCU {
namespace {

bool isStore0Waitcnt(Operation *op) {
  auto waitOp = dyn_cast_or_null<triton::amdgpu::MemoryCounterWaitOp>(op);
  if (!waitOp)
    return false;
  auto storeAttr = waitOp.getStoreAttr();
  return storeAttr && storeAttr.getInt() == 0;
}

bool needsStore0WaitAfterWtBufferStore(Operation *op) {
  // Pattern #1:
  //   amdg.buffer_store(cacheModifier = wt)
  //   => if next op is not memory_counter_wait(store=0), insert one.
  auto storeOp = dyn_cast<triton::amdgpu::BufferStoreOp>(op);
  if (!storeOp || storeOp.getCache() != triton::CacheModifier::WT)
    return false;
  return !isStore0Waitcnt(op->getNextNode());
}

} // namespace

// HCU Note: Starting with ZD and newer chips, s_barrier no longer implies
// s_waitcnt vmcnt(0). For patterns that require this ordering guarantee,
// we detect them explicitly and insert s_waitcnt vmcnt(0) in IR.
void addWaitCntHCU(ModuleOp mod) {
  SmallVector<Operation *> anchorsNeedStore0Wait;
  mod.walk([&](Operation *op) {
    if (needsStore0WaitAfterWtBufferStore(op))
      anchorsNeedStore0Wait.push_back(op);
    // Future patterns can be added as additional if-branches here.
  });

  OpBuilder builder(mod.getContext());
  for (Operation *anchorOp : anchorsNeedStore0Wait) {
    builder.setInsertionPointAfter(anchorOp);
    auto storeAttr = builder.getI32IntegerAttr(0);
    triton::amdgpu::MemoryCounterWaitOp::create(
        builder, anchorOp->getLoc(), /*load=*/nullptr, /*store=*/storeAttr,
        /*ds=*/nullptr);
  }
}

} // namespace mlir::triton::HCU
