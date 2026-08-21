// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include <climits>
#include <cstdlib>
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "TritonHCU/Passes.h"
#include "TritonHCU/WdraSplitPlan.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_HCUCONVERTWARPSPECIALIZETOLLVM
#include "TritonHCU/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// convertOpTypes
//===----------------------------------------------------------------------===//

static void convertOpTypes(Operation *op, const TypeConverter &typeConverter) {
  ImplicitLocOpBuilder b(op->getLoc(), op);
  SmallVector<Value> operands = llvm::to_vector(op->getOperands());
  for (Value &operand : operands) {
    Type type = typeConverter.convertType(operand.getType());
    if (type != operand.getType()) {
      operand =
          b.create<UnrealizedConversionCastOp>(type, operand).getResult(0);
    }
  }
  op->setOperands(operands);

  for (Region &region : op->getRegions()) {
    b.setInsertionPointToStart(&region.front());
    for (BlockArgument arg : llvm::to_vector(region.getArguments())) {
      Type type = typeConverter.convertType(arg.getType());
      BlockArgument newArg = region.addArgument(type, arg.getLoc());
      auto cast = b.create<UnrealizedConversionCastOp>(arg.getType(), newArg);
      arg.replaceAllUsesWith(cast.getResult(0));
      region.eraseArgument(0);
    }
  }

  SmallVector<Type> resultTypes;
  (void)typeConverter.convertTypes(op->getResultTypes(), resultTypes);
  if (TypeRange(resultTypes) == op->getResultTypes())
    return;
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       resultTypes, op->getAttrs());
  for (Region &region : op->getRegions())
    state.addRegion()->takeBody(region);
  b.setInsertionPoint(op);
  Operation *newOp = b.create(state);

  SmallVector<Value> results;
  for (auto [i, result, type] :
       llvm::enumerate(newOp->getResults(), op->getResultTypes())) {
    auto cast = b.create<UnrealizedConversionCastOp>(type, result);
    op->getResult(i).replaceAllUsesWith(cast.getResult(0));
  }
  op->erase();
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Reserve one barrier for the default warp group, one for the start barrier,
// and one for the end barrier.
enum BarrierIndex {
  kSwitchLoopBarrierIdx,
  // Shared by both MMA consumers (tritonhcu-consumer-pingpong). Replaces the
  // unused partition-start slot so reserved IDs stay 0..5.
  kConsumerPingpongBarrierIdx,
  kAbarrierInvBarrierIdx,
  kPartitionEndBarrierIdx,
  kDefaultWarpGroupBarrierIdx,
  kReturnBarrierIdx,
  kNumReservedBarriers,
  kNumBarriers = 16
};

static void createBarrier(TritonLLVMIRRewriter &b, unsigned barIdx,
                          std::optional<unsigned> numWarps) {
  assert(barIdx < 16 && "not enough barriers");
  if (numWarps.has_value()) {
    b.create<ROCDL::HCUEbarrierSyncOp>(barIdx, numWarps.value());
  } else {
    b.create<ROCDL::HCUEbarrierSyncOp>(barIdx, 0);
  }
}

static bool isWarpGroupBarrierOp(Operation *op) {
  return isa<ROCDL::BarrierOp, ROCDL::SBarrierOp>(op);
}

static void rewriteBarrierOp(Operation *op, unsigned barIdx,
                             std::optional<unsigned> numWarps) {
  TritonLLVMIRRewriter b(op->getLoc(), op);
  createBarrier(b, barIdx, numWarps);
  op->erase();
}

static void lowerHCUAbarrierArriveOp(ROCDL::HCUAbarrierArriveOp bar, unsigned barIdx, unsigned warpStartId, unsigned numWarps) {
  TritonLLVMIRRewriter b(bar.getLoc(), bar);
  Value warpId = b.create<ROCDL::HCUGetWaveIdOp>();
  Value zero = b.i32_val(warpStartId);
  Value isWarpZero = b.icmp_eq(warpId, zero);
  Block *currentBlock = bar->getBlock();
  Block *afterBlock = currentBlock->splitBlock(bar);
  Block *barrierBlock = b.createBlock(afterBlock);
  b.setInsertionPointToEnd(currentBlock);
  createBarrier(b, barIdx, numWarps);
  b.create<LLVM::CondBrOp>(isWarpZero, barrierBlock, afterBlock);
  bar->moveBefore(barrierBlock, barrierBlock->begin());
  b.setInsertionPointToEnd(barrierBlock);
  b.create<LLVM::BrOp>(afterBlock);
}

//===----------------------------------------------------------------------===//
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

static LogicalResult findTrivialSubcomputation(LLVM::LLVMFuncOp func,
                                               Value capture,
                                               SetVector<Operation *> &ops) {
  SetVector<Value> worklist;
  worklist.insert(capture);
  for (unsigned i = 0; i != worklist.size(); ++i) {
    Value capture = worklist[i];
    // Check for a kernel argument.
    if (auto arg = dyn_cast<BlockArgument>(capture)) {
      if (arg.getOwner() == &func.getBody().front())
        continue;
      // Otherwise, this is some other block argument that cannot be elided.
      return failure();
    }

    Operation *op = capture.getDefiningOp();
    // Check if the defining op can be rematerialized. At the LLVM level,
    // checking for pure is probably a good enough heuristic.
    if (isPure(op)) {
      ops.insert(op);
      worklist.insert(op->operand_begin(), op->operand_end());
      continue;
    }
    // The op cannot be rematerialized.
    return failure();
  }

  // Cap the number of ops that can be rematerialized.
  // FIXME: This is arbitrary. Maybe use as a autotune parameter.
  //return success(ops.size() <= 16);
  return success();
}

static void elideTrivialCaptures(LLVM::LLVMFuncOp func,
                                 ArrayRef<WarpSpecializeOp> wsOps) {
  // The goal is to completely eliminate captures by hoisting or rematerializing
  // computations. We could minimize captures by rematerializing
  // subcomputations, but that is much more complicated. Prefer rematerializing
  // because that reduces liveranges. If subgraphs are duplicated more than
  // once, we will rely on CSE to clean them up.
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    llvm::BitVector toErase(wsOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
      subgraph.clear();
      if (failed(findTrivialSubcomputation(func, capture, subgraph)))
        continue;
      toErase.set(i);
      subgraph = topologicalSort(subgraph);

      for (Region *region : wsOp.getPartitionRegions()) {
        OpBuilder b(region);
        IRMapping mapping;
        for (Operation *op : subgraph) {
          b.clone(*op, mapping);
        }
        Value remat = capture;
        if (!subgraph.empty()) {
          unsigned resultIdx = cast<OpResult>(capture).getResultNumber();
          remat = mapping.lookup(subgraph.back())->getResult(resultIdx);
        }
        region->getArgument(i).replaceAllUsesWith(remat);
      }
    }

    wsOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions()) {
      region->front().eraseArguments(toErase);
    }
  }
}

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

// Assign hardware barriers to each warp group and rewrite warp group barriers
// into `ebarrier` instructions. There is a maximum number of barriers.
static LogicalResult rewriteWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                              ArrayRef<WarpSpecializeOp> wsOps,
                                              unsigned defaultNumWarps,
                                              unsigned mmaNumWarps) {
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (isWarpGroupBarrierOp(op)) {
      rewriteBarrierOp(op, kDefaultWarpGroupBarrierIdx, defaultNumWarps);
      return WalkResult::skip();
    }

    return WalkResult::advance();
  });

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
      unsigned warpGroupSize = op.getPartitionNumWarps()[idx];
      partition->walk([&](Operation *bar) {
        if (!isWarpGroupBarrierOp(bar))
          return;
        // Both MMA consumers must meet at the same ebarrier; a per-partition
        // ID would make the pingpong offset a no-op.
        if (bar->hasAttr(triton::HCU::kConsumerPingpongBarrierAttrName)) {
          rewriteBarrierOp(bar, kConsumerPingpongBarrierIdx, mmaNumWarps);
          return;
        }
        rewriteBarrierOp(bar, barIdx, warpGroupSize);
      });
    }
  }

  return success();
}

static void rewritePartitionRegions(WarpSpecializeOp ws, SmallVector<Block *> &sinkBlocks,
                                    const AMD::TargetInfo &targetInfo,
                                    const triton::HCU::WdraTopology &topo) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws.getContext());

  for (auto [idx, partition] : llvm::enumerate(ws.getPartitionRegions())) {
    Block *sinkBlock = sinkBlocks[idx];
    // Load the explicit captures from shared memory and replace the block args
    // if there are any.
    b.setInsertionPointToStart(&partition->front());
    createBarrier(b, kSwitchLoopBarrierIdx, /*numWarps=*/std::nullopt);

    if (partition->getNumArguments()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(partition->getArgumentTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);
      for (auto [i, arg] :
           llvm::zip(llvm::seq<int32_t>(partition->getNumArguments()),
                     partition->getArguments())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        // Each thread in the warp group needs a copy of the value.
        Value value = b.load(arg.getType(), ptr, /*align=*/1);
        arg.replaceAllUsesWith(value);
      }
      partition->front().eraseArguments([](auto) { return true; });
    }

    // The shared memory is only live for the entry into the region, so put
    // another barrier here. ID 1 is reserved for consumer pingpong instead.

    // For MMA partitions, insert an ebarrier with id kAbarrierInvBarrierIdx
    // before the earliest HCUAbarrierInvOp.
    if (topo.isMmaPartition(idx)) {
      ROCDL::HCUAbarrierInvOp earliestInvOp = nullptr;
      partition->walk<mlir::WalkOrder::PreOrder>(
          [&](ROCDL::HCUAbarrierInvOp op) {
            earliestInvOp = op;
            return WalkResult::interrupt();
          });
      if (earliestInvOp) {
        TritonLLVMIRRewriter invBuilder(earliestInvOp.getLoc(), earliestInvOp);
        createBarrier(invBuilder, kAbarrierInvBarrierIdx,
                      /*numWarps=*/std::nullopt);
      }
    }

    // Rewrite all warp returns.
    Block *sink = sinkBlock;
    bool isLoadPartition = topo.isLoadPartition(idx);
    partition->walk([&, sink, isLoadPartition](WarpReturnOp op) {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      if (sink != nullptr) {
        if (isLoadPartition)
          createBarrier(b, kAbarrierInvBarrierIdx,
                        /*numWarps=*/std::nullopt);
        createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
        b.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
      } else {
        if (isLoadPartition)
          createBarrier(b, kAbarrierInvBarrierIdx,
                        /*numWarps=*/std::nullopt);
        createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
        b.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
      }
      });
    }
  }

  // LLVM's LICM will be tempted to hoist code out of the switch loop generated by
  // the `ttg.warp_specialize` lowering. However, neither NVPTX or `ptxas` will
  // rematerialize this code back in to the partition regions, resulting in long
  // liveranges for an arbitrary number of registers.
  //
  // Due to reduced warp group registers, these live values can induce spilling
  // in the partition regions. Prevent this by disabling LICM on the switch loop.
  static void disableLICM(LLVM::BrOp latchBr) {
    Builder b(latchBr.getContext());
    MLIRContext *ctx = b.getContext();
    auto licmMD = LLVM::LoopLICMAttr::get(ctx, b.getBoolAttr(true), {});
    auto loopMD =
        LLVM::LoopAnnotationAttr::get(b.getContext(), {}, {}, {}, {}, {}, licmMD,
                                      {}, {}, {}, {}, {}, {}, {}, {}, {});
    latchBr.setLoopAnnotationAttr(loopMD);
  }

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const AMD::TargetInfo &targetInfo,
                                         int waspNumLoadWarps,
                                         int waspNumMmaWarps,
                                         bool wdraEnabled, 
                                         int wdraNumLoadRegs, 
                                         int wdraNumMmaRegsMain,
                                         int wdraNumMmaRegsTail) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  // wdra requires all warpgroups to synchronize at the end of the kernel.
  func.walk([&](LLVM::ReturnOp op) {
    TritonLLVMIRRewriter b(op.getLoc(), op);
    createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
  });

  // Before lowering away `ttg.warp_specialize`, lower warp group barriers.
  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  auto topo = triton::HCU::getWdraTopologyAttr(module);
  // If module attr missing (should be set by AutomaticWarpSpecialization),
  // derive from pass options so Load-first helpers still work.
  if (topo.kind == triton::HCU::WdraTopoKind::None && wdraEnabled)
    topo = triton::HCU::deriveWdraTopology(wdraEnabled, waspNumLoadWarps,
                                           waspNumMmaWarps);

  // Default-region / preamble barriers are executed only by partition0 warps
  // (Load-first: the Load group). Sizing them with MMA warps deadlocks WASP
  // when load_warps != mma_warps.
  unsigned defaultNumWarps = wsOps.front().getPartitionNumWarps()[0];

  // Clang's branch-averaged VGPR check uses BranchAvailable derived from the
  // per-partition hcu.s.set.vgpr.size sequence (observed as reverse emission
  // order for TwoPTwoC). WdraInit alone is not enough — pad the live budgets
  // that feed SetVgpr as well.
  //
  // Granularity is 4. Equal-weight branches need sum % (numBranches * 4) == 0
  // so the average is an integer multiple of the granularity:
  //   TwoPTwoC (4 branches): multiple of 16
  //   OnePTwoC (3 branches): multiple of 12
  // Load-first TwoPTwoC SetVgpr emission: [load, load, main, tail]
  // ⇒ printed BranchAvailable ≈ [tail, main, load, load].
  // Pad tail so e.g. [88, 88, 144, 140] → [88, 88, 144, 144]
  // ⇒ Available [144, 144, 88, 88] (avg 116, %4==0).
  if (wdraEnabled && topo.kind == triton::HCU::WdraTopoKind::TwoPTwoC) {
    int sum = wdraNumLoadRegs + wdraNumLoadRegs + wdraNumMmaRegsMain +
              wdraNumMmaRegsTail;
    int rem = sum % 16;
    if (rem != 0)
      wdraNumMmaRegsTail += 16 - rem;
  } else if (wdraEnabled && topo.kind == triton::HCU::WdraTopoKind::OnePTwoC) {
    int sum = wdraNumLoadRegs + wdraNumMmaRegsMain + wdraNumMmaRegsTail;
    // Do NOT pad to 16 here: e.g. 88+208+208=504 already has avg 168 (%4==0),
    // but padding to 512 makes sum % 3 != 0 and clang rejects the kernel.
    int rem = sum % 12;
    if (rem != 0)
      wdraNumMmaRegsMain += 12 - rem;
  }

  // Replace the s_barrier in each partition with ebarrier.
  unsigned mmaNumWarps =
      wdraEnabled ? static_cast<unsigned>(waspNumMmaWarps) : defaultNumWarps;
  if (failed(rewriteWarpGroupBarriers(func, wsOps, defaultNumWarps,
                                      mmaNumWarps)))
    return failure();

  // Attempt to elide captures of trivial computations by hoisting them into the
  // header or rematerializing them into each partition.
  elideTrivialCaptures(func, wsOps);

  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);
  LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);

  // The default partition is executed by partition0 — Load-first layout, so
  // allocate load VGPRs for the first partition when WDRA is enabled.
  b.setInsertionPointToStart(entry);
  if (wdraEnabled)
    b.create<ROCDL::HCUSetVgprSizeOp>(wdraNumLoadRegs);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });

  // This is the default destination for the switch op.
  // Does nothing except arriving all the barriers and then return.
  Region::BlockListType &funcBlocks = func.getBody().getBlocks();
  Block *defaultBlock = new Block;
  auto firstWsOp = wsOps.front();
  Block *firstWsOpBlock = firstWsOp.getOperation()->getBlock();
  funcBlocks.insert(firstWsOpBlock->getIterator(), defaultBlock);
  b.setInsertionPointToStart(defaultBlock);
  createBarrier(b, kSwitchLoopBarrierIdx, /*numWarps=*/std::nullopt);
  createBarrier(b, kPartitionEndBarrierIdx, /*numWarps=*/std::nullopt);
  createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
  b.create<LLVM::ReturnOp>(ValueRange());

  // Create the switch op in the first block.
  b.setInsertionPointToStart(header);
  // Declare per-branch VGPR sizes once at kernel entry for WDRA. The backend
  // HCUInsertWDRAInit pass consumes this: on HW it emits the s_trap 0xff wdra
  // prologue with s_set_vgpr_size per branch; on the model (PMD) it lowers to a
  // comment and VGPR allocation is resolved from HSACO metadata.
  //
  // Ordered by ascending wave id (see ROCDL_HCUWdraInitOp):
  //   TwoPTwoC Load-first: [load, load, mma_main, mma_tail]
  if (wdraEnabled) {
    if (topo.kind == triton::HCU::WdraTopoKind::TwoPTwoC) {
      b.create<ROCDL::HCUWdraInitOp>(wdraNumLoadRegs, wdraNumLoadRegs,
                                     wdraNumMmaRegsMain, wdraNumMmaRegsTail);
    } else if (topo.kind == triton::HCU::WdraTopoKind::OnePTwoC) {
      b.create<ROCDL::HCUWdraInitOp>(wdraNumLoadRegs, wdraNumMmaRegsMain,
                                     wdraNumMmaRegsTail, /*branch4=*/0);
    } else {
      // Unsplit WDRA [Load, MMA]
      b.create<ROCDL::HCUWdraInitOp>(wdraNumLoadRegs, wdraNumMmaRegsMain,
                                     /*branch3=*/0, /*branch4=*/0);
    }
  }
  Value wid = b.create<ROCDL::HCUGetWaveIdOp>();

  // First partition (Load) warps land on the entry block.
  unsigned firstPartWarps = firstWsOp.getPartitionNumWarps()[0];
  SmallVector<Block *> mmaPartitionVgprSettingBlocks;
  SmallVector<Block *> loadPartitionVgprSettingBlocks;
  SmallVector<Block *> switchBlocks(firstPartWarps, entry);

  auto caseValues = llvm::to_vector(llvm::map_range(
      llvm::seq<int8_t>(firstPartWarps),
      [](int8_t i) { return APInt(8, i); }));

  // Helper: VGPR budget for partition idx under Load-first layout.
  auto regsForPartition = [&](unsigned idx) -> int {
    if (topo.isLoadPartition(idx))
      return wdraNumLoadRegs;
    if (topo.isMmaMainPartition(idx))
      return wdraNumMmaRegsMain;
    return wdraNumMmaRegsTail;
  };

  // Set the rest target blocks for the switch op.
  {
    TritonLLVMIRRewriter b(firstWsOp.getLoc(), firstWsOp);
    // Counter to generate unique block identifiers to prevent block merging
    unsigned blockCounter = 0;
    unsigned nextCaseValue = firstPartWarps;
    for (auto [idx, tuple] : llvm::enumerate(llvm::zip(
      *firstWsOp.getWarpGroupStartIds(), firstWsOp.getPartitionNumWarps(), firstWsOp.getPartitionRegions()))) {
      auto [startId, numWarps, partition] = tuple;
      // idx 0: first Load partition, already reached via entry block; skip.
      if (idx == 0)
        continue;
      Block *vgprSettingBlock = b.createBlock(entry);
      b.setInsertionPointToStart(vgprSettingBlock);
      int numRegs = regsForPartition(idx);
      auto vgprOp = b.create<ROCDL::HCUSetVgprSizeOp>(numRegs);
      vgprOp->setAttr("block_id", b.getI32IntegerAttr(blockCounter));
      auto brOp = b.create<LLVM::BrOp>(&partition->front());
      brOp->setAttr("block_id", b.getI32IntegerAttr(blockCounter));
      blockCounter++;
      if (topo.isLoadPartition(idx))
        loadPartitionVgprSettingBlocks.push_back(vgprSettingBlock);
      else
        mmaPartitionVgprSettingBlocks.push_back(vgprSettingBlock);
      for (auto i : llvm::seq(numWarps)) {
        Block *targetBlock = wdraEnabled ? vgprSettingBlock : &partition->front();
        caseValues.push_back(APInt(8, nextCaseValue + i));
        switchBlocks.push_back(targetBlock);
      }
      nextCaseValue += numWarps;
    }
  }

  b.create<LLVM::SwitchOp>(wid, defaultBlock, ValueRange(), caseValues, switchBlocks,
                           SmallVector<ValueRange>(switchBlocks.size()));

  for (auto [idx, ws] : llvm::enumerate(wsOps)) {
    bool isFirst = (idx == 0);
    bool isLast = (idx == wsOps.size() - 1);

    Block *before = ws->getBlock();
    Block *after = b.splitBlock(before, ws->getIterator());
    TritonLLVMIRRewriter b(ws.getLoc(), OpBuilder::atBlockEnd(before));

    // Store the captures if there are any.
    if (ws.getNumOperands()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(ws.getOperandTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      for (auto [i, arg] : llvm::zip(llvm::seq<int32_t>(ws.getNumOperands()),
                                     ws.getOperands())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        b.store(arg, ptr, /*align=*/1);
      }
    }

    b.create<LLVM::BrOp>(&ws.getPartitionRegions()[0]->front());

    ws.getDefaultRegion().walk([&, ws = ws](WarpYieldOp op) mutable {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      createBarrier(b, kPartitionEndBarrierIdx, /*numWarps=*/std::nullopt);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, op.getOperands(), after);
    });
    after->getParent()->getBlocks().splice(after->getIterator(),
                                           ws.getDefaultRegion().getBlocks());

    // Replace the results.
    auto outputs = after->addArguments(
        ws.getResultTypes(),
        SmallVector<Location>(ws.getNumResults(), ws.getLoc()));
    ws.replaceAllUsesWith(outputs);

    // Rewrite the partition regions
    SmallVector<Block *> sinkBlocks(ws.getPartitionRegions().size(), nullptr);
    rewritePartitionRegions(ws, sinkBlocks, targetInfo, topo);
  }

  // Reorder basic blocks so that blocks belonging to different branches
  // (partitions) stay grouped together without interleaving, and tag MMA
  // partitions with distinct attributes on their epilogue blocks so that later
  // optimization passes do not merge them.
  //
  // Layout follows Load-first partition index / wave-id order:
  //   WASP / unsplit: Load, MMA
  //   OnePTwoC: Load, MMA_main, MMA_tail
  //   TwoPTwoC: Load_main, Load_tail, MMA_main, MMA_tail
  auto partitionNumWarps = firstWsOp.getPartitionNumWarps();
  unsigned numPartitions = partitionNumWarps.size();

  auto tagMmaPartitionBlocks = [&](Region *region, unsigned partitionId) {
    // Insert a tiny partition-specific inline asm marker in the epilogue blocks
    // (right before returns). This makes the epilogues of different MMA
    // partitions structurally different so that the backend won't merge them,
    // while having no semantic effect.
    for (Block &b : region->getBlocks()) {
      for (Operation &op : b) {
        if (auto ret = dyn_cast<LLVM::ReturnOp>(&op)) {
          OpBuilder markerBuilder(ret);
          auto *ctx = markerBuilder.getContext();
          std::string asmStr = "; mma_epilogue_partition_" + std::to_string(partitionId);
          auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
          markerBuilder.create<LLVM::InlineAsmOp>(
              ret.getLoc(),
              TypeRange(),          // no results
              ValueRange(),         // no operands
              asmStr,               // asm string
              "",                   // constraints
              /*hasSideEffects=*/true,
              /*isAlignStack=*/false,
              LLVM::TailCallKind::None,
              asmDialect,
              ArrayAttr::get(ctx, {}));
          // Only need one marker per epilogue block.
          break;
        }
      }
    }
  };

  unsigned mmaVgprIdx = 0;
  unsigned loadVgprIdx = 0;
  for (unsigned p = 0; p < numPartitions; ++p) {
    bool isLoad = topo.isLoadPartition(p);
    if (p != 0) {
      Block *vgprBlock = nullptr;
      if (isLoad) {
        assert(loadVgprIdx < loadPartitionVgprSettingBlocks.size());
        vgprBlock = loadPartitionVgprSettingBlocks[loadVgprIdx++];
      } else {
        assert(mmaVgprIdx < mmaPartitionVgprSettingBlocks.size());
        vgprBlock = mmaPartitionVgprSettingBlocks[mmaVgprIdx++];
      }
      vgprBlock->getParent()->getBlocks().remove(vgprBlock);
      funcBlocks.insert(funcBlocks.end(), vgprBlock);
    }

    for (auto ws : wsOps) {
      if (p >= ws.getPartitionRegions().size())
        continue;
      auto region = ws.getPartitionRegions()[p];
      if (!isLoad)
        tagMmaPartitionBlocks(region, /*partitionId=*/p);
      funcBlocks.splice(funcBlocks.end(), region->getBlocks());
    }
  }

  for (auto ws : wsOps) {
    ws.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct HCUConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::HCUConvertWarpSpecializeToLLVMBase<
          HCUConvertWarpSpecializeToLLVM> {
  explicit HCUConvertWarpSpecializeToLLVM(StringRef targetArch, int waspNumLoadWarps, 
      int waspNumMmaWarps, bool wdraEnabled, int wdraNumLoadRegs, 
      int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {

    this->arch = targetArch.str();
    this->waspNumLoadWarps = waspNumLoadWarps;
    this->waspNumMmaWarps = waspNumMmaWarps;
    this->wdraEnabled = wdraEnabled;
    this->wdraNumLoadRegs = wdraNumLoadRegs;
    this->wdraNumMmaRegsMain = wdraNumMmaRegsMain;
    this->wdraNumMmaRegsTail = wdraNumMmaRegsTail;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    AMD::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getGPUKind() != llvm::AMDGPU::GK_GFX946) {
      mod.emitError("wasp unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    // Convert types and cleanup unrealized conversions.
    mlir::LowerToLLVMOptions option(&getContext());
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(&getContext(), option, targetInfo);
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        convertOpTypes(op, typeConverter);
    });
    RewritePatternSet patterns(&getContext());
    UnrealizedConversionCastOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }

    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo, 
        this->waspNumLoadWarps, this->waspNumMmaWarps, this->wdraEnabled,
        this->wdraNumLoadRegs, this->wdraNumMmaRegsMain, this->wdraNumMmaRegsTail)))
        return signalPassFailure();

    // Convert GPU dialect to ROCDL dialect
    FailureOr<mlir::amdgpu::Chipset> maybeChipset = mlir::amdgpu::Chipset::parse(this->arch.getValue());
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid AMDGPU chipset name: " + this->arch.getValue());
      return signalPassFailure();
    }

    // Create conversion target for GPU to ROCDL conversion
    ConversionTarget gpuTarget(getContext());
    gpuTarget.addLegalDialect<LLVM::LLVMDialect>();
    gpuTarget.addLegalDialect<ROCDL::ROCDLDialect>();
    gpuTarget.addIllegalDialect<mlir::gpu::GPUDialect>();

    // Create patterns for GPU to ROCDL conversion
    RewritePatternSet gpuPatterns(&getContext());
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, gpuPatterns, mlir::gpu::amd::HIP, *maybeChipset);

    // Apply GPU to ROCDL conversion
    if (failed(applyPartialConversion(mod, gpuTarget, std::move(gpuPatterns)))) {
      return signalPassFailure();
    }

    // Elide canceling casts: unrealized_conversion_cast (A -> B) followed by
    // arith.index_cast (B -> A) can be removed.
    SmallVector<std::pair<UnrealizedConversionCastOp, arith::IndexCastOp>> cancelingPairs;
    mod.walk([&](UnrealizedConversionCastOp unrealizedCast) {
      if (unrealizedCast.getNumOperands() != 1 || unrealizedCast.getNumResults() != 1)
        return;
      Value original = unrealizedCast.getOperand(0);
      Value mid = unrealizedCast.getResult(0);
      if (!mid.hasOneUse())
        return;
      Operation *user = *mid.getUsers().begin();
      auto indexCast = dyn_cast<arith::IndexCastOp>(user);
      if (!indexCast)
        return;
      // Check types form A->B then B->A
      if (indexCast.getType() != original.getType())
        return;
      if (indexCast.getIn().getType() != mid.getType())
        return;
      cancelingPairs.emplace_back(unrealizedCast, indexCast);
    });
    for (auto [unrealizedCast, indexCast] : cancelingPairs) {
      Value original = unrealizedCast.getOperand(0);
      indexCast.getResult().replaceAllUsesWith(original);
      indexCast.erase();
      unrealizedCast.erase();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Factory Function
//===----------------------------------------------------------------------===//
namespace mlir::triton {

  std::unique_ptr<OperationPass<ModuleOp>>
  createHCUConvertWarpSpecializeToLLVM(StringRef targetArch, int waspNumLoadWarps, int waspNumMmaWarps, 
     bool wdraEnabled, int wdraNumLoadRegs, int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {
    return std::make_unique<HCUConvertWarpSpecializeToLLVM>(targetArch, waspNumLoadWarps, waspNumMmaWarps,
      wdraEnabled, wdraNumLoadRegs, wdraNumMmaRegsMain, wdraNumMmaRegsTail);
  }
  
  } // namespace mlir::triton