// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "TritonHCU/Passes.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "TritonHCU/BufferLdsPipeline.h"
#include "TritonHCU/BufferLdsEncoding.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;
namespace mlir::triton::HCU::planned {
static bool requiresWrappedProducer(ttg::MemDescType type) {
  return isa<ttg::HCUBufferLdsSharedEncodingAttr>(type.getEncoding());
}

static bool hasKnownPublicationEffects(Operation *op) {
  if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin() == "llvm.amdgcn.readfirstlane";
  if (auto assembly = dyn_cast<LLVM::InlineAsmOp>(op))
    return assembly.getAsmString().starts_with("s_waitcnt") ||
           assembly.getAsmString() == "s_barrier";
  return isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::LocalAllocOp,
             ttg::LocalDeallocOp, ttg::LocalBarrierOp,
             ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp,
             ttg::AsyncWaitOp, triton::amdgpu::BufferLoadToLocalOp,
             triton::amdgpu::BufferLoadOp, triton::LoadOp,
             triton::amdgpu::MemoryCounterWaitOp, ROCDL::SchedBarrier,
             ROCDL::SBarrierOp, ROCDL::SetPrioOp, mlir::gpu::BarrierOp,
             triton::amdgpu::CondBarrierOp>(op) ||
         isMemoryEffectFree(op);
}

// Partial publication is proved only for independent static allocations.
// A statically bounded subslice aliases its backing allocation as a whole:
// this may reject a disjoint-half optimization, but can never hide a reuse of
// either half. Dynamic indices, reinterpretations and CFG merges still drain.
static Value staticSlot(Value value) {
  while (auto view = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
    auto src = view.getSrc().getType();
    auto dst = view.getType();
    auto offsets = view.getOffsets();
    if (src.getRank() != dst.getRank() ||
        src.getRank() != static_cast<int64_t>(offsets.size()) ||
        src.getElementType() != dst.getElementType() ||
        src.getEncoding() != dst.getEncoding())
      return {};
    for (auto [offset, srcExtent, dstExtent] :
         llvm::zip(offsets, src.getShape(), dst.getShape()))
      if (offset < 0 || dstExtent > srcExtent || offset > srcExtent - dstExtent)
        return {};
    value = view.getSrc();
  }
  auto alloc = value.getDefiningOp<ttg::LocalAllocOp>();
  return alloc && !alloc.getSrc() ? value : Value();
}

static bool isReadDrain(Operation *op) {
  auto wait = dyn_cast<triton::amdgpu::MemoryCounterWaitOp>(op);
  if (!wait || !wait.getDsAttr() || wait.getDsAttr().getInt() != 0)
    return false;
  auto next = op->getNextNode();
  while (next && isa<ROCDL::SchedBarrier>(next))
    next = next->getNextNode();
  return next && isa<ROCDL::SBarrierOp>(next);
}

static bool hasEntryReadDrain(scf::ForOp loop) {
  for (Operation *op = loop->getPrevNode(); op; op = op->getPrevNode()) {
    if (isa<mlir::gpu::BarrierOp, ttg::LocalBarrierOp>(op) || isReadDrain(op))
      return true;
    if (op->getNumRegions() || !hasKnownPublicationEffects(op) ||
        isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttg::LocalAllocOp,
            ttg::LocalDeallocOp>(op))
      return false;
  }
  return false;
}

// These reads produce future SSA fragments and cannot alias the current DMA
// destinations. A full DS wait + workgroup rendezvous bounds both sides of
// the phase, including the backedge. Thus no *additional* DS wait is needed
// at publication. This proof is independent of instruction width/count.
static bool canDeferPublicationReads(ArrayRef<Operation *> body, unsigned pos) {
  llvm::DenseSet<Value> reads, writes;
  bool priorDrain = false;
  for (unsigned step = 1; step <= body.size(); ++step) {
    Operation *op = body[(pos + body.size() - step) % body.size()];
    if (isReadDrain(op)) {
      priorDrain = true;
      break;
    }
    if (op->hasAttr(triton::HCU::kBufferToLdsPublication))
      return false;
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      Value slot = staticSlot(load.getSrc());
      if (!slot)
        return false;
      reads.insert(slot);
    } else if (auto copy = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
      Value slot = staticSlot(copy.getDest());
      if (!slot)
        return false;
      writes.insert(slot);
    } else if (isa<ttg::LocalStoreOp, ttg::AsyncCopyGlobalToLocalOp,
                   ttg::LocalAllocOp, ttg::LocalDeallocOp>(op)) {
      return false;
    }
  }
  if (!priorDrain || reads.empty() || writes.empty() ||
      llvm::any_of(reads, [&](Value slot) { return writes.contains(slot); }))
    return false;
  for (unsigned step = 1; step < body.size(); ++step) {
    Operation *op = body[(pos + step) % body.size()];
    if (isReadDrain(op))
      return true;
    if (op->hasAttr(triton::HCU::kBufferToLdsPublication) ||
        isa<ttg::LocalStoreOp, ttg::AsyncCopyGlobalToLocalOp,
            triton::amdgpu::BufferLoadToLocalOp, ttg::LocalAllocOp,
            ttg::LocalDeallocOp>(op))
      return false;
  }
  return false;
}

// Count newer *commit groups* before reaching a commit that covers the
// requested slot. One group may lower to several DMA instructions; LLVM owns
// that conversion. The sequence is reverse program order on one CFG path.
std::optional<unsigned>
newerGroupsForSlot(ArrayRef<Operation *> reversePath, Value slot) {
  unsigned newer = 0;
  for (Operation *op : reversePath) {
    if (!hasKnownPublicationEffects(op))
      return std::nullopt;
    if (auto commit = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
      for (Value token : commit.getOperands()) {
        auto copy = token.getDefiningOp<triton::amdgpu::BufferLoadToLocalOp>();
        if (copy && copy.getDest() == slot)
          return newer;
      }
      // LLVM's CFG merge retains at most 16 marks. Do not depend on a
      // truncated history or overflow the i16 wait_asyncmark operand.
      if (++newer >= 16)
        return std::nullopt;
    }
    if (op->getNumRegions())
      return std::nullopt;
    if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
      if (!staticSlot(store.getDst()) || store.getDst() == slot)
        return std::nullopt;
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::LocalDeallocOp>(op))
      return std::nullopt;
  }
  return std::nullopt;
}

// The initial register prefetch may also feed the first loop iteration before
// its first publication. Include that read prefix when bounding the entry
// wait: waiting for only the first A/B pair would leave the next pair racing
// its DMA. Independent static slots and a straight-line first iteration make
// this proof independent of the number of instructions per copy.
unsigned entryPrefetchPendingGroups(ttg::LocalLoadOp first) {
  if (!isa<triton::FuncOp>(first->getParentOp()))
    return 0;
  llvm::DenseSet<Value> targets, written;
  bool enteredLoop = false, reachedBoundary = false;
  Operation *cursor = first;
  for (; cursor; cursor = cursor->getNextNode()) {
    if (auto loop = dyn_cast<scf::ForOp>(cursor)) {
      if (enteredLoop || !loop->hasAttr("hcu.block_pingpong.static_buffers"))
        return 0;
      enteredLoop = true;
      cursor = &loop.getBody()->front();
    }
    if (auto copy = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(cursor)) {
      Value slot = staticSlot(copy.getDest());
      if (!enteredLoop || !slot)
        return 0;
      written.insert(slot);
      continue;
    }
    if (isa<ttg::AsyncWaitOp, ttg::LocalBarrierOp>(cursor)) {
      reachedBoundary = true;
      break;
    }
    if (auto load = dyn_cast<ttg::LocalLoadOp>(cursor)) {
      Value slot = staticSlot(load.getSrc());
      if (!slot || written.contains(slot))
        return 0;
      targets.insert(slot);
    } else if (cursor->getNumRegions() || !hasKnownPublicationEffects(cursor) ||
               isa<ttg::LocalStoreOp, ttg::LocalAllocOp, ttg::LocalDeallocOp,
                   ttg::AsyncCopyGlobalToLocalOp>(cursor)) {
      return 0;
    }
  }
  if (!enteredLoop || !reachedBoundary || targets.empty())
    return 0;
  SmallVector<Operation *> reverseEntry;
  for (Operation *op = first->getPrevNode(); op; op = op->getPrevNode())
    reverseEntry.push_back(op);
  unsigned pending = 15;
  for (Value slot : targets) {
    auto count = newerGroupsForSlot(reverseEntry, slot);
    if (!count)
      return 0;
    pending = std::min(pending, *count);
  }
  return pending;
}

static bool isMatrixRead(ttg::LocalLoadOp read) {
  return triton::HCU::canUseBufferLdsMatrixRead(
      read.getSrc().getType(),
      dyn_cast<ttg::DotOperandEncodingAttr>(read.getType().getEncoding()));
}

// DMA-first is a static-slot optimization, not a change to the abstract ring
// schedule. Keep the dynamic remainder and pipeline epilogue untouched. At
// this point the copies have their final destinations, so delaying independent
// reads across them requires neither pointer retiming nor a guessed ring index.
static void scheduleStaticCopiesBeforeReads(ModuleOp module) {
  module.walk([&](scf::ForOp loop) {
    if (!loop->hasAttr("hcu.block_pingpong.static_buffers"))
      return;
    SmallVector<ttg::LocalLoadOp> reads;
    SmallVector<triton::amdgpu::BufferLoadToLocalOp> copies;
    bool known = true;
    for (Operation &op : *loop.getBody()) {
      if (op.hasAttr(triton::HCU::kBufferToLdsPublication)) {
        if (known && !reads.empty() && !copies.empty()) {
          Operation *last = copies.back();
          bool independent = llvm::all_of(reads, [&](ttg::LocalLoadOp read) {
            Value slot = staticSlot(read.getSrc());
            return slot && read->isBeforeInBlock(copies.front()) &&
                   llvm::none_of(
                       copies,
                       [&](auto copy) { return copy.getDest() == slot; }) &&
                   llvm::all_of(read->getUsers(), [&](Operation *user) {
                     return user->getBlock() == last->getBlock() &&
                            last->isBeforeInBlock(user);
                   });
          });
          if (independent) {
            // Prefer the matrix-read fragment before ordinary DS reads. Use
            // the lowering contract, not A/B names or a fixed element shape.
            llvm::stable_sort(reads, [&](auto a, auto b) {
              return isMatrixRead(a) && !isMatrixRead(b);
            });
            for (auto read : reads) {
              read->moveAfter(last);
              last = read;
            }
          }
        }
        reads.clear();
        copies.clear();
        known = true;
      } else if (isReadDrain(&op)) {
        reads.clear();
        copies.clear();
        known = true;
      } else if (auto read = dyn_cast<ttg::LocalLoadOp>(op)) {
        reads.push_back(read);
      } else if (auto copy =
                     dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
        known &= bool(staticSlot(copy.getDest()));
        copies.push_back(copy);
      } else if (isa<ROCDL::SBarrierOp>(op)) {
        known &= reads.empty() && copies.empty();
      } else if (op.getNumRegions() ||
                 !(isMemoryEffectFree(&op) ||
                   isa<ROCDL::SchedBarrier, ROCDL::SetPrioOp>(op))) {
        known = false;
      }
    }
  });
}


// Derive partial waits from the next consumers and both loop-entry/backedge
// producer histories. Unknown histories keep a full wait.
static void resolvePublicationWaits(ModuleOp module) {
  SmallVector<Operation *> publications;
  module.walk([&](Operation *op) {
    if (op->hasAttr(triton::HCU::kBufferToLdsPublication) &&
        isa<ttg::LocalBarrierOp>(op))
      publications.push_back(op);
  });
  // Analyze before mutating any barrier so earlier replacements cannot alter
  // the phase boundaries used to prove later publications.
  SmallVector<std::pair<unsigned, bool>> plans;
  for (auto barrier : publications) {
    unsigned pending = 0;
    bool deferReads = false;
    auto loop = dyn_cast<scf::ForOp>(barrier->getParentOp());
    if (loop && loop->hasAttr("hcu.block_pingpong.static_buffers") &&
        llvm::all_of(*loop.getBody(), [](Operation &op) {
          return op.getNumRegions() == 0 && hasKnownPublicationEffects(&op);
        })) {
      SmallVector<Operation *> body;
      for (Operation &op : *loop.getBody())
        body.push_back(&op);
      unsigned pos = llvm::find(body, barrier) - body.begin();
      llvm::DenseSet<Value> targets, written;
      bool known = true;
      for (unsigned step = 1; step < body.size(); ++step) {
        Operation *op = body[(pos + step) % body.size()];
        if (op->hasAttr(triton::HCU::kBufferToLdsPublication))
          break;
        if (auto copy = dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op))
          written.insert(copy.getDest());
        if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
          Value slot = staticSlot(load.getSrc());
          if (!slot || written.contains(slot)) {
            known = false;
            break;
          }
          targets.insert(slot);
        }
      }
      // Both loop entry and backedge must permit the count. Taking the
      // minimum also waits for the newest producer among all next reads.
      SmallVector<Operation *> entry, backedge;
      for (unsigned i = pos; i > 0; --i)
        entry.push_back(body[i - 1]);
      backedge = entry;
      for (Operation *op : llvm::reverse(body))
        backedge.push_back(op);
      for (Operation *op = loop->getPrevNode(); op; op = op->getPrevNode())
        entry.push_back(op);
      unsigned bound = 15;
      for (Value slot : targets) {
        auto initial = newerGroupsForSlot(entry, slot);
        auto recurrent = newerGroupsForSlot(backedge, slot);
        if (!initial || !recurrent) {
          known = false;
          break;
        }
        bound = std::min({bound, *initial, *recurrent});
      }
      if (known && !targets.empty())
        pending = bound;
      deferReads = hasEntryReadDrain(loop) &&
                   canDeferPublicationReads(body, pos);
    }
    plans.push_back({pending, deferReads});
  }
  for (auto [barrier, plan] : llvm::zip(publications, plans)) {
    OpBuilder builder(barrier);
    ttg::AsyncWaitOp::create(builder, barrier->getLoc(), ValueRange{},
                             plan.first);
    barrier->removeAttr(triton::HCU::kBufferToLdsPublication);
    if (plan.second)
      barrier->setAttr(triton::HCU::kDropPublicationLgkmcnt,
                       builder.getUnitAttr());
  }
}
static bool hasFourScalarAndFourMatrixReads(
    ArrayRef<ttg::LocalLoadOp> reads) {
  if (reads.size() != 2)
    return false;
  unsigned matrixReads = 0;
  for (auto read : reads) {
    auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(read.getType().getEncoding());
    if (!dot || dot.getKWidth() != 4)
      return false;
    bool matrix = isMatrixRead(read);
    matrixReads += matrix;
    // This is a supported instruction footprint, not a general LDS counter
    // model. Wider/narrower fragments retain the conservative schedule.
    unsigned elementsPerInstruction = matrix ? 8 : 4;
    if (ttg::getTotalElemsPerThread(read.getType()) !=
        4 * elementsPerInstruction)
      return false;
  }
  return matrixReads == 1;
}

// Stagger the two wave groups of a proven four-generation/R1 pipeline.
// Selection depends on the actual LDS slots, fragments and rendezvous, never
// on a constant reduction bound. A mismatch leaves the original loop intact.
// The runtime trip count and the generic prologue/epilogue are preserved.
static LogicalResult staggerBufferLdsWaveGroups(ModuleOp module) {
  bool transformed = false;
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) {
    if (loop->hasAttr("hcu.block_pingpong.static_buffers"))
      loops.push_back(loop);
  });
  if (loops.size() != 1)
    return failure();
  for (auto loop : loops) {
    unsigned waves = ttg::lookupNumWarps(loop);
    if (waves != 8 || !hasEntryReadDrain(loop))
      continue;
    struct Phase {
      ttg::AsyncWaitOp wait;
      ttg::LocalBarrierOp publication;
      triton::amdgpu::MemoryCounterWaitOp drain;
      ROCDL::SBarrierOp control;
      SmallVector<ttg::LocalLoadOp> reads;
      SmallVector<triton::amdgpu::BufferLoadToLocalOp> copies;
    };
    SmallVector<Phase> phases;
    SmallVector<ttg::LocalLoadOp> firstReads;
    triton::amdgpu::BufferLoadToLocalOp firstCopy;
    llvm::DenseSet<Value> reads, writes;
    llvm::DenseSet<Operation *> controls;
    Phase phase;
    bool valid = true;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (op.getNumRegions() || !hasKnownPublicationEffects(&op)) {
        valid = false;
        break;
      }
      if (auto read = dyn_cast<ttg::LocalLoadOp>(op)) {
        Value slot = staticSlot(read.getSrc());
        valid &= bool(slot) && !phase.publication;
        reads.insert(slot);
        phase.reads.push_back(read);
        if (phases.empty())
          firstReads.push_back(read);
      } else if (auto copy =
                     dyn_cast<triton::amdgpu::BufferLoadToLocalOp>(op)) {
        Value slot = staticSlot(copy.getDest());
        valid &= bool(slot) && !phase.publication;
        writes.insert(slot);
        phase.copies.push_back(copy);
        auto type = copy.getDest().getType();
        valid &=
            requiresWrappedProducer(type) &&
            (type.getElementType().isF16() || type.getElementType().isBF16());
        if (!firstCopy)
          firstCopy = copy;
      } else if (auto publication = dyn_cast<ttg::LocalBarrierOp>(op)) {
        valid &= !phase.publication &&
                 publication->hasAttr(triton::HCU::kDropPublicationLgkmcnt);
        phase.publication = publication;
        phase.wait = dyn_cast_or_null<ttg::AsyncWaitOp>(op.getPrevNode());
        valid &= bool(phase.wait);
      } else if (isReadDrain(&op)) {
        phase.drain = cast<triton::amdgpu::MemoryCounterWaitOp>(op);
        valid &= bool(phase.publication) && !phase.drain.getLoadAttr() &&
                 !phase.drain.getStoreAttr() && !reads.empty() &&
                 !writes.empty();
        valid &= llvm::none_of(
            reads, [&](Value slot) { return writes.contains(slot); });
        Operation *next = op.getNextNode();
        while (isa_and_nonnull<ROCDL::SchedBarrier>(next))
          next = next->getNextNode();
        phase.control = dyn_cast_or_null<ROCDL::SBarrierOp>(next);
        if (!phase.control) {
          valid = false;
          break;
        }
        controls.insert(next);
        phases.push_back(phase);
        phase = Phase();
        reads.clear();
        writes.clear();
      } else if (isa<ROCDL::SBarrierOp>(op)) {
        valid &= controls.contains(&op);
      } else if (isa<ttg::LocalStoreOp, ttg::LocalAllocOp, ttg::LocalDeallocOp,
                     ttg::AsyncCopyGlobalToLocalOp, mlir::gpu::BarrierOp,
                     triton::amdgpu::CondBarrierOp>(op)) {
        valid = false;
      }
    }
    if (!valid || phases.size() != 4 || phase.publication || !reads.empty() ||
        !writes.empty() || !firstCopy || firstReads.empty())
      continue;
    if (llvm::any_of(phases, [](auto &p) {
          return p.reads.size() != 2 || p.copies.size() != 2;
        }))
      continue;
    // Reads moved earlier must retain dominating address definitions; reads
    // moved later must still precede their consumers.
    DominanceInfo dominance(module);
    for (auto [index, p] : llvm::enumerate(phases)) {
      Operation *anchor = p.wait;
      for (auto read : p.reads) {
        valid &= llvm::all_of(read->getUsers(), [&](Operation *user) {
          return dominance.properlyDominates(anchor, user);
        });
        if (index == 0)
          valid &= llvm::all_of(read->getOperands(), [&](Value operand) {
            return dominance.dominates(operand, p.copies.front());
          });
      }
    }
    if (!valid)
      continue;
    // Preserve the established b64/matrix-read retirement contract. The
    // count describes the newer fragment, not K or the number of loop trips.
    // Other fragment widths keep the conservative unsplit pipeline.
    constexpr unsigned pendingReads = 8;
    if (llvm::any_of(phases, [](auto &p) {
          return !hasFourScalarAndFourMatrixReads(p.reads);
        }))
      continue;
    triton::amdgpu::CondBarrierOp entry;
    for (Operation *op = loop->getPrevNode(); op; op = op->getPrevNode()) {
      if (isa<mlir::gpu::BarrierOp, ttg::LocalBarrierOp>(op) ||
          op->getNumRegions())
        break;
      if ((entry = dyn_cast<triton::amdgpu::CondBarrierOp>(op)))
        break;
    }
    auto cmp = entry ? entry.getPred().getDefiningOp<arith::CmpIOp>()
                     : arith::CmpIOp();
    APInt half;
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::ult ||
        !matchPattern(cmp.getRhs(), m_ConstantInt(&half)) ||
        half.getZExtValue() != waves / 2)
      continue;
    SmallVector<triton::amdgpu::CondBarrierOp> exits;
    for (Operation *op = loop->getNextNode(); op; op = op->getNextNode())
      if (auto exit = dyn_cast<triton::amdgpu::CondBarrierOp>(op))
        exits.push_back(exit);
    if (exits.size() != 1)
      continue;
    auto exitCmp = exits.front().getPred().getDefiningOp<arith::CmpIOp>();
    if (!exitCmp || exitCmp.getPredicate() != arith::CmpIPredicate::uge ||
        exitCmp.getLhs() != cmp.getLhs() || exitCmp.getRhs() != cmp.getRhs())
      continue;
    // Match the reference's A-before-B startup as well as its steady state.
    // These initial fragments have no users before the loop's entry drain.
    SmallVector<ttg::LocalLoadOp> initialReads;
    for (Operation *op = loop->getPrevNode(); op; op = op->getPrevNode())
      if (auto read = dyn_cast<ttg::LocalLoadOp>(op))
        initialReads.push_back(read);
    if (initialReads.size() == 2) {
      auto a = isMatrixRead(initialReads[0]) ? initialReads[1] : initialReads[0];
      auto bRead =
          isMatrixRead(initialReads[0]) ? initialReads[0] : initialReads[1];
      if (a != bRead && bRead->isBeforeInBlock(a) &&
          llvm::all_of(a->getOperands(), [&](Value operand) {
            return dominance.dominates(operand, bRead);
          })) {
        a->moveBefore(bRead);
        OpBuilder fence(bRead);
        ROCDL::SchedBarrier::create(fence, loop.getLoc(), 0);
      }
    }
    OpBuilder b(loop);
    auto groups = scf::IfOp::create(b, loop.getLoc(), loop.getResultTypes(),
                                    entry.getPred(), true);
    b.setInsertionPointToStart(groups.thenBlock());
    IRMapping lowMap;
    auto low = cast<scf::ForOp>(b.clone(*loop, lowMap));
    for (auto &p : phases) {
      auto drain = cast<triton::amdgpu::MemoryCounterWaitOp>(
          lowMap.lookup(p.drain.getOperation()));
      drain.setDsAttr(b.getI32IntegerAttr(pendingReads));
      drain->moveAfter(lowMap.lookup(p.publication.getOperation()));
    }
    lowMap.lookup(phases.back().control.getOperation())->erase();
    scf::YieldOp::create(b, loop.getLoc(), low.getResults());
    b.setInsertionPointToStart(groups.elseBlock());
    IRMapping highMap;
    auto high = cast<scf::ForOp>(b.clone(*loop, highMap));
    for (auto &p : phases) {
      auto drain = cast<triton::amdgpu::MemoryCounterWaitOp>(
          highMap.lookup(p.drain.getOperation()));
      drain.setDsAttr(b.getI32IntegerAttr(pendingReads));
      drain->moveAfter(highMap.lookup(p.publication.getOperation()));
    }
    Operation *firstControl =
        highMap.lookup(phases.front().control.getOperation());
    highMap.lookup(phases.front().wait.getOperation())
        ->moveBefore(firstControl);
    // Keep the async wait adjacent to the publication for MembarAnalysis.
    b.setInsertionPoint(firstControl);
    auto highPublication = ttg::LocalBarrierOp::create(b, loop.getLoc());
    highPublication->setAttr(triton::HCU::kDropPublicationLgkmcnt,
                             b.getUnitAttr());
    firstControl->erase();
    highMap.lookup(phases.front().publication.getOperation())->erase();
    auto orderPhase = [&](IRMapping &map, Phase &p, bool readFirst) {
      SmallVector<ttg::LocalLoadOp> ordered = p.reads;
      llvm::stable_sort(ordered, [](auto a, auto b) {
        return !isMatrixRead(a) && isMatrixRead(b);
      });
      Operation *anchor = readFirst
                              ? map.lookup(p.copies.front().getOperation())
                              : map.lookup(p.wait.getOperation());
      // Both copies and offset operands precede their phase's wait. Moving
      // only independent reads to this boundary keeps every definition valid.
      OpBuilder fence(anchor);
      ROCDL::SchedBarrier::create(fence, loop.getLoc(), 0);
      for (auto read : ordered)
        map.lookup(read.getOperation())->moveBefore(anchor);
      fence.setInsertionPoint(anchor);
      ROCDL::SchedBarrier::create(fence, loop.getLoc(), 0);
    };
    for (auto [i, p] : llvm::enumerate(phases)) {
      orderPhase(lowMap, p, false);
      orderPhase(highMap, p, i == 0);
    }
    b.setInsertionPointAfter(high);
    scf::YieldOp::create(b, loop.getLoc(), high.getResults());
    b.setInsertionPointAfter(groups);
    ttg::AsyncWaitOp::create(b, loop.getLoc(), ValueRange{}, 0);
    mlir::gpu::BarrierOp::create(b, loop.getLoc());
    entry.erase();
    exits.front().erase();
    llvm::DenseSet<Value> wideSlots;
    for (auto &p : phases)
      for (auto copy : p.copies) {
        auto placement = cast<ttg::HCUBufferLdsSharedEncodingAttr>(
            copy.getDest().getType().getEncoding());
        if (placement.getContiguousDim() == 1 &&
            copy.getDest().getType().getShape()[1] *
                    placement.getElementBytes() >
                256)
          wideSlots.insert(staticSlot(copy.getDest()));
      }
    loop.replaceAllUsesWith(groups.getResults());
    loop.erase();
    // A wide non-K-contiguous producer already spans multiple cache sets.
    // Keep the narrow K-contiguous operand's descriptor policy unchanged.
    module.walk([&](triton::amdgpu::BufferLoadToLocalOp copy) {
      if (wideSlots.contains(staticSlot(copy.getDest())))
        copy.getStrideMutable().clear();
    });
    transformed = true;
  }
  return success(transformed);
}

} // namespace mlir::triton::HCU::planned

namespace mlir::triton::HCU {

static void configureScheduledCopyCache(ModuleOp module) {
  llvm::DenseSet<Operation *> ordinaryFunctions;
  llvm::DenseSet<Value> unprovedCacheResources;
  module.walk([&](scf::ForOp loop) {
    if (loop->hasAttr("hcu.block_pingpong.applied"))
      ordinaryFunctions.insert(loop->getParentOfType<triton::FuncOp>());
  });
  module.walk([&](triton::amdgpu::BufferLoadToLocalOp copy) {
    Operation *function = copy->getParentOfType<triton::FuncOp>();
    if (ordinaryFunctions.contains(function) &&
        planned::requiresWrappedProducer(copy.getDest().getType()) &&
        copy.getStride() &&
        !copy.getStride().getDefiningOp<arith::ConstantOp>())
      unprovedCacheResources.insert(copy.getPtr());
  });
  module.walk([&](triton::amdgpu::BufferLoadToLocalOp copy) {
    auto placement = dyn_cast<ttg::HCUBufferLdsSharedEncodingAttr>(
        copy.getDest().getType().getEncoding());
    // Ordinary expansion can infer a dynamic row pitch in the prologue but
    // use a constant cache stride for loop-carried producers. Such a dynamic
    // hint is not proved encodable and must not override copy correctness.
    // Copy conversion proves the resource pointer invariant across entry and
    // backedge. Keep all generations of that resource on the same policy,
    // without changing unrelated operands or loops in the same function.
    // TODO: normalize and propagate dynamic cache strides in the common
    // buffer-descriptor pass, so every generation uses one proven policy.
    bool unprovedDynamicCacheStride =
        unprovedCacheResources.contains(copy.getPtr());
    if (placement && unprovedDynamicCacheStride)
      copy.getStrideMutable().clear();
  });
}
} // namespace mlir::triton::HCU

namespace mlir {
#define GEN_PASS_DEF_TRITONHCUSCHEDULEBUFFERTOLDS
#define GEN_PASS_DEF_TRITONHCUSYNCHRONIZEBUFFERLDSCOPIES
#include "TritonHCU/Passes.h.inc"
using namespace triton::HCU::planned;

// Standalone synchronization remains a separate pass from copy conversion.
// Pipeline users instead let ScheduleBufferToLds own publication and reuse.
struct TritonHCUSynchronizeBufferLdsCopiesPass
    : impl::TritonHCUSynchronizeBufferLdsCopiesBase<
          TritonHCUSynchronizeBufferLdsCopiesPass> {
  using Base::Base;

  void runOnOperation() override {
    // Snapshot the producers before inserting synchronization. Native generic
    // async copies keep their existing synchronization path unchanged.
    SmallVector<triton::amdgpu::BufferLoadToLocalOp> copies;
    getOperation().walk([&](triton::amdgpu::BufferLoadToLocalOp copy) {
      if (isa<triton::gpu::HCUBufferLdsSharedEncodingAttr>(
              copy.getDest().getType().getEncoding()))
        copies.push_back(copy);
    });
    for (auto copy : copies) {
      OpBuilder builder(copy);
      builder.setInsertionPointAfter(copy);
      auto commit = triton::gpu::AsyncCommitGroupOp::create(
          builder, copy.getLoc(), ValueRange{copy.getResult()});
      triton::gpu::AsyncWaitOp::create(
          builder, copy.getLoc(), ValueRange{commit.getResult()}, 0);
      if (!copy->getParentOfType<triton::gpu::WarpSpecializeOp>())
        triton::gpu::LocalBarrierOp::create(builder, copy.getLoc());
    }
  }
};

struct TritonHCUScheduleBufferToLdsPass
    : impl::TritonHCUScheduleBufferToLdsBase<TritonHCUScheduleBufferToLdsPass> {
  using Base::Base;
  void runOnOperation() override {
    triton::HCU::configureScheduledCopyCache(getOperation());
    scheduleStaticCopiesBeforeReads(getOperation());

    // Late producer conversion runs after slot rotation and pingpong
    // scheduling. Commit each copy here, after its final placement is known.
    // A copy is one group even when lowering emits multiple DMA instructions.
    // Publication waits count these groups, never individual VMEM events.
    SmallVector<triton::amdgpu::BufferLoadToLocalOp> asyncCopies;
    getOperation().walk([&](triton::amdgpu::BufferLoadToLocalOp copy) {
      asyncCopies.push_back(copy);
    });
    for (auto copy : asyncCopies) {
      OpBuilder builder(copy);
      builder.setInsertionPointAfter(copy);
      ttg::AsyncCommitGroupOp::create(builder, copy.getLoc(),
                                      ValueRange{copy.getResult()});
    }
    resolvePublicationWaits(getOperation());
    // A partial publication barrier only retires the generation required by
    // the next steady-state read, not every slot in the ring. In particular,
    // the epilogue consumes several remaining slots without issuing another
    // producer: its first read must drain the outstanding groups explicitly.
    // Generic membar analysis treats a barrier as publishing all writes and
    // cannot infer this async completion requirement from slot aliasing.
    SmallVector<ttg::LocalLoadOp> drainReads;
    getOperation().walk([&](ttg::LocalLoadOp load) {
      auto synced =
          load->getAttrOfType<BoolAttr>("ttg.amdg.syncedViaAsyncWait");
      if (requiresWrappedProducer(load.getSrc().getType()) &&
          !(synced && synced.getValue()))
        drainReads.push_back(load);
    });
    for (auto load : drainReads) {
      OpBuilder builder(load);
      ttg::AsyncWaitOp::create(builder, load.getLoc(), ValueRange{},
                               entryPrefetchPendingGroups(load));
      ttg::LocalBarrierOp::create(builder, load.getLoc());
    }
    if (!asyncCopies.empty())
      getOperation().walk([&](triton::ReturnOp ret) {
        OpBuilder builder(ret);
        ttg::AsyncWaitOp::create(builder, ret.getLoc(), ValueRange{}, 0);
      });
    if (ldsPrefetchDistance == 1)
      (void)staggerBufferLdsWaveGroups(getOperation());
  }
};

} // namespace mlir
