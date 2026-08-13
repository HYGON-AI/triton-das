#include "TritonHCU/Passes.h"
#include "TritonHCU/WdraSplitPlan.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonhcu-consumer-pingpong"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: " << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace hcu = mlir::triton::HCU;

namespace mlir {
#define GEN_PASS_DEF_TRITONHCUCONSUMERPINGPONG
#include "TritonHCU/Passes.h.inc"
} // namespace mlir

namespace {

// Shared execution barrier used as AMD BlockPingpong's gpu.barrier +
// cond_barrier. ConvertWarpSpecializeToLLVM rewrites ops with this attr to
// one ebarrier ID counted across both MMA consumer groups.
static void insertClusterBarrier(OpBuilder &b, Location loc) {
  ROCDL::SchedBarrier::create(b, loc, /*mask=*/0);
  auto bar = ROCDL::SBarrierOp::create(b, loc);
  bar->setAttr(hcu::kConsumerPingpongBarrierAttrName, b.getUnitAttr());
}

static bool isPingpongBarrier(Operation *op) {
  return op && isa<ROCDL::SBarrierOp>(op) &&
         op->hasAttr(hcu::kConsumerPingpongBarrierAttrName);
}

// Transform one consumer K-loop into Memory |BAR| Dot |BAR| clusters and
// wrap the dot with s_setprio so the other consumer cannot interrupt the
// matrix pipe. Requires the dot to be a direct child of the for-body.
static LogicalResult pingpongConsumerLoop(scf::ForOp forOp, bool isTail,
                                          bool isMain) {
  SmallVector<Operation *> dots;
  for (Operation &op : *forOp.getBody()) {
    if (isa<DotOpInterface>(&op))
      dots.push_back(&op);
  }
  if (dots.size() != 1) {
    LDBG("skip loop: expected exactly 1 direct-body dot, found "
         << dots.size());
    return success();
  }
  Operation *dot = dots.front();
  if (isPingpongBarrier(dot->getPrevNode()))
    return success();

  OpBuilder b(forOp);
  Location loc = dot->getLoc();

  // MMA_tail waits here so its first in-loop barrier meets MMA_main's first
  // cluster barrier — same trick as BlockPingpong::addAsymmetricSyncToLoop.
  if (isTail) {
    b.setInsertionPoint(forOp);
    insertClusterBarrier(b, forOp.getLoc());
  }

  b.setInsertionPoint(dot);
  insertClusterBarrier(b, loc);
  ROCDL::SetPrioOp::create(b, loc, /*priority=*/1);

  b.setInsertionPointAfter(dot);
  ROCDL::SetPrioOp::create(b, loc, /*priority=*/0);
  insertClusterBarrier(b, loc);

  if (isMain) {
    b.setInsertionPointAfter(forOp);
    insertClusterBarrier(b, forOp.getLoc());
  }

  LDBG("staggered consumer loop (tail=" << isTail << ", main=" << isMain
                                        << ")");
  return success();
}

static LogicalResult
pingpongMmaPartition(Region *partition, bool isTail, bool isMain) {
  SmallVector<scf::ForOp> loops;
  partition->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    if (failed(pingpongConsumerLoop(forOp, isTail, isMain)))
      return failure();
  }
  return success();
}

} // namespace

namespace mlir {

class ConsumerPingpongPass
    : public impl::TritonHCUConsumerPingpongBase<ConsumerPingpongPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto topo = hcu::getWdraTopologyAttr(mod);
    if (topo.kind != hcu::WdraTopoKind::OnePTwoC &&
        topo.kind != hcu::WdraTopoKind::TwoPTwoC) {
      LDBG("skip: not a two-consumer WDRA topology");
      return;
    }

    WalkResult result = mod.walk([&](WarpSpecializeOp wsOp) {
      for (auto [idx, partition] :
           llvm::enumerate(wsOp.getPartitionRegions())) {
        if (!topo.isMmaPartition(idx))
          continue;
        if (failed(pingpongMmaPartition(partition, topo.isMmaTailPartition(idx),
                                        topo.isMmaMainPartition(idx))))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace mlir
