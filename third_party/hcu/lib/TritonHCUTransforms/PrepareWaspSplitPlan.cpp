#include "TritonHCU/WaspSplitPlan.h"

#include "TritonHCU/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUPREPAREWASPSPLITPLAN
#include "TritonHCU/Passes.h.inc"

class PrepareWaspSplitPlanPass
    : public impl::TritonHCUPrepareWaspSplitPlanBase<PrepareWaspSplitPlanPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned factor = numConsumerGroups;
    if (mod->hasAttr(HCU::kWaspPartitionWarpsAttrName)) {
      auto topo = HCU::getWaspTopologyFromModule(mod);
      if (!topo) {
        mod.emitError("invalid hcu.wasp_partition_warps attribute");
        return signalPassFailure();
      }
      factor = topo->numConsumerGroups();
    }
    if (factor == 1)
      return;
    // The user selected the topology explicitly. A small/unsplittable CTA
    // tile must fail before layout selection, never silently become 1P1C.
    if (mod->hasAttr(HCU::kWaspPartitionWarpsAttrName)) {
      WalkResult result = mod.walk([&](DotOpInterface dot) {
        bool specialized = false;
        for (Operation *parent = dot->getParentOp(); parent;
             parent = parent->getParentOp())
          specialized |= parent->hasAttr("tt.warp_specialize");
        if (specialized && !HCU::inferWaspSplitPlan(dot, factor)) {
          dot.emitError("wasp_partition_warps requests two Consumers but the CTA "
                        "tile cannot be split");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp loop) {
      // This pass runs before loop fusion/canonicalization, which can create
      // or move the eventual tt.warp_specialize loop. With split consumers the
      // enclosing dot loop is the stable anchor; later passes retain both its
      // policy and the dot-specific plan through cloning.
      bool hasPlan = false;
      loop.getBody()->walk([&](DotOpInterface dot) {
        if (auto plan = HCU::inferWaspSplitPlan(dot, factor)) {
          HCU::setWaspSplitPlan(dot, *plan);
          hasPlan = true;
        }
      });
      if (hasPlan)
        HCU::setWaspSplitPolicy(loop, factor);
    });
  }
};

} // namespace mlir
