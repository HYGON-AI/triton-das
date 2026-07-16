#include "TritonHCU/WdraSplitPlan.h"

#include "TritonHCU/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUPREPAREWDRASPLITPLAN
#include "TritonHCU/Passes.h.inc"

class PrepareWdraSplitPlanPass
    : public impl::TritonHCUPrepareWdraSplitPlanBase<
          PrepareWdraSplitPlanPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp loop) {
      // This pass runs before loop fusion/canonicalization, which can create
      // or move the eventual tt.warp_specialize loop. With WDRA enabled the
      // enclosing dot loop is the stable anchor; later passes retain both its
      // policy and the dot-specific plan through cloning.
      bool hasPlan = false;
      loop.getBody()->walk([&](DotOpInterface dot) {
        if (auto plan = HCU::inferWdraSplitPlan(dot, numConsumerGroups)) {
          HCU::setWdraSplitPlan(dot, *plan);
          hasPlan = true;
        }
      });
      if (hasPlan)
        HCU::setWdraSplitPolicy(loop, numConsumerGroups);
    });
  }
};

} // namespace mlir
