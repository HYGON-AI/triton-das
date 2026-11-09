#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Transforms/InliningUtils.h"
using namespace mlir;
// Gluon emits helper calls before lowering; these explicit synchronization
// primitives may be inlined into the same worker region without changing scope.
struct HcuGluonInlinerInterface : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *op, Region *, bool, IRMapping &) const final {
    return isa<ROCDL::HCUEbarrierArriveOp, ROCDL::HCUEbarrierSyncOp,
               ROCDL::HCUAbarrierInitOp, ROCDL::HCUAbarrierInvOp,
               ROCDL::HCUAbarrierTryWaitOp, ROCDL::HCUAbarrierTestWaitOp,
               ROCDL::HCUAbarrierSeqOp,
               ROCDL::HCUAbarrierArriveOp, ROCDL::SchedBarrier,
               ROCDL::SchedGroupBarrier, ROCDL::SetPrioOp>(op);
  }
};
void enableHcuGluonInlining(MLIRContext *ctx) {
  auto *dialect = ctx->getOrLoadDialect<ROCDL::ROCDLDialect>();
  if (!dialect->getRegisteredInterface<DialectInlinerInterface>())
    dialect->addInterfaces<HcuGluonInlinerInterface>();
}
