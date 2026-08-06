#include "TritonHCU/WdraSplitPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::HCU {

WdraTopology deriveWdraTopology(bool wdraEnabled, int waspNumLoadWarps,
                                int waspNumMmaWarps) {
  WdraTopology topo;
  if (!wdraEnabled)
    return topo;
  topo.warpsPerPartition = 4;
  if (waspNumLoadWarps == 8 && waspNumMmaWarps == 8) {
    topo.kind = WdraTopoKind::TwoPTwoC;
    topo.numConsumerGroups = 2;
    topo.numLoadGroups = 2;
    return topo;
  }
  if (waspNumLoadWarps == 4 && waspNumMmaWarps == 8) {
    topo.kind = WdraTopoKind::OnePTwoC;
    topo.numConsumerGroups = 2;
    topo.numLoadGroups = 1;
    return topo;
  }
  // wdra + 4+4: single MMA partition, no DataPartition split.
  return topo;
}

void setWdraTopologyAttr(ModuleOp mod, WdraTopology topo) {
  mod->setAttr(kWdraTopoAttrName,
               IntegerAttr::get(IntegerType::get(mod.getContext(), 32),
                                static_cast<int32_t>(topo.kind)));
}

static void setModuleBoolAttr(ModuleOp mod, StringRef name, bool enabled) {
  mod->setAttr(name, BoolAttr::get(mod.getContext(), enabled));
}

static bool getModuleBoolAttr(Operation *op, StringRef name, bool defaultVal) {
  ModuleOp mod = dyn_cast<ModuleOp>(op);
  if (!mod)
    mod = op->getParentOfType<ModuleOp>();
  if (!mod)
    return defaultVal;
  if (auto attr = mod->getAttrOfType<BoolAttr>(name))
    return attr.getValue();
  if (auto attr = mod->getAttrOfType<IntegerAttr>(name))
    return attr.getInt() != 0;
  return defaultVal;
}

void setSchedBarrierBeforeMmacAttr(ModuleOp mod, bool enabled) {
  setModuleBoolAttr(mod, kSchedBarrierBeforeMmacAttrName, enabled);
}

void setEmptyArriveAfterMmacAttr(ModuleOp mod, bool enabled) {
  setModuleBoolAttr(mod, kEmptyArriveAfterMmacAttrName, enabled);
}

void setSchedBarrierBetweenABLoadsAttr(ModuleOp mod, bool enabled) {
  setModuleBoolAttr(mod, kSchedBarrierBetweenABLoadsAttrName, enabled);
}

bool getSchedBarrierBeforeMmacAttr(Operation *op, bool defaultVal) {
  return getModuleBoolAttr(op, kSchedBarrierBeforeMmacAttrName, defaultVal);
}

bool getEmptyArriveAfterMmacAttr(Operation *op, bool defaultVal) {
  return getModuleBoolAttr(op, kEmptyArriveAfterMmacAttrName, defaultVal);
}

bool getSchedBarrierBetweenABLoadsAttr(Operation *op, bool defaultVal) {
  return getModuleBoolAttr(op, kSchedBarrierBetweenABLoadsAttrName, defaultVal);
}

WdraTopology getWdraTopologyAttr(Operation *op) {
  ModuleOp mod = dyn_cast<ModuleOp>(op);
  if (!mod)
    mod = op->getParentOfType<ModuleOp>();
  if (!mod)
    return {};
  auto attr = mod->getAttrOfType<IntegerAttr>(kWdraTopoAttrName);
  if (!attr)
    return {};
  WdraTopology topo;
  topo.kind = static_cast<WdraTopoKind>(attr.getInt());
  topo.warpsPerPartition = 4;
  switch (topo.kind) {
  case WdraTopoKind::TwoPTwoC:
    topo.numConsumerGroups = 2;
    topo.numLoadGroups = 2;
    break;
  case WdraTopoKind::OnePTwoC:
    topo.numConsumerGroups = 2;
    topo.numLoadGroups = 1;
    break;
  default:
    break;
  }
  return topo;
}

std::optional<WdraSplitPlan>
inferWdraSplitPlan(DotOpInterface dot, unsigned factor) {
  if (factor < 2)
    return std::nullopt;

  auto resultTy = dyn_cast<RankedTensorType>(dot->getResult(0).getType());
  if (!resultTy)
    return std::nullopt;
  SmallVector<int64_t> shapePerCTA = gpu::getShapePerCTA(resultTy);
  if (shapePerCTA.size() != 2)
    return std::nullopt;

  // Keep the existing DataPartition priority: M/A first, then N/B. The full
  // slice-closure legality check remains in DataPartition, where it has the
  // complete transformed def-use graph.
  for (auto [dim, operand] : llvm::enumerate(ArrayRef<unsigned>{0, 1})) {
    int64_t extent = shapePerCTA[dim];
    if (extent >= static_cast<int64_t>(factor * 16) &&
        extent % static_cast<int64_t>(factor) == 0)
      return WdraSplitPlan{static_cast<unsigned>(dim), factor, operand};
  }
  return std::nullopt;
}

void setWdraSplitPolicy(Operation *loop, unsigned factor) {
  loop->setAttr(kWdraSplitPolicyAttrName,
                IntegerAttr::get(IntegerType::get(loop->getContext(), 32),
                                 factor));
}

bool hasWdraSplitPolicy(Operation *op) {
  return op && op->hasAttr(kWdraSplitPolicyAttrName);
}

void setWdraSplitPlan(Operation *dot, WdraSplitPlan plan) {
  dot->setAttr(kWdraSplitPlanAttrName,
               DenseI32ArrayAttr::get(dot->getContext(),
                                      {static_cast<int32_t>(plan.resultDim),
                                       static_cast<int32_t>(plan.factor),
                                       static_cast<int32_t>(
                                           plan.partitionedOperand)}));
}

std::optional<WdraSplitPlan> getWdraSplitPlan(Operation *dot) {
  auto attr =
      dot->getAttrOfType<DenseI32ArrayAttr>(kWdraSplitPlanAttrName);
  if (!attr || attr.size() != 3)
    return std::nullopt;
  auto values = attr.asArrayRef();
  if (values[0] > 1 || values[1] < 2 || values[2] > 1 ||
      values[2] != values[0])
    return std::nullopt;
  return WdraSplitPlan{static_cast<unsigned>(values[0]),
                       static_cast<unsigned>(values[1]),
                       static_cast<unsigned>(values[2])};
}

SmallVector<int64_t>
getWdraEffectiveResultShape(DotOpInterface dot,
                            const WdraSplitPlan &plan) {
  auto resultTy = cast<RankedTensorType>(dot->getResult(0).getType());
  SmallVector<int64_t> shape(resultTy.getShape());
  if (shape.size() >= 2) {
    unsigned dim = shape.size() - 2 + plan.resultDim;
    if (shape[dim] > 0 && shape[dim] % plan.factor == 0)
      shape[dim] /= plan.factor;
  }
  return shape;
}

} // namespace mlir::triton::HCU
