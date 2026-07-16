#include "TritonHCU/WdraSplitPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::HCU {

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
