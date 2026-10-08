// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "TritonHCU/WaspSplitPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::HCU {

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

std::optional<WaspTopology> WaspTopology::get(DenseI32ArrayAttr partitionWarps) {
  if (!partitionWarps || partitionWarps.size() < 2 || partitionWarps.size() > 4)
    return std::nullopt;
  auto waves = partitionWarps.asArrayRef();
  if (llvm::any_of(waves, [](int n) {
        return n != 1 && n != 2 && n != 4 && n != 8;
      }))
    return std::nullopt;
  unsigned producers = waves.size() == 4 ? 2 : 1;
  unsigned total = 0;
  for (auto [i, n] : llvm::enumerate(waves)) {
    if (n != waves[i < producers ? 0 : producers])
      return std::nullopt;
    total += n;
  }
  if (total > 16 || waves[producers] < waves.front())
    return std::nullopt;
  return WaspTopology(partitionWarps);
}

std::optional<WaspTopology> getWaspTopologyFromModule(Operation *op) {
  ModuleOp mod = dyn_cast<ModuleOp>(op);
  if (!mod)
    mod = op->getParentOfType<ModuleOp>();
  if (!mod)
    return std::nullopt;
  return WaspTopology::get(
      mod->getAttrOfType<DenseI32ArrayAttr>(kWaspPartitionWarpsAttrName));
}

std::optional<WaspSplitPlan> inferWaspSplitPlan(DotOpInterface dot,
                                                unsigned factor) {
  if (factor < 2)
    return std::nullopt;

  auto resultTy = dyn_cast<RankedTensorType>(dot->getResult(0).getType());
  if (!resultTy)
    return std::nullopt;
  SmallVector<int64_t> shapePerCTA = resultTy.getEncoding()
                                       ? gpu::getShapePerCTA(resultTy)
                                       : llvm::to_vector(resultTy.getShape());
  if (shapePerCTA.size() != 2)
    return std::nullopt;

  // Keep the existing DataPartition priority: M/A first, then N/B. The full
  // slice-closure legality check remains in DataPartition, where it has the
  // complete transformed def-use graph.
  for (unsigned dim = 0; dim < 2; ++dim) {
    int64_t extent = shapePerCTA[dim];
    if (extent >= static_cast<int64_t>(factor * 16) &&
        extent % static_cast<int64_t>(factor) == 0)
      return WaspSplitPlan{dim, factor, dim};
  }
  return std::nullopt;
}

void setWaspSplitPolicy(Operation *loop, unsigned factor) {
  loop->setAttr(
      kWaspSplitPolicyAttrName,
      IntegerAttr::get(IntegerType::get(loop->getContext(), 32), factor));
}

bool hasWaspSplitPolicy(Operation *op) {
  return op && op->hasAttr(kWaspSplitPolicyAttrName);
}

void setWaspSplitPlan(Operation *dot, WaspSplitPlan plan) {
  dot->setAttr(
      kWaspSplitPlanAttrName,
      DenseI32ArrayAttr::get(dot->getContext(),
                             {static_cast<int32_t>(plan.resultDim),
                              static_cast<int32_t>(plan.factor),
                              static_cast<int32_t>(plan.partitionedOperand)}));
}

std::optional<WaspSplitPlan> getWaspSplitPlan(Operation *dot) {
  auto attr = dot->getAttrOfType<DenseI32ArrayAttr>(kWaspSplitPlanAttrName);
  if (!attr || attr.size() != 3)
    return std::nullopt;
  auto values = attr.asArrayRef();
  if (values[0] > 1 || values[1] < 2 || values[2] > 1 || values[2] != values[0])
    return std::nullopt;
  return WaspSplitPlan{static_cast<unsigned>(values[0]),
                       static_cast<unsigned>(values[1]),
                       static_cast<unsigned>(values[2])};
}

SmallVector<int64_t> getWaspEffectiveResultShape(DotOpInterface dot,
                                                 const WaspSplitPlan &plan) {
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
