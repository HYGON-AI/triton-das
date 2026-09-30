// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "TritonHCU/BufferLdsEncoding.h"
#include "TritonHCU/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
#define GEN_PASS_DEF_TRITONHCUSELECTBUFFERLDSCONFIG
#include "TritonHCU/Passes.h.inc"

namespace {

// Recover the value before the dot-operand conversion. AccelerateMatmul may
// also have selected the existing panel-major ds_read_m path; its initialized
// allocation is still an ordinary producer and can be replaced by a better
// proven shared encoding here.
static Value getBufferLdsSource(Value operand) {
  if (auto convert = operand.getDefiningOp<ttg::ConvertLayoutOp>())
    return convert.getSrc();
  if (auto load = operand.getDefiningOp<ttg::LocalLoadOp>()) {
    auto alloc = load.getSrc().getDefiningOp<ttg::LocalAllocOp>();
    if (!alloc || !alloc.getSrc() ||
        isa<ttg::HCUBufferLdsSharedEncodingAttr>(alloc.getType().getEncoding()))
      return {};
    return alloc.getSrc();
  }
  return {};
}

// BufferLds is currently materialized next to the dot operand, so selecting it
// also retimes the source load to that position. Reject the selection when the
// move would cross a global write; leaving the operand unchanged lets the
// existing generic async-copy pipeline handle the loop instead.
static bool mayWriteGlobalMemory(Operation *op) {
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    bool writes = false;
    op->walk([&](Operation *nested) {
      if (nested != op && mayWriteGlobalMemory(nested))
        writes = true;
    });
    return writes;
  }
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return !isMemoryEffectFree(op);
  SmallVector<MemoryEffects::EffectInstance> globalEffects;
  effects.getEffectsOnResource(tt::GlobalMemory::get(), globalEffects);
  return llvm::any_of(globalEffects, [](const auto &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  });
}

static bool canRetimeBufferLdsSource(Value source, Operation *destination) {
  while (auto convert = source.getDefiningOp<ttg::ConvertLayoutOp>())
    source = convert.getSrc();
  auto load = source.getDefiningOp<tt::LoadOp>();
  if (!load || load->getBlock() != destination->getBlock())
    return false;
  for (Operation *op = load->getNextNode(); op && op != destination;
       op = op->getNextNode())
    if (mayWriteGlobalMemory(op))
      return false;
  return load->isBeforeInBlock(destination);
}

struct AssignBufferLdsOperandEncoding : OpRewritePattern<tt::DotOp> {
  AssignBufferLdsOperandEncoding(MLIRContext *context, StringRef arch)
      : OpRewritePattern(context), arch(arch.str()) {}

  LogicalResult matchAndRewrite(tt::DotOp dot,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (unsigned index = 0; index < 2; ++index) {
      Value operand = dot->getOperand(index);
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType)
        continue;
      auto consumer =
          dyn_cast<ttg::DotOperandEncodingAttr>(operandType.getEncoding());
      auto mfma = consumer
                      ? dyn_cast<ttg::AMDMfmaEncodingAttr>(consumer.getParent())
                      : ttg::AMDMfmaEncodingAttr();
      if (!consumer || !mfma)
        continue;

      Value source = getBufferLdsSource(operand);
      auto sourceType = source ? dyn_cast<RankedTensorType>(source.getType())
                               : RankedTensorType();
      if (!sourceType ||
          sourceType.getElementType() != operandType.getElementType())
        continue;
      if (!canRetimeBufferLdsSource(source, dot))
        continue;

      unsigned waves = 1;
      for (unsigned count : mfma.getWarpsPerCTA())
        waves *= count;
      auto sharedEncoding = tt::HCU::selectBufferLdsOperandEncoding(
          source, sourceType, consumer, std::nullopt,
          operandType.getElementType(), waves, arch);
      if (!sharedEncoding)
        continue;

      Value convertedOperand = tt::HCU::materializeBufferLdsOperand(
          rewriter, dot.getLoc(), source, consumer, sharedEncoding);
      rewriter.modifyOpInPlace(
          dot, [&] { dot->setOperand(index, convertedOperand); });
      changed = true;
    }
    return success(changed);
  }

private:
  std::string arch;
};

} // namespace

struct TritonHCUSelectBufferLdsConfigPass
    : impl::TritonHCUSelectBufferLdsConfigBase<
          TritonHCUSelectBufferLdsConfigPass> {
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();
    auto target = module->getAttrOfType<StringAttr>("ttg.target");
    if (!target || !target.getValue().starts_with("hip:"))
      return;
    StringRef arch = target.getValue().drop_front(4);
    if (!tt::HCU::getBufferLdsTargetProfile(
            std::string_view(arch.data(), arch.size())))
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<AssignBufferLdsOperandEncoding>(&getContext(), arch);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir
