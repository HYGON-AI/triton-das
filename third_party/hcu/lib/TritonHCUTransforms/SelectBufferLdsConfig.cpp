// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "TritonHCU/BufferLdsConfigSelection.h"
#include "TritonHCU/BufferLdsEncoding.h"
#include "TritonHCU/Passes.h"
#include "TritonHCU/WaspSplitPlan.h"
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
    auto topology = tt::HCU::getWaspTopologyFromModule(dot);
    auto plan = tt::HCU::getWaspSplitPlan(dot);
    if (topology && topology->producerSliced() &&
        (!plan || plan->factor != topology->numConsumerGroups()))
      return rewriter.notifyMatchFailure(
          dot, "missing matching WASP producer split plan");

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
      unsigned maxCopyBytes = 0;
      auto sharedEncoding = tt::HCU::selectBufferLdsOperandEncoding(
          source, sourceType, consumer, std::nullopt,
          operandType.getElementType(), waves, arch, &maxCopyBytes);
      if (!sharedEncoding)
        continue;

      unsigned kDim = consumer.getOpIdx() == 0 ? 1 : 0;
      auto instr = mfma.getInstrShapeForOperand(consumer.getKWidth(),
                                                consumer.getOpIdx());
      bool matrixRead = mfma.getTilesPerWarp()[1 - kDim] == 2 &&
                        mfma.getTilesPerWarp()[kDim] == 1;
      if (topology && topology->producerSliced()) {
        SmallVector<int64_t> producerShape(sourceType.getShape());
        unsigned nonKDim = 1 - kDim;
        if (plan->partitionedOperand == index) {
          if (producerShape[nonKDim] <= 0 ||
              producerShape[nonKDim] % plan->factor)
            continue;
          producerShape[nonKDim] /= plan->factor;
        }

        auto profile = tt::HCU::getBufferLdsTargetProfile(
            std::string_view(arch.data(), arch.size()));
        assert(profile && instr.size() == 2 && maxCopyBytes &&
               "selected BufferLds operand must retain its search context");
        unsigned contiguousDim = sharedEncoding.getContiguousDim();
        auto producerConfig = tt::HCU::selectMmacBufferLdsConfig(
            producerShape[1 - contiguousDim], producerShape[contiguousDim],
            sharedEncoding.getElementBytes(), topology->getPartitionWarps(0),
            /*copyBytes=*/0,
            /*kContiguous=*/contiguousDim == kDim, instr[1 - kDim],
            consumer.getKWidth(), profile->lds, matrixRead, maxCopyBytes,
            profile->wrap, std::string_view(arch.data(), arch.size()));
        if (!producerConfig)
          continue;

        Value loadSource = source;
        while (auto convert = loadSource.getDefiningOp<ttg::ConvertLayoutOp>())
          loadSource = convert.getSrc();
        auto load = loadSource.getDefiningOp<tt::LoadOp>();
        assert(load && "selected BufferLds operand must come from tt.load");
        load->setAttr(tt::HCU::kBufferLdsProducerConfigAttrName,
                      tt::HCU::encodeBufferLdsConfig(rewriter.getContext(),
                                                     *producerConfig));
      }

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
