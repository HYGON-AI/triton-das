// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Rewrite eligible tt.descriptor_load operations to tt.matrix_load in TTIR,
// before the generic descriptor-to-pointer pass.
//
// Enabled by default on gfx946/gfx948 and controlled by
// TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS. Convert 2-D zero-padded loads after
// proving address alignment/ranges and checking MLS tile compatibility for all
// dot consumers. Loads without a dot consumer keep the generic pointer path.
// A sole transpose user can be absorbed by swapping address metadata and
// boundary-check axes. Existing MLS passes select layouts/instructions and
// implement boundary padding.
//
// Descriptor stores, unsupported loads, and modules with explicit matrix stores
// retain the generic descriptor-to-pointer lowering.

#include "Analysis/AxisInfoExt.h"
#include "Analysis/RangeAnalysis.h"
#include "TritonHCU/MlsGroup.h"
#include "TritonHCU/Passes.h"
#include "TritonHCU/TargetUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <limits>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
#define GEN_PASS_DEF_TRITONHCUREWRITETENSORDESCRIPTORTOMLS
#include "TritonHCU/Passes.h.inc"

namespace {

bool isDescriptorMlsEnabled(llvm::StringRef arch) {
  if (arch != "gfx946" && arch != "gfx948")
    return false;
  std::string value =
      tt::tools::getStrEnv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS");
  return tt::tools::isEnvValueBool(value).value_or(true);
}

bool isDescriptorLoadCandidate(tt::DescriptorLoadOp load) {
  auto desc = load.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
  if (!desc || load.getType().getRank() != 2 ||
      desc.getType().getBlockType().getRank() != 2 ||
      desc.getPadding() != tt::PaddingOption::PAD_ZERO ||
      load.getCache() != tt::CacheModifier::NONE ||
      load.getEvict() != tt::EvictionPolicy::NORMAL)
    return false;
  Type elem = load.getType().getElementType();
  return elem.isF16() || elem.isBF16() || elem.isInteger(8) ||
         isa<Float8E4M3FNType, Float8E5M2Type>(elem);
}

bool isInRange(const DataFlowSolver &solver, Value value, int64_t min,
               int64_t max) {
  auto ranges = tt::AMD::collectRanges(solver, ValueRange{value});
  return ranges && (*ranges)[0] && (*ranges)[0]->smin().getSExtValue() >= min &&
         (*ranges)[0]->smax().getSExtValue() <= max;
}

bool isAligned(tt::AMD::ModuleAxisInfoAnalysis &axis, Value value,
               int64_t alignment) {
  auto *info = axis.getAxisInfo(value);
  return info && info->getDivisibility(0) % alignment == 0;
}

// Prove alignment, non-negative indices and safe i32 tile endpoints for MLS.
bool isMlsAddressCompatible(tt::MakeTensorDescOp desc, ValueRange indices,
                            const DataFlowSolver &solver,
                            tt::AMD::ModuleAxisInfoAnalysis &axis) {
  auto blockTy = desc.getType().getBlockType();
  if (blockTy.getRank() != 2 || indices.size() != 2 ||
      !matchPattern(desc.getStrides()[1], m_One()))
    return false;
  unsigned bytes = blockTy.getElementType().getIntOrFloatBitWidth() / 8;
  if ((bytes != 1 && bytes != 2) || !isAligned(axis, desc.getBase(), 16) ||
      !isAligned(axis, desc.getStrides()[0], 16 / bytes) ||
      !isAligned(axis, indices[1], 16 / bytes) ||
      !isInRange(solver, desc.getStrides()[0], 1,
                 std::numeric_limits<uint32_t>::max()))
    return false;
  for (unsigned d = 0; d < 2; ++d)
    if (!isInRange(solver, desc.getShape()[d], 0,
                   std::numeric_limits<int32_t>::max()) ||
        !isInRange(solver, indices[d], 0,
                   std::numeric_limits<int32_t>::max() - blockTy.getShape()[d]))
      return false;
  return true;
}

// Collect every dependent dot operand, including uses outside yielding regions.
// Failure means that the consumer path cannot be established safely.
FailureOr<SmallVector<OpOperand *>> collectDotOperands(Value value) {
  SetVector<Operation *> slice;
  getForwardSlice(value, &slice);
  SmallVector<OpOperand *> operands;
  for (unsigned i = 0; i < slice.size(); ++i) {
    Operation *op = slice[i];
    if (isa<tt::DotOpInterface>(op)) {
      auto count = operands.size();
      for (auto &operand : op->getOpOperands())
        if (operand.get() == value ||
            slice.contains(operand.get().getDefiningOp()))
          operands.push_back(&operand);
      if (operands.size() == count)
        return failure();
    } else if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      // A while yield forwards to the before-region arguments, which need not
      // match the operation's results. Do not map it as a for/if yield.
      if (isa<scf::WhileOp>(yield->getParentOp()))
        return failure();
      for (auto [idx, operand] : llvm::enumerate(yield.getOperands())) {
        if (operand != value && !slice.contains(operand.getDefiningOp()))
          continue;
        SetVector<Operation *> outer;
        getForwardSlice(yield->getParentOp()->getResult(idx), &outer);
        slice.insert(outer.begin(), outer.end());
      }
    } else if (isa<scf::ConditionOp>(op)) {
      return failure();
    }
  }
  return operands;
}

bool isDotOperandCompatible(Value value, tt::DotOpInterface dot, unsigned opIdx,
                            bool transpose, unsigned mlsVersion) {
  auto type = cast<RankedTensorType>(value.getType());
  Type elem = type.getElementType();
  unsigned bits = elem.getIntOrFloatBitWidth();
  // Packed/sub-byte scaled-dot formats need more than the descriptor's element
  // type to establish their physical MLS tile. Retain the generic path here.
  if (!isa<tt::DotOp>(dot.getOperation()) || opIdx > 1)
    return false;
  auto resultTy = cast<RankedTensorType>(dot->getResult(0).getType());
  // The existing accelerator uses 16x16 MMAC instructions. A software dot
  // cannot consume an MLS dot-operand layout.
  if (resultTy.getRank() != 2 || resultTy.getShape()[0] % 16 ||
      resultTy.getShape()[1] % 16)
    return false;
  Value operand = opIdx == 0 ? dot.getA() : dot.getB();
  if (cast<RankedTensorType>(operand.getType()).getElementType() != elem)
    return false;
  while (operand != value) {
    Operation *def = operand.getDefiningOp();
    if (!def || !isa<arith::ExtFOp, arith::TruncFOp, tt::FpToFpOp>(def))
      return false;
    operand = def->getOperand(0);
  }
  bool kMajor = (1 - opIdx) == (transpose ? 0 : 1);
  auto candidates = tt::HCU::MlsInsn::getMlsElemsTileCandidates(
      opIdx, kMajor, bits, ttg::MlsElemBitTyKind::None, mlsVersion);
  int64_t nonK = type.getShape()[opIdx];
  int64_t k = type.getShape()[1 - opIdx];
  return llvm::any_of(candidates, [&](auto tile) {
    return nonK % tile.first == 0 && k % tile.second == 0;
  });
}

bool isMlsLoadCompatible(Value value, bool transpose, unsigned mlsVersion) {
  auto operands = collectDotOperands(value);
  if (failed(operands) || operands->empty())
    return false;

  // Acceleration can change which dot the existing MLS consumer query finds.
  // Each possible consumer must satisfy the MLS tile contract.
  return llvm::all_of(*operands, [&](OpOperand *operand) {
    return isDotOperandCompatible(
        value, cast<tt::DotOpInterface>(operand->getOwner()),
        operand->getOperandNumber(), transpose, mlsVersion);
  });
}

SmallVector<int32_t> getBoundaryChecks(tt::MakeTensorDescOp desc,
                                     ValueRange indices,
                                     const DataFlowSolver &solver) {
  auto shapes = tt::AMD::collectRanges(solver, desc.getShape());
  auto offsets = tt::AMD::collectRanges(solver, indices);
  auto tile = desc.getType().getBlockType().getShape();
  SmallVector<int32_t> checks;
  for (unsigned d = 0; d < 2; ++d) {
    bool inBounds = shapes && offsets && (*shapes)[d] && (*offsets)[d] &&
                    (*offsets)[d]->smax().getSExtValue() <=
                        (*shapes)[d]->smin().getSExtValue() - tile[d];
    if (!inBounds)
      checks.push_back(d);
  }
  return checks;
}

// Rewrite a validated load, optionally absorbing its single transpose user.
void rewriteDescriptorLoad(IRRewriter &rewriter, tt::DescriptorLoadOp load,
                           tt::TransOp transpose, const DataFlowSolver &solver) {
  auto desc = load.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
  auto type = transpose ? transpose.getType() : load.getType();
  auto shape = llvm::to_vector_of<int32_t>(type.getShape());
  auto checks = getBoundaryChecks(desc, load.getIndices(), solver);
  auto matrixShape = llvm::to_vector(desc.getShape());
  auto matrixStrides = llvm::to_vector(desc.getStrides());
  auto matrixIndices = llvm::to_vector(load.getIndices());
  if (transpose) {
    std::swap(matrixShape[0], matrixShape[1]);
    std::swap(matrixStrides[0], matrixStrides[1]);
    std::swap(matrixIndices[0], matrixIndices[1]);
    for (auto &dim : checks)
      dim = 1 - dim;
    llvm::sort(checks);
  }

  // Keep the read at its original position across intervening memory writes.
  rewriter.setInsertionPoint(load);
  auto matrix = tt::MatrixLoadOp::create(
      rewriter, load.getLoc(), type, desc.getBase(), matrixShape, matrixStrides,
      rewriter.getDenseI32ArrayAttr(shape), matrixIndices,
      rewriter.getDenseI32ArrayAttr(checks), load.getCache(), load.getEvict(),
      false);

  // Value hints must describe the result's logical axes after any transpose.
  Operation *resultOp = transpose ? transpose.getOperation() : load.getOperation();
  for (auto attr : resultOp->getAttrs())
    if (attr.getName().getValue() != "order" &&
        attr.getName().getValue() != "operandSegmentSizes")
      matrix->setAttr(attr.getName(), attr.getValue());
  rewriter.replaceOp(resultOp, matrix->getResults());
  if (transpose)
    rewriter.eraseOp(load);
}

} // namespace

class TritonHCURewriteTensorDescriptorToMLSPass
    : public impl::TritonHCURewriteTensorDescriptorToMLSBase<
          TritonHCURewriteTensorDescriptorToMLSPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (!isDescriptorMlsEnabled(arch))
      return;
    auto version =
        tt::HCU::getMlsVersionFromFeatures(tt::HCU::deduceHCUISAFeature(arch));
    ModuleOp mod = getOperation();
    SmallVector<tt::DescriptorLoadOp> loads;
    mod.walk([&](tt::DescriptorLoadOp load) {
      if (isDescriptorLoadCandidate(load))
        loads.push_back(load);
    });
    if (loads.empty())
      return;
    // Explicit stores may impose function-wide B interleave requirements.
    // Keep descriptors on their original path in mixed kernels.
    if (mod.walk([](tt::MatrixStoreOp) { return WalkResult::interrupt(); })
            .wasInterrupted())
      return;

    auto solver = createDataFlowSolver();
    auto assumptions =
        tt::AMD::TritonIntegerRangeAnalysis::collectAssumptions(mod);
    auto *ranges = solver->load<tt::AMD::TritonIntegerRangeAnalysis>(
        assumptions, &getAnalysis<DominanceInfo>());
    tt::AMD::initializeFuncOps(mod, ranges);
    if (failed(solver->initializeAndRun(mod)))
      return;
    tt::AMD::ModuleAxisInfoAnalysis axis(mod);
    IRRewriter rewriter(&getContext());
    for (auto load : loads) {
      auto desc = load.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
      if (!isMlsAddressCompatible(desc, load.getIndices(), *solver, axis))
        continue;
      tt::TransOp transpose;
      Value value = load.getResult();
      if (value.hasOneUse()) {
        auto user = dyn_cast<tt::TransOp>(*value.getUsers().begin());
        if (user && user.getOrder() == ArrayRef<int32_t>{1, 0}) {
          transpose = user;
          value = user.getResult();
        }
      }
      if (!isMlsLoadCompatible(value, bool(transpose), version))
        continue;
      rewriteDescriptorLoad(rewriter, load, transpose, *solver);
    }
  }
};

} // namespace mlir
