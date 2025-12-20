#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using ::mlir::LLVM::AMD::getVectorSize;
using mlir::triton::AMD::ISAFamily;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOBUFFEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// ========================== HCU Utility Functions ==========================
void collectAssumptionsForFuncArgPtr(ModuleOp mod, DenseMap<Value, SetVector<Operation *>> &assumptions) {
  mod.walk([&](LLVM::AssumeOp op) {
    if (auto cmpIOp = op.getCond().getDefiningOp<arith::CmpIOp>()) {
      /*  HCU: add ConvertToBufferOp with ptr's offset are always non-negative support.
       *
       *     This is a trick method to annotate ptr's offset are always non-negative.
       *  CanonicalizeationPointersPass only propagate the ptr & ptr's user attributes, and this
       *  lead the attributes which tagged on the offsets maybe abandoned.
       *     So we need to put the assume info in ptr, not offsets.
       *     Besides, annotate_hint operations currently can only operate on op, not block arguments.
       *  and tl.assume operation is mlir & llvm native operation.
       *
       *  Eg, below code annotate all loaded offset calc from sorted_token_ids_ptr are non-negative.
       *    tl.assume(sorted_token_ids_ptr.to(tl.int64) >= 0)
       *    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
       *    .....
       *    a_ptrs = a_ptr + (offs_token[:, None]
       *
       *  The pattern in triton ir is(%arg6 is the sorted_token_ids_ptr):
       *
       *    %1 = tt.ptr_to_int %arg6 : !tt.ptr<i32> -> i64
       *    %2 = arith.cmpi sge, %1, %c0_i64 : i64
       *    llvm.intr.assume %2 : i1
       *
       *  Here we detect the pattern and add the %arg6(sorted_token_ids_ptr) to the assumption.
      **/
      bool isGe = (cmpIOp.getPredicate() == arith::CmpIPredicate::sge);
      if (isGe && cmpIOp->getOperand(0).getDefiningOp<triton::PtrToIntOp>()) {
        auto ptrToIntOp = cmpIOp->getOperand(0).getDefiningOp<triton::PtrToIntOp>();
        if (isa<BlockArgument>(ptrToIntOp->getOperand(0))
            && ptrToIntOp->getBlock()->isEntryBlock()) {
          assumptions[ptrToIntOp->getOperand(0)].insert(ptrToIntOp);
        }
      }
    }
  });
}

bool isFuncArgPtrWithNonNegativeAssumption(mlir::Value value,
  const DenseMap<Value, SetVector<Operation *>> &assumptions) {
  while (value.getDefiningOp() && isa<triton::AddPtrOp>(value.getDefiningOp())) {
    value = value.getDefiningOp<triton::AddPtrOp>().getPtr();
  }

  if (value.getDefiningOp())
    return false;

  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(value);
  auto blk = blockArg.getOwner();
  auto funcOp = dyn_cast_or_null<tt::FuncOp>(blk->getParentOp());

  if (funcOp && blk == &funcOp->getRegion(0).front()) {
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (arg != value)
        continue;
      return assumptions.contains(value);
    }
  }

  if (auto forOp = llvm::dyn_cast<scf::ForOp>(blk->getParentOp())) {
    if (blk == forOp.getBody()) {
      unsigned argNum = blockArg.getArgNumber();
      // scf.for body args: (iv, iter_args...)
      if (argNum == 0)
        return false;
      unsigned iterIdx = argNum - 1;
      auto initArgs = forOp.getInitArgs();
      if (iterIdx >= initArgs.size())
        return false;
      return isFuncArgPtrWithNonNegativeAssumption(initArgs[iterIdx],
                                                   assumptions);
    }
  }

  return false;
}

// ========================== HCU Utility Functions ==========================

bool verifyNonSmallerByAssumption(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions,
    const std::function<bool(Value)> &matchesOther) {
  if (!assumptions.contains(expr))
    return false;
  for (Operation *assume : assumptions.at(expr)) {
    auto cmpOp = llvm::dyn_cast<arith::CmpIOp>(assume);
    if (!cmpOp)
      continue;
    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::sgt: {
      if (cmpOp.getLhs() == expr && matchesOther(cmpOp.getRhs())) {
        LDBG("  " << expr << " non-neg by assumption " << cmpOp);
        return true;
      }
      break;
    }
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::slt: {
      if (cmpOp.getRhs() == expr && matchesOther(cmpOp.getLhs())) {
        LDBG("  " << expr << " non-neg by assumption " << cmpOp);
        return true;
      }
      break;
    }
    default:
      break;
    }
  }
  return false;
}

bool verifyNonSmallerByAssumption(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions,
    Value other) {
  return verifyNonSmallerByAssumption(
      expr, assumptions, [&](auto otherAssum) { return otherAssum == other; });
}

bool verifyNonNegativeExpr(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions,
    std::shared_ptr<DataFlowSolver> solver) {
  LDBG("Determing if non-negative: " << expr);

  auto nonNegativePred = [&solver](Value v) -> bool {
    if (const auto *r =
            solver->lookupState<dataflow::IntegerValueRangeLattice>(v)) {
      if (r->getValue().isUninitialized())
        return false;
      if (AMD::isEmptyInitializedRange(r->getValue().getValue()))
        return false;
    }
    return succeeded(dataflow::staticallyNonNegative(*solver, v));
  };

  if (nonNegativePred(expr))
    return true;

  // Recurse if the operation is defined
  Operation *op = expr.getDefiningOp();
  if (!op) {
    LDBG("  No defining op, assuming possibly negative");
    return false;
  }

  bool nonNegative =
      llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
          // Various unary triton ops that don't change the sign of the operand
          .Case<triton::TransOp, triton::SplitOp, triton::BroadcastOp,
                triton::ExpandDimsOp, triton::SplatOp, triton::ReshapeOp,
                triton::gpu::ConvertLayoutOp>([&](auto unaryOp) {
            return verifyNonNegativeExpr(unaryOp.getOperand(), assumptions,
                                         solver);
          })
          .Case<triton::GatherOp>([&](auto gatherOp) {
            return verifyNonNegativeExpr(gatherOp.getSrc(), assumptions,
                                         solver);
          })
          // Joining two non-negative tensors is still non-negative
          .Case<triton::JoinOp, triton::CatOp>([&](auto joinOp) {
            return verifyNonNegativeExpr(joinOp.getLhs(), assumptions,
                                         solver) &&
                   verifyNonNegativeExpr(joinOp.getRhs(), assumptions, solver);
          })
          // Returns a tensor representing histogram: histograms only contain
          // buckets of non-negative values.
          .Case<triton::HistogramOp>([&](auto) { return true; })
          .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
            // See the warning in TritonOps.td: getStart/getEnd return unsigned,
            // so we need to look through get*Attr.
            return makeRangeOp.getStartAttr().getInt() >= 0 &&
                   makeRangeOp.getEndAttr().getInt() >= 0;
          })
          .Case<arith::ConstantIntOp>(
              [&](auto constIntOp) { return constIntOp.value() >= 0; })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
            Value val = constOp.getResult();
            DenseIntElementsAttr constVal;
            if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
              return constVal.getSplatValue<APInt>().isNonNegative();
            return false;
          })
          .Case<triton::GetNumProgramsOp, triton::GetProgramIdOp>([&](auto) {
            // These are defined as signless, but are actually unsigned
            return true;
          })
          .Case<arith::MaxSIOp>([&](auto maxOp) {
            // max(a,b) >= 0 iff a>=0 || b>=0
            return verifyNonNegativeExpr(maxOp.getLhs(), assumptions, solver) ||
                   verifyNonNegativeExpr(maxOp.getRhs(), assumptions, solver);
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions, solver);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions,
                                         solver);
          })
          // Casting from arbitrary data does *not* guarantee the offset is in
          // range (even if pointer, or the data is non-negative when
          // interpreted as the src's type).
          .Case<triton::PtrToIntOp, triton::BitcastOp>(
              [&](auto) { return false; })
          .Case<arith::CeilDivUIOp, arith::DivUIOp, arith::ExtUIOp,
                arith::FPToUIOp, arith::MaxUIOp, arith::MinUIOp, arith::RemUIOp,
                arith::ShRUIOp>(
              // These OPs also return unsigned values.
              // TODO: We can also sniff whether a Value is unsigned by looking
              //       for whether or not it's used as an argument to one of
              //       these OPs.
              [&](auto uOp) { return true; })
          .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
              // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
              // OP != sub
              [&](Operation *binOp) {
                return verifyNonNegativeExpr(binOp->getOperand(0), assumptions,
                                             solver) &&
                       verifyNonNegativeExpr(binOp->getOperand(1), assumptions,
                                             solver);
              })
          // TODO: more scf
          .Case<scf::IfOp>([&](auto ifOp) {
            auto results = ifOp.getResults();
            auto it = std::find(results.begin(), results.end(), expr);
            assert(it != results.end() && "expr should be the result of ifOp");
            auto resultIdx = it - results.begin();

            // If we're here then we must have both then/else regions
            // (each with 1 block) and each region must terminate with an
            // `scf.yield` expression.
            auto thenYield = cast<scf::YieldOp>(ifOp.thenYield());
            auto elseYield = cast<scf::YieldOp>(ifOp.elseYield());
            return verifyNonNegativeExpr(thenYield->getOperand(resultIdx),
                                         assumptions, solver) &&
                   verifyNonNegativeExpr(elseYield->getOperand(resultIdx),
                                         assumptions, solver);
          })
          .Case<arith::SubIOp>([&](auto op) {
            // If a user annotates tl.assume(a >= b) then we know a - b >= 0
            return verifyNonSmallerByAssumption(op.getLhs(), assumptions,
                                                op.getRhs());
          })
          .Default([&](Operation *) {
            // HCU: if op is explicitly marked "non-negative", return true
            if (auto attr = op->template getAttrOfType<mlir::BoolAttr>("non-negative")) {
              bool nonNegative = attr.getValue();
              if (nonNegative) {
                return true;
              }
            }
            // Conservatively assume that the expression is negative
            LDBG("  Unhandled op, cannot assume non-negative");
            return false;
          });
  return nonNegative;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr,
                     const DenseMap<Value, SetVector<Operation *>> &assumptions,
                     std::shared_ptr<DataFlowSolver> solver) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition

  LDBG("Buffer op checks for: " << ptr);
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return false;

  auto maybeSplatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!maybeSplatOp)
    return false;
  LDBG("Pattern matched");

  // 2. Check if the offset is a 32-bit tensor
  Value offset = addPtrOp.getOffset();
  if (cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth() != 32)
    return false;
  LDBG("32 bit offset");

  return verifyNonNegativeExpr(offset, assumptions, std::move(solver)) ||
         isFuncArgPtrWithNonNegativeAssumption(maybeSplatOp.getSrc(), assumptions);
}

// Extract stride of the blocked offset of LD/ST ops.
Value getBlockStride(Location loc, Value offset, PatternRewriter &rewriter) {
  // canonicalize pointer pass sets block stride via
  // `offset:add-broadcast-muli-splat`, backtrace that pattern to reach the
  // stride.
  if (auto maybeAdd = offset.getDefiningOp<arith::AddIOp>())
    for (auto addOpr : maybeAdd.getOperands())
      if (auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>()) {
        auto bcSrc = maybeBC.getSrc();
        if (auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>())
          for (auto mulOpr : maybeMul.getOperands())
            if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>())
              return maybeSplat.getSrc();
      }
  return nullptr;
}

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto atomicRmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    // In addition to the `canUserBufferOps` check, we should ensure that
    // 1. Perform the canUserBufferOps check
    if (!canUseBufferOps(ptr, assumptions, solver)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    // 2. Check the scope. We support GPU and CTA for now (SYSTEM scope is not
    // supported yet)
    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "RMW with unsupported scope");
    }
    LDBG("RMW supported scope");

    // 3. Check the memory ordering.
    //    TODO: support monotonic
    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "RMW with unsupported memory ordering");
    }

    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value tensorOffset = addPtrOp.getOffset();
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // 4. Buffer atomic RMW does not support FP8 ops
    //    easier to just check what we support
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isF16() || checkType.isBF16() ||
                           checkType.isF32() || checkType.isF64() ||
                           checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "RMW with unsupported type");
    }
    LDBG("RMW supported type");

    auto vecSize = getVectorSize(ptr, axisAnalysisPass);
    // f16/bf16 dtypes could only be efficiently calculated using instructions
    // that pack 2 elements (e.g. @llvm.amdgcn.raw.buffer.atomic.fadd.v2f16)
    if (vecSize % 2 != 0 && (checkType.isF16() || checkType.isBF16())) {
      return rewriter.notifyMatchFailure(
          op, "RMW float 16 dtypes must be aligned by 2");
    }
    LDBG("RMW passed alignment check");

    // 5. Check if the RMWOp is supported
    switch (atomicRmwOp) {
    case RMWOp::AND:
    case RMWOp::OR:
    case RMWOp::XOR:
    case RMWOp::ADD:
    case RMWOp::FADD:
    case RMWOp::MAX:
    case RMWOp::MIN:
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    default:
      auto rmwOpStr = stringifyRMWOp(atomicRmwOp).str();
      return rewriter.notifyMatchFailure(op, "RMW with unsupported op: " +
                                                 rmwOpStr);
    }
    LDBG("RMW supported Op");

    // 6. Buffer atomics support 32 and 64-bit operations, so inputs must be at
    //    least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      // We can't just compute the opBitWidth using the numElements *
      // elemBitWidth here. In cases such as tensor<2xf16...>, if the elements
      // are contiguous we can emit the buffer op. Otherwise, the buffer ops
      // lowering will try to emit individual (unsupported) f16/bf16 ops.
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth = vecSize * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(op, "RMW requires opBitWidth >= 32");
    }

    Value maybeMask{};
    if (op.getMask() && !isZeroConst(op.getMask()))
      maybeMask = op.getMask();
    Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicRMWOp>(
        op, op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset,
        op.getVal(), blockStride, sem, scope, maybeMask);

    return success();
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
};

// Workaround to allow static_assert(false) on older compilers as it was
// ill-formed before defect report CWG2518
// (https://cplusplus.github.io/CWG/issues/2518.html)
template <typename T> struct always_false : std::false_type {};

template <typename SourceOp>
struct ConvertTritonLoadToBufferLoad : public mlir::OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      std::shared_ptr<DataFlowSolver> solver)
      : mlir::OpRewritePattern<SourceOp>(context), assumptions(assumptions),
        solver(std::move(solver)) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getOperand(0);

    if (canUseBufferOps(ptr, assumptions, solver)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isOneConst(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);

      auto bufferLoadOp = [&]() {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          return rewriter.create<triton::amdgpu::BufferLoadOp>(
              op->getLoc(), op.getType(), basePtr, tensorOffset, blockStride,
              op.getCache(), maybeMask, maybeOther);
        } else if constexpr (std::is_same_v<
                                 SourceOp,
                                 triton::gpu::AsyncCopyGlobalToLocalOp>) {
          return rewriter.create<triton::amdgpu::BufferLoadToLocalOp>(
              op->getLoc(), op.getType(), op.getResult(), basePtr, tensorOffset,
              maybeMask, maybeOther, blockStride, op.getCache());
        } else {
          static_assert(always_false<SourceOp>::value,
                        "Unsupported type in ConvertTritonLoadToBufferLoad");
        }
      }();

      assert(bufferLoadOp);

      // Propagate `OpIdxAttr` if the currently processed `tt.LoadOp` was
      // labeled it. The attribute needs to be preserved for custom instruction
      // scheduling.
      if (auto opIdxAttr =
              op->template getAttrOfType<triton::amdgpu::OpIdxAttr>(
                  triton::amdgpu::OpIdxAttr::getMnemonic())) {
        bufferLoadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(),
                              opIdxAttr);
      }
      rewriter.replaceOp(op, bufferLoadOp);
      return success();
    }

    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  std::shared_ptr<DataFlowSolver> solver;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      std::shared_ptr<DataFlowSolver> solver)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions), solver(std::move(solver)) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions, solver)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      if (op.getMask() && !isOneConst(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, blockStride, op.getCache(),
          maybeMask);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert StoreOp");
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  std::shared_ptr<DataFlowSolver> solver;
};

} // anonymous namespace

class TritonAMDGPUConvertToBufferOpsPass
    : public impl::TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {

public:
  using impl::TritonAMDGPUConvertToBufferOpsBase<
      TritonAMDGPUConvertToBufferOpsPass>::TritonAMDGPUConvertToBufferOpsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    // Collect assumptions in the function
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    collectAssumptionsForFuncArgPtr(mod, assumptions);
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(assumptions);
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad<tt::LoadOp>,
                 ConvertTritonLoadToBufferLoad<ttg::AsyncCopyGlobalToLocalOp>,
                 ConvertTritonStoreToBufferStore>(context, assumptions, solver);

    // Gate buffer atomics behind CDNA3 for now
    // GFX942-specific assumptions regarding cache coherence are made when
    // lowering to LLVM
    if (ISAFamily::CDNA3 == triton::AMD::deduceISAFamily(archGenerationName))
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis, solver);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
