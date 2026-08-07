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
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

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

/// If `value` is a `ttg.warp_specialize` partition block argument, return the
/// matching explicit capture; otherwise return `value` unchanged. Walk through
/// nested captures until a non-partition value is reached.
static Value resolveThroughWarpSpecializeCaptures(Value value) {
  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Region *region = blockArg.getParentRegion();
    auto wsOp = region->getParentOfType<ttg::WarpSpecializeOp>();
    if (!wsOp)
      break;

    bool isPartitionEntry = false;
    for (Region *part : wsOp.getPartitionRegions()) {
      if (blockArg.getOwner() == &part->front()) {
        isPartitionEntry = true;
        break;
      }
    }
    if (!isPartitionEntry)
      break;

    unsigned idx = blockArg.getArgNumber();
    ValueRange captures = wsOp.getExplicitCaptures();
    if (idx >= captures.size())
      break;
    LDBG("WS capture unwrap: partition arg #" << idx << " -> " << captures[idx]);
    value = captures[idx];
  }
  return value;
}

bool isFuncArgPtrWithNonNegativeAssumption(mlir::Value value,
  const DenseMap<Value, SetVector<Operation *>> &assumptions) {

  while (value.getDefiningOp() && isa<triton::AddPtrOp>(value.getDefiningOp())) {
    value = value.getDefiningOp<triton::AddPtrOp>().getPtr();
  }

  value = resolveThroughWarpSpecializeCaptures(value);

  if (value.getDefiningOp() || !isa<mlir::BlockArgument>(value))
    return false;

  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(value);
  auto blk = blockArg.getOwner();
  auto funcOp = dyn_cast_or_null<tt::FuncOp>(blk->getParentOp());

  if (funcOp && blk == &funcOp->getRegion(0).front()) {
    if (assumptions.contains(value))
      return true;

    // compatibility with 3.2.x to support easy gen buffer ops for triton.
    // i.e. if any ptr function argument has non-negative assumption, consider
    // this ptr as non-negative as well.
    for (Value arg : funcOp.getArguments()) {
      if (!isa<triton::PointerType>(arg.getType()))
        continue;
      if (assumptions.contains(arg))
        return true;
    }

    return false;
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

// Return true iff the given value v is a tensor splatting from 1 (int).
// The usefulness of this func stems from the fact than if a buffer-op's mask
// operand is a all-1-tensor, it does not need to take this operand.
bool isSplatOneConstTensor(const Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;

  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isOne();

  return false;
}

bool isByteOffsetWithin4GB(triton::AddPtrOp addPtrOp,
                           std::shared_ptr<DataFlowSolver> solver) {
  Value elemIdx = addPtrOp.getOffset();
  LDBG("Determing value-range of element-index: " << elemIdx);

  // step 1: Get the value range of the element index
  const auto *lattice =
      solver->lookupState<dataflow::IntegerValueRangeLattice>(elemIdx);
  if (!lattice) {
    // Note that it is not always able to get lattice, e.g. the element-index
    // is defined by a tt.load.
    LDBG("Cannot get lattice");
    return false;
  }

  const mlir::IntegerValueRange &vr = lattice->getValue();
  if (vr.isUninitialized() || AMD::isEmptyInitializedRange(vr.getValue())) {
    LDBG("Cannot get value range of the offset");
    return false;
  };

  const auto &smin = vr.getValue().smin();
  const auto &smax = vr.getValue().smax();

  LDBG("Element-index value-range: " << smin << " : " << smax);
  if (smin.isNegative() || smax.isNegative())
    return false;

  // step 2: Get element type and size.
  // e.g. addPtrOp.getType is tensor<64x64x!tt.ptr<f16>, then elemTy is
  // !tt.ptr<f16>, and dereferencing elemTy gets f16.
  // TODO: Not sure if we need to keep dereferencing in a loop.
  Type elemTy = getElementTypeOrSelf(addPtrOp.getType());
  while (auto ptrTy = dyn_cast<triton::PointerType>(elemTy))
    elemTy = ptrTy.getPointeeType();

  if (!elemTy || !elemTy.isIntOrFloat()) {
    LDBG("unknown element type: " << elemTy);
    return false;
  }

  // step 3: check if byte-offset is within (4GiB - 9) = -9 (num_records).
  // Sentinel is -8 so that sentinel + inst_size - 1 always exceeds
  // num_records (-9). The limit matches num_records in the resource desc.
  int64_t elemBitSz = elemTy.getIntOrFloatBitWidth();
  int64_t elemMaxIdx = smax.getSExtValue();
  int64_t byteOfst = (elemBitSz * elemMaxIdx + elemBitSz + 7) / 8;
  int64_t szLimit4GB = (int64_t{1} << 32) - 9;

  LDBG("element bit sz:" << elemBitSz << ", max byte offset:" << byteOfst
                         << ((szLimit4GB > byteOfst) ? ", out of range"
                                                     : ", in range"));

  return byteOfst <= szLimit4GB;
}

bool verifyNonNegativeByAssumption(
    Value expr,
    const DenseMap<Value, SetVector<Operation *>> &assumptions) {
  for (const auto &[assumeVal, ops] : assumptions) {
    for (Operation *op : ops) {
      if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
        bool isGreaterThan =
            (cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
             cmpOp.getPredicate() == arith::CmpIPredicate::sgt);
        APInt cst;
        if (isGreaterThan && (cmpOp.getLhs() == expr) &&
            matchPattern(cmpOp.getRhs(), m_ConstantInt(&cst))) {
          return cst.isNonNegative();
        }
      }
    }
    if (auto blockArg = dyn_cast<BlockArgument>(assumeVal)) {
      if (blockArg.getOwner()->isEntryBlock() &&
          isa<tt::PointerType>(blockArg.getType())) {
        return true;
      }
    }
  }
  return false;
}

// Lightweight non-negativity proof used as the default convert-to-buffer-ops
// fallback (aligned with Triton 3.2 after Drop RangeAnalysis).
bool verifyNonNegativeExpr(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions) {
  if (verifyNonNegativeByAssumption(expr, assumptions)) {
    LDBG("Non negative by assumption");
    return true;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(expr)) {
    Block *block = blockArg.getOwner();

    if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp())) {
      if (forOp.getInductionVar() == expr) {
        LDBG("matched forOp loop args");
        Value lowerBound = forOp.getLowerBound();
        Value upperBound = forOp.getUpperBound();
        Value step = forOp.getStep();

        bool lowerBoundNonNeg = verifyNonNegativeExpr(lowerBound, assumptions);
        bool upperBoundNonNeg = verifyNonNegativeExpr(upperBound, assumptions);
        bool stepNonNeg = verifyNonNegativeExpr(step, assumptions);
        LDBG("forOp lowerBoundNonNeg:" << lowerBoundNonNeg);
        LDBG("forOp upperBoundNonNeg:" << upperBoundNonNeg);
        LDBG("forOp stepNonNeg:" << stepNonNeg);
        if (stepNonNeg) {
          return lowerBoundNonNeg;
        }
        return lowerBoundNonNeg && upperBoundNonNeg;
      }
    }
  }

  Operation *op = expr.getDefiningOp();
  if (!op)
    return false;

  return llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
      .Case<triton::BroadcastOp>([&](auto broadcastOp) {
        return verifyNonNegativeExpr(broadcastOp.getSrc(), assumptions);
      })
      .Case<triton::ExpandDimsOp>([&](auto expandOp) {
        return verifyNonNegativeExpr(expandOp.getSrc(), assumptions);
      })
      .Case<triton::SplatOp>([&](auto splatOp) {
        return verifyNonNegativeExpr(splatOp.getSrc(), assumptions);
      })
      .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
        return makeRangeOp.getStart() >= 0 && makeRangeOp.getEnd() >= 0;
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
      .Case<triton::GetProgramIdOp>([&](auto pidOp) { return true; })
      .Case<arith::MaxSIOp>([&](auto maxOp) {
        bool nnLhs = verifyNonNegativeExpr(maxOp.getLhs(), assumptions);
        bool nnRhs = verifyNonNegativeExpr(maxOp.getRhs(), assumptions);
        return nnLhs || nnRhs;
      })
      .Case<arith::RemSIOp>([&](auto remsiOp) {
        return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions);
      })
      .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
        return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions);
      })
      .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
          [&](Operation *binOp) {
            bool nnLhs =
                verifyNonNegativeExpr(binOp->getOperand(0), assumptions);
            bool nnRhs =
                verifyNonNegativeExpr(binOp->getOperand(1), assumptions);
            return nnLhs && nnRhs;
          })
      // Joined offsets from TTGIR `tl.where` / Python `if` (scf.if). Both arms
      // must be non-negative — same soundness as requiring all phi predecessors.
      .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
        return verifyNonNegativeExpr(selectOp.getTrueValue(), assumptions) &&
               verifyNonNegativeExpr(selectOp.getFalseValue(), assumptions);
      })
      .Case<scf::IfOp>([&](scf::IfOp ifOp) {
        // Walk the result index of `expr` through then/else yields.
        unsigned resIdx = mlir::cast<OpResult>(expr).getResultNumber();
        auto thenYield = ifOp.thenYield();
        if (!thenYield || resIdx >= thenYield.getNumOperands())
          return false;
        if (!verifyNonNegativeExpr(thenYield.getOperand(resIdx), assumptions))
          return false;
        // If without else: only then region defines the result.
        if (ifOp.getElseRegion().empty())
          return true;
        auto elseYield = ifOp.elseYield();
        if (!elseYield || resIdx >= elseYield.getNumOperands())
          return false;
        return verifyNonNegativeExpr(elseYield.getOperand(resIdx), assumptions);
      })
      .Default([&](Operation *op) {
        if (auto attr =
                op->template getAttrOfType<mlir::BoolAttr>("non-negative")) {
          if (attr.getValue())
            return true;
        }
        return false;
      });
}

bool isFuncArgWith32bitPtrRange(mlir::Value value) {
  // WASP/WDRA capture kernel pointer args into ttg.warp_specialize partitions.
  // Those partition block args do not carry tt.pointer_range; look through the
  // capture list back to the original func arg.
  value = resolveThroughWarpSpecializeCaptures(value);

  if (value.getDefiningOp())
    return false;

  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(value);
  auto blk = blockArg.getOwner();
  auto funcOp = dyn_cast_or_null<tt::FuncOp>(blk->getParentOp());

  if (funcOp && blk == &funcOp->getRegion(0).front()) {
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (arg != value)
        continue;
      auto attr = funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range");
      return attr && attr.getInt() <= 32;
    }
  }

  return false;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr,
                     const DenseMap<Value, SetVector<Operation *>> &assumptions,
                     std::shared_ptr<DataFlowSolver> solver,
                     bool analyzeSmallTensorOfst, bool useRangeAnalysis) {
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

  // 2. check if the offset is either 32 or 64-bit.
  Value offset = addPtrOp.getOffset();
  auto ofstBit =
      cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth();
  LLVM_DEBUG(llvm::dbgs() << "offset bits:" << ofstBit << "\n");

  // TODO: step 3 and 4 can be reversed to further optimize for performance.
  // When the base-ptr is func argument and has tt.pointer_range=32 attribute,
  // it's safe to promote the mem-op into buffer-op even if offset is a 64-bit
  // value. If this is the case, offset need to be cast down to 32-bit.

  // 3. Bail out if ofst cannot fit in 32-bit.
  if (ofstBit != 32)
    return false;

  // 4. If the base is function formal argument which has attribute
  //  tt.point_range=32, then it's safe to promote this memory op into
  //  bufferOp. In this case, if offset is 64-bit, we should cast it down to
  //  32-bit.
  if (!analyzeSmallTensorOfst &&
      isFuncArgWith32bitPtrRange(maybeSplatOp.getSrc())) {
    LDBG("base-ptr as tt.pointer_range=32 attribute");
    return true;
  }

  // Note: seeem unnecessary to check this here. current reserve this feature and need remove in future ?
  if (isFuncArgPtrWithNonNegativeAssumption(maybeSplatOp.getSrc(), assumptions))
    return true;

  if (useRangeAnalysis) {
    if (!solver)
      return false;
    return isByteOffsetWithin4GB(addPtrOp, std::move(solver));
  }
  return verifyNonNegativeExpr(offset, assumptions);
}

/// Insert `tt.assert` before each `amdgpu.buffer_{load,store}` so that the
/// buffer instruction's element offset tensor encodes a byte offset in
/// [0, 2^32-9] for active lanes (same field lowered to raw.ptr.buffer.* and
/// compared against num_records). Scalar chunks folded into the resource base
/// by pointer canonicalization are not added back here.
static void emitBufferOpOffsetAssert(PatternRewriter &rewriter, Location loc,
                                     Value tensorOffset, Type valueTy,
                                     Value maskOrNull) {
  auto offTy = cast<RankedTensorType>(tensorOffset.getType());
  Value offsets = tensorOffset;
  Type valueElemTy = getElementTypeOrSelf(valueTy);
  const unsigned valueElemNBits =
      std::max(8u, valueElemTy.getIntOrFloatBitWidth());
  const int64_t elementByteWidth = valueElemNBits / 8;

  auto i64TensorTy = RankedTensorType::get(
      offTy.getShape(), rewriter.getI64Type(), offTy.getEncoding());
  Value offI64 = rewriter.create<arith::ExtSIOp>(loc, i64TensorTy, offsets);

  Value strideVal = rewriter.create<arith::ConstantOp>(
      loc, i64TensorTy,
      DenseElementsAttr::get(i64TensorTy,
                             rewriter.getI64IntegerAttr(elementByteWidth)));
  Value byteOff = rewriter.create<arith::MulIOp>(loc, offI64, strideVal);

  const int64_t kMaxByteOff = (int64_t{1} << 32) - 9;
  Value limitVal = rewriter.create<arith::ConstantOp>(
      loc, i64TensorTy,
      DenseElementsAttr::get(i64TensorTy,
                             rewriter.getI64IntegerAttr(kMaxByteOff)));
  Value withinLimit = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, byteOff, limitVal);

  Value zeroOff = rewriter.create<arith::ConstantOp>(
      loc, offTy, DenseElementsAttr::get(offTy, rewriter.getI32IntegerAttr(0)));
  Value nonNeg = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, offsets, zeroOff);

  Value boundsOk = rewriter.create<arith::AndIOp>(loc, nonNeg, withinLimit);

  Value assertCond = boundsOk;
  if (maskOrNull) {
    auto maskTy = cast<RankedTensorType>(maskOrNull.getType());
    Value splatFalse = rewriter.create<arith::ConstantOp>(
        loc, maskTy, DenseElementsAttr::get(maskTy, rewriter.getBoolAttr(false)));
    Value maskOff = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, maskOrNull, splatFalse);
    assertCond = rewriter.create<arith::OrIOp>(loc, maskOff, boundsOk);
  }

  rewriter.create<triton::AssertOp>(
      loc, assertCond,
      rewriter.getStringAttr(
          "triton amdgpu buffer op: element_offset * elem_size must be in "
          "[0, 2^32-9] bytes"));
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

static constexpr int64_t kCacheSwizzleWideAccessBytes = 256;
static constexpr int32_t kDefaultCacheSwizzleStrideBytes = 8192;

static unsigned getValueElementByteWidth(RankedTensorType tensorTy) {
  Type elemTy = tensorTy.getElementType();
  while (auto ptrTy = dyn_cast<triton::PointerType>(elemTy))
    elemTy = ptrTy.getPointeeType();
  return std::max(1u, elemTy.getIntOrFloatBitWidth() / 8);
}

static std::optional<int64_t> getSplatConstantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    return getSplatConstantInt(splat.getSrc());
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
      if (attr.isSplat())
        return attr.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

static std::optional<int64_t>
getUniformLaneElementStrideImpl(Value expr, llvm::DenseSet<Value> &visited) {
  if (!visited.insert(expr).second)
    return std::nullopt;

  if (Operation *op = expr.getDefiningOp()) {
    return llvm::TypeSwitch<Operation *, std::optional<int64_t>>(op)
        .Case<triton::MakeRangeOp>(
            [&](auto) { return std::optional<int64_t>(1); })
        .Case<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(
            [&](Operation *uop) {
              return getUniformLaneElementStrideImpl(uop->getOperand(0),
                                                     visited);
            })
        .Case<arith::ConstantOp, arith::ConstantIntOp, triton::GetProgramIdOp>(
            [&](auto) { return std::optional<int64_t>(0); })
        .Case<arith::AddIOp>([&](arith::AddIOp addOp) -> std::optional<int64_t> {
          auto s0 = getUniformLaneElementStrideImpl(addOp.getLhs(), visited);
          auto s1 = getUniformLaneElementStrideImpl(addOp.getRhs(), visited);
          if (!s0 || !s1)
            return std::nullopt;
          if (*s0 == 0)
            return s1;
          if (*s1 == 0)
            return s0;
          if (*s0 == *s1)
            return s0;
          return std::nullopt;
        })
        .Case<arith::MulIOp>([&](arith::MulIOp mulOp) -> std::optional<int64_t> {
          auto s0 = getUniformLaneElementStrideImpl(mulOp.getLhs(), visited);
          auto s1 = getUniformLaneElementStrideImpl(mulOp.getRhs(), visited);
          if (!s0 || !s1)
            return std::nullopt;
          if (*s0 == 0 && *s1 == 0)
            return 0;
          if (*s0 == 0) {
            if (auto k = getSplatConstantInt(mulOp.getRhs()))
              return *k * *s1;
            return std::nullopt;
          }
          if (*s1 == 0) {
            if (auto k = getSplatConstantInt(mulOp.getLhs()))
              return *k * *s0;
            return std::nullopt;
          }
          return std::nullopt;
        })
        .Case<arith::DivSIOp, arith::RemSIOp, arith::MaxSIOp, arith::MinSIOp>(
            [&](auto) -> std::optional<int64_t> { return std::nullopt; })
        .Default([&](auto) -> std::optional<int64_t> { return std::nullopt; });
  }
  return std::optional<int64_t>(0);
}

static std::optional<int64_t> getUniformLaneElementStride(Value offset) {
  llvm::DenseSet<Value> visited;
  return getUniformLaneElementStrideImpl(offset, visited);
}

static std::optional<int64_t>
getContiguousBytesPerWarp(RankedTensorType tensorTy) {
  Attribute encoding = tensorTy.getEncoding();
  if (!encoding || !isa<ttg::BlockedEncodingAttr>(encoding))
    return std::nullopt;

  SmallVector<unsigned> order = ttg::getOrder(tensorTy);
  if (order.empty())
    return std::nullopt;

  unsigned contigDim = order[0];
  auto shapePerCTA = ttg::getShapePerCTA(tensorTy);
  auto warpsPerCTA = ttg::getWarpsPerCTA(tensorTy);
  if (contigDim >= shapePerCTA.size() || contigDim >= warpsPerCTA.size())
    return std::nullopt;

  int64_t elemsPerWarp =
      shapePerCTA[contigDim] / std::max(1u, warpsPerCTA[contigDim]);
  elemsPerWarp = std::min<int64_t>(elemsPerWarp, shapePerCTA[contigDim]);
  return elemsPerWarp * getValueElementByteWidth(tensorTy);
}

static std::optional<int64_t>
getLinearContiguousAccessBytesPerWarp(RankedTensorType tensorTy,
                                      Value tensorOffset) {
  auto laneStrideElems = getUniformLaneElementStride(tensorOffset);
  if (!laneStrideElems || *laneStrideElems != 1)
    return std::nullopt;
  return getContiguousBytesPerWarp(tensorTy);
}

static Value getBlockStrideFromOffset(Value offset) {
  auto maybeAdd = offset.getDefiningOp<arith::AddIOp>();
  if (!maybeAdd)
    return nullptr;
  for (Value addOpr : maybeAdd.getOperands()) {
    auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>();
    if (!maybeBC)
      continue;
    Value bcSrc = maybeBC.getSrc();
    auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>();
    if (!maybeMul)
      continue;
    for (Value mulOpr : maybeMul.getOperands()) {
      if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>())
        return maybeSplat.getSrc();
    }
  }
  return nullptr;
}

static Value strideElementsToBytes(Location loc, Value strideElems,
                                   unsigned elemBytes,
                                   PatternRewriter &rewriter) {
  if (elemBytes == 1)
    return strideElems;
  Value elemBytesVal =
      rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 32);
  return rewriter.create<arith::MulIOp>(loc, strideElems, elemBytesVal);
}

static std::optional<int64_t>
getMatrixRowPitchBytes(RankedTensorType ptrTensorTy, Value tensorOffset) {
  unsigned elemBytes = getValueElementByteWidth(ptrTensorTy);
  if (Value strideElems = getBlockStrideFromOffset(tensorOffset)) {
    if (auto cst = strideElems.getDefiningOp<arith::ConstantIntOp>())
      return cst.value() * elemBytes;
  }
  return std::nullopt;
}

static Value makeNormalizedCacheSwizzleStride(Location loc, int64_t strideBytes,
                                              int64_t minRowPitchBytes,
                                              PatternRewriter &rewriter) {
  int64_t normalized = AMD::normalizeCacheSwizzleStrideBytes(
      strideBytes, minRowPitchBytes);
  if (normalized <= 0) {
    LDBG("Skip cache swizzle: cannot encode stride bytes=" << strideBytes
         << " minRowPitch=" << minRowPitchBytes);
    return nullptr;
  }
  return rewriter.create<arith::ConstantIntOp>(loc, normalized, 32);
}

static Value resolveCacheSwizzleStride(Location loc, RankedTensorType ptrTensorTy,
                                       Value tensorOffset,
                                       PatternRewriter &rewriter,
                                       bool bufferCacheSwizzleEnabled) {
  if (!bufferCacheSwizzleEnabled)
    return nullptr;

  int64_t minRowPitchBytes = 0;
  if (auto rowPitch = getMatrixRowPitchBytes(ptrTensorTy, tensorOffset))
    minRowPitchBytes = *rowPitch;

  if (auto bytesPerWarp =
          getLinearContiguousAccessBytesPerWarp(ptrTensorTy, tensorOffset)) {
    if (*bytesPerWarp >= kCacheSwizzleWideAccessBytes) {
      LDBG("Skip cache swizzle: linear contiguous bytes per warp = "
           << *bytesPerWarp);
      return nullptr;
    }
  }

  unsigned elemBytes = getValueElementByteWidth(ptrTensorTy);
  if (Value strideElems = getBlockStrideFromOffset(tensorOffset)) {
    if (auto cst = strideElems.getDefiningOp<arith::ConstantIntOp>()) {
      int64_t strideBytes = cst.value() * elemBytes;
      LDBG("Inferred cache swizzle stride bytes: " << strideBytes
           << " rowPitch=" << minRowPitchBytes);
      return makeNormalizedCacheSwizzleStride(loc, strideBytes, minRowPitchBytes,
                                              rewriter);
    }
    return strideElementsToBytes(loc, strideElems, elemBytes, rewriter);
  }

  LDBG("Using default cache swizzle stride bytes: "
       << kDefaultCacheSwizzleStrideBytes
       << " rowPitch=" << minRowPitchBytes);
  return makeNormalizedCacheSwizzleStride(loc, kDefaultCacheSwizzleStrideBytes,
                                          minRowPitchBytes, rewriter);
}

// /*-----------------AtomicCAS-------------------*/

struct ConvertTritonAtomicCASOpToBufferAtomicCAS
    : public mlir::OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicCASOpToBufferAtomicCAS(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_,
      bool useRangeAnalysis_, bool emitBufferOpsOffsetAssert_)
      : mlir::OpRewritePattern<triton::AtomicCASOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_),
        useRangeAnalysis(useRangeAnalysis_),
        emitBufferOpsOffsetAssert(emitBufferOpsOffsetAssert_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicCASOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto sem = op.getSem();
    auto scope = op.getScope();

    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst,
                       useRangeAnalysis)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "CAS with unsupported scope");
    }
    LDBG("CAS supported scope");

    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "CAS with unsupported memory ordering");
    }

    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value tensorOffset = addPtrOp.getOffset();
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // Buffer atomic CAS only supports i32/i64
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "AtomicCAS with unsupported type");
    }
    LDBG("AtomicCAS supported type");

    // Buffer atomics support 32 and 64-bit operations, so inputs must be at
    // least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth =
          getVectorSize(basePtr, tensorOffset, axisAnalysisPass) * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(
          op, "BufferAtomicCAS requires opBitWidth >= 32");
    }
    if (emitBufferOpsOffsetAssert)
      emitBufferOpOffsetAssert(rewriter, op->getLoc(), tensorOffset,
                               op.getVal().getType(), Value());
    Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicCASOp>(
        op, op.getVal().getType(), basePtr, tensorOffset, op.getCmp(),
        op.getVal(), blockStride, sem, scope);
    return success();
  }

private:
  // Assumptions collected through the function
  const DenseMap<Value, SetVector<Operation *>> &assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
  bool useRangeAnalysis;
  bool emitBufferOpsOffsetAssert;
};

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, ISAFamily isaFamily,
      bool analyzeSmallTensorOfst_, bool useRangeAnalysis_,
      bool emitBufferOpsOffsetAssert_, std::string arch_)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)), isaFamily(isaFamily), arch(std::move(arch_)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_),
        useRangeAnalysis(useRangeAnalysis_),
        emitBufferOpsOffsetAssert(emitBufferOpsOffsetAssert_) {}

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
    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst,
                       useRangeAnalysis)) {
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

    // HCU: bf16 is unsupported everywhere on CDNA3; f16 only works on gfx942
    if (isaFamily == ISAFamily::CDNA3 && atomicRmwOp == RMWOp::FADD &&
        (checkType.isBF16() || (checkType.isF16() && arch != "gfx942"))) {
      return rewriter.notifyMatchFailure(
          op, "CDNA3 does not support 16-bit buffer atomic fadd");
    }
    LDBG("RMW FADD supported 16-bit type");

    auto vecSize = getVectorSize(ptr, axisAnalysisPass);
    if (auto mask = op.getMask()) {
      vecSize = std::min(vecSize, axisAnalysisPass.getMaskAlignment(mask));
    }
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
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    case RMWOp::MAX:
    case RMWOp::MIN:
      // TODO: It likely means smax/smin, for now intrinsic
      // llvm.amdgcn.raw.ptr.buffer.atomic.{min|max} is emitted, and llvm get
      // confused as how to deal with {f|s|u}{min|max}.
      if (!checkType.isInteger())
        break;
      // else fall through
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
    if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
      maybeMask = op.getMask();
    if (emitBufferOpsOffsetAssert)
      emitBufferOpOffsetAssert(rewriter, op->getLoc(), tensorOffset,
                               op.getVal().getType(), maybeMask);
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
  ISAFamily isaFamily;
  std::string arch;
  bool analyzeSmallTensorOfst;
  bool useRangeAnalysis;
  bool emitBufferOpsOffsetAssert;
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
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_,
      bool useRangeAnalysis_, bool emitBufferOpsOffsetAssert_,
      bool bufferCacheSwizzleEnabled_)
      : mlir::OpRewritePattern<SourceOp>(context), assumptions(assumptions),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_),
        useRangeAnalysis(useRangeAnalysis_),
        emitBufferOpsOffsetAssert(emitBufferOpsOffsetAssert_),
        bufferCacheSwizzleEnabled(bufferCacheSwizzleEnabled_) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getOperand(0);

    // Triton 3.2 has no f32+tt.dot LLC dodge; allow converting f32 loads even
    // when the function contains tt.dot.

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst,
                       useRangeAnalysis)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
      if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
        auto ptrTensorTy = cast<RankedTensorType>(ptr.getType());
        blockStride = resolveCacheSwizzleStride(
            op->getLoc(), ptrTensorTy, tensorOffset, rewriter,
            bufferCacheSwizzleEnabled);
      }

      if (emitBufferOpsOffsetAssert) {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          emitBufferOpOffsetAssert(rewriter, op->getLoc(), tensorOffset,
                                   op.getType(), maybeMask);
        }
      }

      auto bufferLoadOp = [&]() {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          return triton::amdgpu::BufferLoadOp::create(
              rewriter, op->getLoc(), op.getType(), basePtr, tensorOffset,
              blockStride, op.getCache(), maybeMask, maybeOther);
        } else if constexpr (std::is_same_v<
                                 SourceOp,
                                 triton::gpu::AsyncCopyGlobalToLocalOp>) {
          return triton::amdgpu::BufferLoadToLocalOp::create(
              rewriter, op->getLoc(), op.getType(), op.getResult(), basePtr,
              tensorOffset, maybeMask, maybeOther, blockStride, op.getCache(),
              op.getContiguity());
        } else {
          static_assert(always_false<SourceOp>::value,
                        "Unsupported type in ConvertTritonLoadToBufferLoad");
        }
      }();

      assert(bufferLoadOp);

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
  bool analyzeSmallTensorOfst;
  bool useRangeAnalysis;
  bool emitBufferOpsOffsetAssert;
  bool bufferCacheSwizzleEnabled;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_,
      bool useRangeAnalysis_, bool emitBufferOpsOffsetAssert_,
      bool bufferCacheSwizzleEnabled_)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions), solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_),
        useRangeAnalysis(useRangeAnalysis_),
        emitBufferOpsOffsetAssert(emitBufferOpsOffsetAssert_),
        bufferCacheSwizzleEnabled(bufferCacheSwizzleEnabled_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst,
                       useRangeAnalysis)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
        maybeMask = op.getMask();
      auto ptrTensorTy = cast<RankedTensorType>(ptr.getType());
      Value blockStride = resolveCacheSwizzleStride(
          op->getLoc(), ptrTensorTy, tensorOffset, rewriter,
          bufferCacheSwizzleEnabled);
      if (emitBufferOpsOffsetAssert)
        emitBufferOpOffsetAssert(rewriter, op->getLoc(), tensorOffset,
                                 op.getValue().getType(), maybeMask);
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
  bool analyzeSmallTensorOfst;
  bool useRangeAnalysis;
  bool emitBufferOpsOffsetAssert;
  bool bufferCacheSwizzleEnabled;
};

} // anonymous namespace

struct TritonAMDGPUConvertToBufferOpsPass
    : impl::TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    auto arch = getAMDArch(mod);
    triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");

    // Collect assumptions in the function (cheap; used by assume / nonneg paths)
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    collectAssumptionsForFuncArgPtr(mod, assumptions);

    std::shared_ptr<DataFlowSolver> solver;
    if (this->useRangeAnalysis) {
      solver = createDataFlowSolver();
      AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
          solver->load<AMD::TritonIntegerRangeAnalysis>(
              assumptions, &getAnalysis<DominanceInfo>());
      AMD::initializeFuncOps(mod, rangeAnalysis);
      if (failed(solver->initializeAndRun(getOperation())))
        return signalPassFailure();
    }

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    const bool bufferCacheSwizzleEnabled =
        this->bufferCacheSwizzle && arch &&
        AMD::supportsBufferCacheSwizzle(*arch);
    patterns.add<ConvertTritonLoadToBufferLoad<tt::LoadOp>,
                 ConvertTritonStoreToBufferStore>(
        context, assumptions, solver, this->analyzeSmallTensorOfst,
        this->useRangeAnalysis, this->emitBufferOpsOffsetAssert,
        bufferCacheSwizzleEnabled);
    // BufferLoadToLds is only supported on CDNA3 and CDNA4
    if (llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily())) {
      patterns
          .add<ConvertTritonLoadToBufferLoad<ttg::AsyncCopyGlobalToLocalOp>>(
              context, assumptions, solver, this->analyzeSmallTensorOfst,
              this->useRangeAnalysis,
              this->emitBufferOpsOffsetAssert,
              /*bufferCacheSwizzleEnabled=*/false);
    }

    // Gate buffer atomics behind CDNA3 for now
    // GFX942-specific assumptions regarding cache coherence are made when
    // lowering to LLVM
    triton::AMD::ISAFamily isaFamily =
        triton::AMD::deduceISAFamily(archGenerationName);
    if (this->allowBufferAtomics &&
        (ISAFamily::CDNA3 == isaFamily || ISAFamily::CDNA4 == isaFamily))
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis, solver, isaFamily,
          this->analyzeSmallTensorOfst, this->useRangeAnalysis,
          this->emitBufferOpsOffsetAssert, archGenerationName);
    patterns.add<ConvertTritonAtomicCASOpToBufferAtomicCAS>(
        context, assumptions, axisInfoAnalysis, solver,
        this->analyzeSmallTensorOfst, this->useRangeAnalysis,
        this->emitBufferOpsOffsetAssert);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
