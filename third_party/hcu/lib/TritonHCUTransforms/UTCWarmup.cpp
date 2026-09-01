#include "TritonHCU/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <array>
#include <memory>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonhcu-utc-warmup"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tta = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUUTCWARMUP
#include "TritonHCU/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// HCU UTC translation warmup for GEMM-like reductions
//===----------------------------------------------------------------------===//
//
// Large GEMMs and GEMM-like kernels can expose many previously unseen 2 MiB
// pages when their first A/B matrix loads enter a K-reduction loop. The burst
// of UTCL1 misses backpressures the TA/TCP request path and delays the matrix
// pipeline. This pass is an opt-in autotune feature that issues asynchronous
// dword loads before the reduction solely to establish those translations. It
// is independent of BlockPingpong: either optimization may be enabled without
// the other, and small or unsupported kernels keep the normal pipeline.
//
// The pass runs immediately after conversion to TTGIR, before Coalesce and
// software pipelining. At this point a GEMM operand still has its logical
// make_range/addptr structure and each payload load exists exactly once. For
// every direct dot-family reduction loop, A and B are analyzed independently:
//
//   1. Find the unique source tt.load or tt.matrix_load for that dot operand.
//   2. Recover the first-iteration address as the strict affine form
//        tileBase + nonKIndex * nonKStride + kIndex * kStride.
//   3. Require a unit K stride and a canonical BK pointer advance.
//   4. Prove that load masks describe a K tail, a complete M/N tile, a
//      CTA-uniform predicate, or an AND of those forms.
//   5. Compute the runtime address span and warm 2..16 complete 2 MiB pages.
//
// Thus a compact operand becomes eligible at BM*K*elementBytes >= 4 MiB for A
// or K*BN*elementBytes >= 4 MiB for B. The actual implementation uses the
// recovered runtime strides, so padded leading dimensions and irregular K are
// handled by the same page-footprint model rather than by fixed K thresholds.
//
// Failure for one operand does not disable the other. Address and mask
// matching construct deferred ScalarExpr/GuardPlan objects and do not mutate
// the IR; operations are materialized only after one operand has passed the
// complete proof. This prevents rejected candidates from leaving dead helper
// arithmetic before the loop.
//
// utc_warmup is an independent autotune dimension from BlockPingpong. Enabling
// it performs two coupled actions for every operand that passes the proof:
//
//   1. Translation warmup: issue buffer_load_dword requests before the
//      reduction loop, under the runtime page-footprint/boundary guard, to
//      establish UTCL1 translations before the first matrix payload request.
//   2. Payload stride override: associate a matched tt.load with an
//      architecture-specific cache-swizzle stride. Matrix loads use their own
//      descriptor path and are deliberately left unchanged.
//
// The generated lifetime is:
//
//   before loop: scf.if(runtime footprint/mask guard) {
//                  amdg.utc_warmup       // one dword request per HW thread
//                }
//   payload:     original tt.load optionally tagged with the private warmup ID
//   after loop:  amdg.utc_warmup_consume // keep async destination VGPR live
//
// The private ID survives later cloning and associates only that payload load
// with the target cache-swizzle stride:
//
//   chip      arch     warmup request       matched payload stride
//   ZhongDa   gfx928   under runtime guard  8 KiB
//   YueYing   gfx92a   under runtime guard  8 KiB
//   BMZ       gfx936   under runtime guard  8 KiB
//   NMZ       gfx938   under runtime guard  32 KiB
//   ShaoBo    gfx946   under runtime guard  32 KiB
//
// The runtime guard controls whether the translation request executes; it
// deliberately does not select the descriptor stride. NMZ and early ShaoBo
// have a known 32 KiB base/offset alias hardware bug (fixed in ShaoBo B1), so
// this option must remain autotuned and covered by end-to-end correctness
// tests.
//
// This pass intentionally rejects gathered, nonlinear, ambiguous or otherwise
// non-affine operand tiles. It never guesses a stride from an
// arbitrary shape-compatible multiply because a false match would affect both
// the extra memory request and, on newer targets, the payload descriptor.

constexpr StringLiteral kUTCWarmupIdAttr = "hcu.utc_warmup.id";
constexpr int64_t kUTCPageBytes = 2 * 1024 * 1024;
constexpr int64_t kMinUTCWarmupPages = 2;
constexpr int64_t kMaxUTCWarmupPages = 16;
constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kBit8Width = 8;
constexpr unsigned kBit16Width = 16;
constexpr unsigned kNumOperands = 2;
constexpr unsigned kOperandA = 0;
constexpr unsigned kOperandB = 1;

struct Candidate {
  scf::ForOp loop;
  std::array<Operation *, kNumOperands> loads;
  Value kExtent;
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;
};

enum class ScalarExprKind {
  Value,
  Constant,
  Add,
  Mul,
  ExtSI,
  ExtUI,
  TruncI,
  IndexCast,
};

struct ScalarExpr;
using ScalarExprPtr = std::shared_ptr<ScalarExpr>;

// A deferred scalar expression. Matching constructs this tree without touching
// the IR; it is materialized only after every proof for one operand succeeds.
struct ScalarExpr {
  ScalarExprKind kind;
  Type type;
  Value value;
  int64_t constant = 0;
  ScalarExprPtr lhs;
  ScalarExprPtr rhs;
};

struct RangeTerm {
  int64_t extent;
  int dimension = -1;
  ScalarExprPtr coefficient;
};

struct OffsetAnalysis {
  ScalarExprPtr uniform;
  SmallVector<RangeTerm> ranges;
};

struct WarmupOperandPlan {
  Value basePtr;
  ScalarExprPtr tileBaseOffset;
  ScalarExprPtr nonKStrideElements;
  ScalarExprPtr kStrideElements;
  int64_t nonKTileSize;
  unsigned elementBytes;
};

struct WarmupOperand {
  Value basePtr;
  Value nonKStrideElements;
  Value kStrideElements;
  int64_t nonKTileSize;
  unsigned elementBytes;
};

struct GuardPlan;
using GuardPlanPtr = std::shared_ptr<GuardPlan>;

struct GuardPlan {
  enum class Kind { True, Scalar, FullTile, And } kind;
  Value scalar;
  ScalarExprPtr tileBase;
  Value extent;
  int64_t tileSize = 0;
  GuardPlanPtr lhs;
  GuardPlanPtr rhs;
};

struct PreparedWarmupOperand {
  unsigned index;
  Operation *payloadLoad;
  WarmupOperandPlan operand;
  GuardPlanPtr maskGuard;
};

Value asI64(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return arith::IndexCastOp::create(builder, loc, builder.getI64Type(), value);
  auto integerType = dyn_cast<IntegerType>(value.getType());
  if (!integerType)
    return nullptr;
  unsigned width = integerType.getWidth();
  if (width == 64)
    return value;
  if (width < 64)
    return arith::ExtSIOp::create(builder, loc, builder.getI64Type(), value);
  return arith::TruncIOp::create(builder, loc, builder.getI64Type(), value);
}

bool hasShape(Type type, ArrayRef<int64_t> expected) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getShape() == expected;
}

// Follow one dot operand only within the reduction body. Exactly one load with
// the logical operand shape is required; auxiliary scale/mask loads are
// ignored, while two same-shaped data loads make this operand ambiguous.
FailureOr<Operation *> findGemmOperandLoad(Value operand, scf::ForOp loop,
                                           ArrayRef<int64_t> expectedShape) {
  Operation *root = operand.getDefiningOp();
  if (!root || root->getBlock() != loop.getBody())
    return failure();

  SetVector<Operation *> dependencies;
  dependencies.insert(root);
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  options.filter = [&](Operation *op) {
    return op->getBlock() == loop.getBody();
  };
  (void)getBackwardSlice(root, &dependencies, options);

  Operation *source = nullptr;
  for (Operation *dependency : dependencies) {
    if (!isa<tt::LoadOp, tt::MatrixLoadOp>(dependency) ||
        dependency->getParentOp() != loop ||
        !hasShape(dependency->getResult(0).getType(), expectedShape))
      continue;
    if (source)
      return failure();
    source = dependency;
  }
  if (!source)
    return failure();
  return source;
}

// Recover K from loops written as 0..ceildiv(K, BK), step 1. Triton emits both
// ceildiv and the equivalent (K + BK - 1) / BK spelling.
Value matchRoundedUpTripCount(Value upperBound, int64_t blockK) {
  Value numerator;
  Value denominator;
  if (auto div = upperBound.getDefiningOp<arith::DivSIOp>()) {
    numerator = div.getLhs();
    denominator = div.getRhs();
  } else if (auto div = upperBound.getDefiningOp<arith::DivUIOp>()) {
    numerator = div.getLhs();
    denominator = div.getRhs();
  } else if (auto div = upperBound.getDefiningOp<arith::CeilDivSIOp>()) {
    if (isConstantIntValue(div.getRhs(), blockK))
      return div.getLhs();
    return nullptr;
  } else if (auto div = upperBound.getDefiningOp<arith::CeilDivUIOp>()) {
    if (isConstantIntValue(div.getRhs(), blockK))
      return div.getLhs();
    return nullptr;
  } else {
    return nullptr;
  }
  if (!isConstantIntValue(denominator, blockK))
    return nullptr;

  auto add = numerator.getDefiningOp<arith::AddIOp>();
  if (!add)
    return nullptr;
  if (isConstantIntValue(add.getLhs(), blockK - 1))
    return add.getRhs();
  if (isConstantIntValue(add.getRhs(), blockK - 1))
    return add.getLhs();
  return nullptr;
}

// Accept the two canonical reduction forms used by GEMM kernels:
//   range(0, K, BK), or range(0, ceildiv(K, BK), 1).
FailureOr<Value> getReductionExtent(scf::ForOp loop, int64_t blockK,
                                    int64_t loopStep) {
  if (loopStep == blockK)
    return loop.getUpperBound();
  if (loopStep != 1)
    return failure();
  Value kExtent = matchRoundedUpTripCount(loop.getUpperBound(), blockK);
  if (!kExtent)
    return failure();
  return kExtent;
}

// Match only the reduction and dot-level contract here. Source-load discovery
// is deliberately per operand: failure to find A must not suppress a valid B,
// and vice versa. Detailed pointer and mask proofs happen independently later.
FailureOr<Candidate> matchCandidate(scf::ForOp loop) {
  APInt lowerBound, step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lowerBound)) ||
      lowerBound.getSExtValue() != 0 ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      ttg::lookupNumCTAs(loop) != 1)
    return failure();

  SmallVector<tt::DotOpInterface> dots;
  loop.walk([&](tt::DotOpInterface dot) { dots.push_back(dot); });
  if (dots.size() != 1 || dots.front()->getParentOp() != loop)
    return failure();

  tt::DotOpInterface dot = dots.front();
  auto resultType =
      dyn_cast<RankedTensorType>(dot->getResult(0).getType());
  auto aType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(dot.getB().getType());
  if (!resultType || !aType || !bType || resultType.getRank() != 2 ||
      aType.getRank() != 2 || bType.getRank() != 2)
    return failure();

  int64_t blockM = resultType.getShape()[0];
  int64_t blockN = resultType.getShape()[1];
  int64_t blockK = aType.getShape()[1];
  if (aType.getShape()[0] != blockM || bType.getShape()[0] != blockK ||
      bType.getShape()[1] != blockN)
    return failure();

  auto kExtent =
      getReductionExtent(loop, blockK, step.getSExtValue());
  if (failed(kExtent))
    return failure();

  auto loadA = findGemmOperandLoad(dot.getA(), loop, {blockM, blockK});
  auto loadB = findGemmOperandLoad(dot.getB(), loop, {blockK, blockN});
  Operation *sourceA = succeeded(loadA) ? *loadA : nullptr;
  Operation *sourceB = succeeded(loadB) ? *loadB : nullptr;
  if ((!sourceA && !sourceB) || (sourceA && sourceB && sourceA == sourceB))
    return failure();

  return Candidate{loop, {sourceA, sourceB}, *kExtent, blockM, blockN, blockK};
}

// A loop-carried payload pointer denotes a later K iteration. Warmup must start
// from its init argument, i.e. the first CTA panel seen by the reduction.
Value resolveFirstIterationValue(Value value, scf::ForOp loop) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != loop.getBody() ||
      blockArg.getArgNumber() == 0)
    return value;
  unsigned initIndex = blockArg.getArgNumber() - 1;
  return initIndex < loop.getInitArgs().size() ? loop.getInitArgs()[initIndex]
                                               : value;
}

ScalarExprPtr makeValueExpr(Value value) {
  return std::make_shared<ScalarExpr>(
      ScalarExpr{ScalarExprKind::Value, value.getType(), value});
}

ScalarExprPtr makeConstantExpr(Type type, int64_t value) {
  ScalarExpr expr{ScalarExprKind::Constant, type};
  expr.constant = value;
  return std::make_shared<ScalarExpr>(std::move(expr));
}

bool isConstantExpr(const ScalarExprPtr &expr, int64_t expected) {
  if (!expr)
    return expected == 0;
  if (expr->kind == ScalarExprKind::Constant)
    return expr->constant == expected;
  if (expr->kind == ScalarExprKind::Value) {
    APInt constant;
    return matchPattern(expr->value, m_ConstantInt(&constant)) &&
           constant.getSExtValue() == expected;
  }
  if (expr->kind == ScalarExprKind::ExtSI ||
      expr->kind == ScalarExprKind::ExtUI ||
      expr->kind == ScalarExprKind::TruncI ||
      expr->kind == ScalarExprKind::IndexCast)
    return isConstantExpr(expr->lhs, expected);
  return false;
}

ScalarExprPtr makeBinaryExpr(ScalarExprKind kind, ScalarExprPtr lhs,
                             ScalarExprPtr rhs) {
  assert(lhs && rhs);
  if (kind == ScalarExprKind::Add) {
    if (isConstantExpr(lhs, 0))
      return rhs;
    if (isConstantExpr(rhs, 0))
      return lhs;
  } else if (kind == ScalarExprKind::Mul) {
    if (isConstantExpr(lhs, 0) || isConstantExpr(rhs, 0))
      return makeConstantExpr(lhs->type, 0);
    if (isConstantExpr(lhs, 1))
      return rhs;
    if (isConstantExpr(rhs, 1))
      return lhs;
  }
  ScalarExpr expr{kind, lhs->type};
  expr.lhs = std::move(lhs);
  expr.rhs = std::move(rhs);
  return std::make_shared<ScalarExpr>(std::move(expr));
}

ScalarExprPtr addExpr(ScalarExprPtr lhs, ScalarExprPtr rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return makeBinaryExpr(ScalarExprKind::Add, std::move(lhs), std::move(rhs));
}

ScalarExprPtr multiplyExpr(ScalarExprPtr lhs, ScalarExprPtr rhs) {
  if (!lhs || !rhs)
    return nullptr;
  return makeBinaryExpr(ScalarExprKind::Mul, std::move(lhs), std::move(rhs));
}

ScalarExprPtr castExpr(ScalarExprKind kind, Type type, ScalarExprPtr operand) {
  if (!operand)
    return nullptr;
  ScalarExpr expr{kind, type};
  expr.lhs = std::move(operand);
  return std::make_shared<ScalarExpr>(std::move(expr));
}

// Matrix-load indices are i32 while its strides are i64. Preserve the same
// signed integer interpretation used by pointer arithmetic when composing the
// deferred tile-base offset.
ScalarExprPtr extendIntegerExprTo(ScalarExprPtr expr, Type targetType) {
  if (!expr || expr->type == targetType)
    return expr;
  auto sourceType = dyn_cast<IntegerType>(expr->type);
  auto destinationType = dyn_cast<IntegerType>(targetType);
  if (!sourceType || !destinationType ||
      sourceType.getWidth() > destinationType.getWidth())
    return nullptr;
  return castExpr(ScalarExprKind::ExtSI, targetType, std::move(expr));
}

Value materializeExpr(OpBuilder &builder, Location loc,
                      const ScalarExprPtr &expr) {
  assert(expr);
  switch (expr->kind) {
  case ScalarExprKind::Value:
    return expr->value;
  case ScalarExprKind::Constant:
    if (expr->type.isIndex())
      return arith::ConstantIndexOp::create(builder, loc, expr->constant);
    return arith::ConstantOp::create(
        builder, loc, expr->type,
        builder.getIntegerAttr(expr->type, expr->constant));
  case ScalarExprKind::Add:
    return arith::AddIOp::create(builder, loc,
                                 materializeExpr(builder, loc, expr->lhs),
                                 materializeExpr(builder, loc, expr->rhs));
  case ScalarExprKind::Mul:
    return arith::MulIOp::create(builder, loc,
                                 materializeExpr(builder, loc, expr->lhs),
                                 materializeExpr(builder, loc, expr->rhs));
  case ScalarExprKind::ExtSI:
    return arith::ExtSIOp::create(builder, loc, expr->type,
                                  materializeExpr(builder, loc, expr->lhs));
  case ScalarExprKind::ExtUI:
    return arith::ExtUIOp::create(builder, loc, expr->type,
                                  materializeExpr(builder, loc, expr->lhs));
  case ScalarExprKind::TruncI:
    return arith::TruncIOp::create(builder, loc, expr->type,
                                   materializeExpr(builder, loc, expr->lhs));
  case ScalarExprKind::IndexCast:
    return arith::IndexCastOp::create(builder, loc, expr->type,
                                      materializeExpr(builder, loc, expr->lhs));
  }
  llvm_unreachable("unknown deferred scalar expression");
}

// A make_range begins as rank 1. Its GEMM dimension becomes unambiguous after
// expand_dims creates [extent, 1] or [1, extent]; later broadcasts preserve it.
void inferRangeDimensions(OffsetAnalysis &analysis, Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getRank() != 2)
    return;
  ArrayRef<int64_t> shape = tensorType.getShape();
  for (RangeTerm &term : analysis.ranges) {
    if (term.dimension >= 0)
      continue;
    bool dimension0 = shape[0] == term.extent && shape[1] == 1;
    bool dimension1 = shape[1] == term.extent && shape[0] == 1;
    if (dimension0 != dimension1)
      term.dimension = dimension0 ? 0 : 1;
  }
}

// Symbolically decompose an integer tensor offset into one CTA-uniform scalar
// expression and a list of unit-range terms with scalar coefficients. The
// accepted operations preserve an affine address (splat/broadcast/expand,
// integer casts, add, and multiplication by a fully uniform value). Multiplying
// two varying expressions is rejected. No IR is created by this analysis.
FailureOr<OffsetAnalysis> analyzeOffset(Value value, scf::ForOp loop) {
  if (!isa<RankedTensorType>(value.getType())) {
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)))
      return OffsetAnalysis{
          makeConstantExpr(value.getType(), constant.getSExtValue()), {}};
    if (!loop.isDefinedOutsideOfLoop(value))
      return failure();
    return OffsetAnalysis{makeValueExpr(value), {}};
  }

  if (auto splat = value.getDefiningOp<tt::SplatOp>())
    return analyzeOffset(splat.getSrc(), loop);

  if (auto range = value.getDefiningOp<tt::MakeRangeOp>()) {
    auto tensorType = cast<RankedTensorType>(range.getType());
    Type elementType = tensorType.getElementType();
    OffsetAnalysis result;
    if (range.getStart() != 0)
      result.uniform = makeConstantExpr(elementType, range.getStart());
    result.ranges.push_back(RangeTerm{range.getEnd() - range.getStart(), -1,
                                      makeConstantExpr(elementType, 1)});
    return result;
  }

  auto analyzeShapeOnly = [&](Value source) -> FailureOr<OffsetAnalysis> {
    auto result = analyzeOffset(source, loop);
    if (succeeded(result))
      inferRangeDimensions(*result, value.getType());
    return result;
  };
  if (auto broadcast = value.getDefiningOp<tt::BroadcastOp>())
    return analyzeShapeOnly(broadcast.getSrc());
  if (auto expand = value.getDefiningOp<tt::ExpandDimsOp>())
    return analyzeShapeOnly(expand.getSrc());
  if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>())
    return analyzeShapeOnly(convert.getSrc());

  auto analyzeCast = [&](Value source,
                         ScalarExprKind kind) -> FailureOr<OffsetAnalysis> {
    auto result = analyzeOffset(source, loop);
    if (failed(result))
      return failure();
    Type elementType = cast<RankedTensorType>(value.getType()).getElementType();
    result->uniform = castExpr(kind, elementType, result->uniform);
    for (RangeTerm &term : result->ranges)
      term.coefficient =
          castExpr(kind, elementType, std::move(term.coefficient));
    inferRangeDimensions(*result, value.getType());
    return result;
  };
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return analyzeCast(ext.getIn(), ScalarExprKind::ExtSI);
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return analyzeCast(ext.getIn(), ScalarExprKind::ExtUI);
  if (auto trunc = value.getDefiningOp<arith::TruncIOp>())
    return analyzeCast(trunc.getIn(), ScalarExprKind::TruncI);
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return analyzeCast(cast.getIn(), ScalarExprKind::IndexCast);

  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = analyzeOffset(add.getLhs(), loop);
    auto rhs = analyzeOffset(add.getRhs(), loop);
    if (failed(lhs) || failed(rhs))
      return failure();
    lhs->uniform = addExpr(std::move(lhs->uniform), std::move(rhs->uniform));
    llvm::append_range(lhs->ranges, std::move(rhs->ranges));
    inferRangeDimensions(*lhs, value.getType());
    return lhs;
  }

  if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = analyzeOffset(multiply.getLhs(), loop);
    auto rhs = analyzeOffset(multiply.getRhs(), loop);
    if (failed(lhs) || failed(rhs) ||
        (!lhs->ranges.empty() && !rhs->ranges.empty()))
      return failure();

    OffsetAnalysis *varying = lhs->ranges.empty() ? &*rhs : &*lhs;
    OffsetAnalysis *uniform = lhs->ranges.empty() ? &*lhs : &*rhs;
    if (!uniform->uniform)
      return failure();
    varying->uniform =
        multiplyExpr(std::move(varying->uniform), uniform->uniform);
    for (RangeTerm &term : varying->ranges)
      term.coefficient =
          multiplyExpr(std::move(term.coefficient), uniform->uniform);
    inferRangeDimensions(*varying, value.getType());
    return std::move(*varying);
  }
  return failure();
}

// Peel shape-only wrappers until a mask reaches the expected singleton GEMM
// dimension. A value with another logical shape is left unchanged and fails
// the caller's structural proof.
Value stripToShape(Value value, ArrayRef<int64_t> expectedShape) {
  while (true) {
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = convert.getSrc();
      continue;
    }
    if (hasShape(value.getType(), expectedShape))
      return value;
    if (auto broadcast = value.getDefiningOp<tt::BroadcastOp>()) {
      value = broadcast.getSrc();
      continue;
    }
    if (auto expand = value.getDefiningOp<tt::ExpandDimsOp>()) {
      value = expand.getSrc();
      continue;
    }
    return value;
  }
}

// Recognize the canonical arange(0, BK) + iv < K predicate. It is safe to
// ignore for warmup because the page-footprint calculation uses the exact
// runtime K and drops the final partial 2 MiB page.
bool isKBoundaryMask(Value mask, scf::ForOp loop,
                     Value kExtent,
                     ArrayRef<int64_t> expectedSourceShape) {
  Value source = stripToShape(mask, expectedSourceShape);
  if (!hasShape(source.getType(), expectedSourceShape))
    return false;
  auto compare = source.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt)
    return false;
  auto upper = compare.getRhs().getDefiningOp<tt::SplatOp>();
  if (!upper || upper.getSrc() != kExtent)
    return false;
  auto add = compare.getLhs().getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  auto lhs = add.getLhs().getDefiningOp<tt::SplatOp>();
  auto rhs = add.getRhs().getDefiningOp<tt::SplatOp>();
  bool lhsIsIV = lhs && lhs.getSrc() == loop.getInductionVar();
  bool rhsIsIV = rhs && rhs.getSrc() == loop.getInductionVar();
  if (lhsIsIV == rhsIsIV)
    return false;
  auto index = analyzeOffset(lhsIsIV ? add.getRhs() : add.getLhs(), loop);
  return succeeded(index) && !index->uniform && index->ranges.size() == 1 &&
         index->ranges.front().extent ==
             expectedSourceShape.front() * expectedSourceShape.back() &&
         isConstantExpr(index->ranges.front().coefficient, 1);
}

// Convert offs_m < M or offs_n < N into a deferred CTA-uniform full-tile
// guard: tileBase + BM/BN <= extent. The varying side must be exactly one unit
// range, so a sparse, multiplied or otherwise irregular index is not mistaken
// for a dense tile.
FailureOr<GuardPlanPtr>
analyzeFullNonKTileGuard(Value mask, scf::ForOp loop,
                         ArrayRef<int64_t> expectedSourceShape,
                         int64_t nonKTileSize) {
  Value source = stripToShape(mask, expectedSourceShape);
  if (!hasShape(source.getType(), expectedSourceShape))
    return failure();
  auto compare = source.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt)
    return failure();
  auto upper = compare.getRhs().getDefiningOp<tt::SplatOp>();
  if (!upper || !loop.isDefinedOutsideOfLoop(upper.getSrc()))
    return failure();
  auto index = analyzeOffset(compare.getLhs(), loop);
  if (failed(index) || index->ranges.size() != 1 ||
      index->ranges.front().extent != nonKTileSize ||
      !isConstantExpr(index->ranges.front().coefficient, 1))
    return failure();
  Type elementType =
      cast<RankedTensorType>(compare.getLhs().getType()).getElementType();
  ScalarExprPtr tileBase = index->uniform ? std::move(index->uniform)
                                          : makeConstantExpr(elementType, 0);
  auto plan = std::make_shared<GuardPlan>();
  plan->kind = GuardPlan::Kind::FullTile;
  plan->tileBase = std::move(tileBase);
  plan->extent = upper.getSrc();
  plan->tileSize = nonKTileSize;
  return plan;
}

// Prove the complete payload mask without generating IR. K-tail predicates do
// not need an extra guard because the page model omits the final partial page;
// non-K boundaries become full-tile guards, and CTA-uniform predicates are
// carried through directly. Any other mask form disables only this operand.
FailureOr<GuardPlanPtr> analyzeLoadMask(Value mask, scf::ForOp loop,
                                        Value kExtent,
                                        ArrayRef<int64_t> kMaskShape,
                                        ArrayRef<int64_t> nonKMaskShape,
                                        int64_t nonKTileSize) {
  if (!mask) {
    auto plan = std::make_shared<GuardPlan>();
    plan->kind = GuardPlan::Kind::True;
    return plan;
  }

  while (auto convert = mask.getDefiningOp<ttg::ConvertLayoutOp>())
    mask = convert.getSrc();

  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    auto lhs = analyzeLoadMask(andOp.getLhs(), loop, kExtent, kMaskShape,
                               nonKMaskShape, nonKTileSize);
    auto rhs = analyzeLoadMask(andOp.getRhs(), loop, kExtent, kMaskShape,
                               nonKMaskShape, nonKTileSize);
    if (failed(lhs) || failed(rhs))
      return failure();
    auto plan = std::make_shared<GuardPlan>();
    plan->kind = GuardPlan::Kind::And;
    plan->lhs = std::move(*lhs);
    plan->rhs = std::move(*rhs);
    return plan;
  }

  if (isKBoundaryMask(mask, loop, kExtent, kMaskShape)) {
    auto plan = std::make_shared<GuardPlan>();
    plan->kind = GuardPlan::Kind::True;
    return plan;
  }

  if (auto splat = mask.getDefiningOp<tt::SplatOp>()) {
    if (splat.getSrc().getType().isInteger(1) &&
        loop.isDefinedOutsideOfLoop(splat.getSrc())) {
      auto plan = std::make_shared<GuardPlan>();
      plan->kind = GuardPlan::Kind::Scalar;
      plan->scalar = splat.getSrc();
      return plan;
    }
  }

  return analyzeFullNonKTileGuard(mask, loop, nonKMaskShape, nonKTileSize);
}

struct PointerAnalysis {
  Value basePtr;
  OffsetAnalysis offset;
};

// Flatten nested tensor addptrs into one allocation base plus the symbolic
// affine offset. Loop-carried pointers are first rewound to iteration zero.
FailureOr<PointerAnalysis> analyzePointer(Value pointer, scf::ForOp loop) {
  pointer = resolveFirstIterationValue(pointer, loop);
  if (!isa<RankedTensorType>(pointer.getType())) {
    if (loop.isDefinedOutsideOfLoop(pointer))
      return PointerAnalysis{pointer, {}};
    return failure();
  }
  if (auto splat = pointer.getDefiningOp<tt::SplatOp>())
    return analyzePointer(splat.getSrc(), loop);
  if (auto broadcast = pointer.getDefiningOp<tt::BroadcastOp>())
    return analyzePointer(broadcast.getSrc(), loop);
  if (auto convert = pointer.getDefiningOp<ttg::ConvertLayoutOp>())
    return analyzePointer(convert.getSrc(), loop);
  auto addPtr = pointer.getDefiningOp<tt::AddPtrOp>();
  if (!addPtr)
    return failure();
  auto pointerPart = analyzePointer(addPtr.getPtr(), loop);
  auto offsetPart = analyzeOffset(addPtr.getOffset(), loop);
  if (failed(pointerPart) || failed(offsetPart))
    return failure();
  pointerPart->offset.uniform = addExpr(std::move(pointerPart->offset.uniform),
                                        std::move(offsetPart->uniform));
  llvm::append_range(pointerPart->offset.ranges, std::move(offsetPart->ranges));
  inferRangeDimensions(pointerPart->offset, pointer.getType());
  return pointerPart;
}

// Turn one payload load into the semantic GEMM panel needed by the page model.
// There must be exactly one unit range for the K dimension and one for the
// non-K dimension; their coefficients are the only accepted strides. Extra or
// unresolved range terms make the operand ambiguous and therefore ineligible.
FailureOr<WarmupOperandPlan> analyzeWarmupOperand(tt::LoadOp load,
                                                  scf::ForOp loop,
                                                  int64_t nonKTileSize,
                                                  unsigned nonKDimension) {
  Value firstPtr = resolveFirstIterationValue(load.getPtr(), loop);
  auto pointer = analyzePointer(firstPtr, loop);
  auto loadType = dyn_cast<RankedTensorType>(load.getType());
  if (failed(pointer) || !loadType)
    return failure();

  unsigned kDimension = 1 - nonKDimension;
  RangeTerm *nonKTerm = nullptr;
  RangeTerm *kTerm = nullptr;
  for (RangeTerm &term : pointer->offset.ranges) {
    if (term.dimension == static_cast<int>(nonKDimension) && !nonKTerm)
      nonKTerm = &term;
    else if (term.dimension == static_cast<int>(kDimension) && !kTerm)
      kTerm = &term;
    else
      return failure();
  }
  if (!nonKTerm || !kTerm || nonKTerm->extent != nonKTileSize ||
      kTerm->extent != loadType.getShape()[kDimension])
    return failure();

  unsigned elementBits = loadType.getElementTypeBitWidth();
  if ((elementBits != kBit8Width && elementBits != kBit16Width) ||
      !nonKTerm->coefficient || !kTerm->coefficient) {
    LDBG("cannot recover unique affine GEMM operand: element-bits="
         << elementBits << " pointer=" << firstPtr);
    return failure();
  }
  return WarmupOperandPlan{pointer->basePtr,      pointer->offset.uniform,
                           nonKTerm->coefficient, kTerm->coefficient,
                           nonKTileSize,          elementBits / kBitsPerByte};
}

// Materialization is intentionally separate from analysis. Once every proof
// succeeds, rebuild only the scalar tile-base arithmetic required by warmup.
WarmupOperand materializeWarmupOperand(OpBuilder &builder, Location loc,
                                       const WarmupOperandPlan &plan) {
  Value tileBase = plan.basePtr;
  if (plan.tileBaseOffset) {
    Value offset = materializeExpr(builder, loc, plan.tileBaseOffset);
    tileBase = tt::AddPtrOp::create(builder, loc, tileBase.getType(), tileBase,
                                    offset);
  }
  return WarmupOperand{tileBase,
                       materializeExpr(builder, loc, plan.nonKStrideElements),
                       materializeExpr(builder, loc, plan.kStrideElements),
                       plan.nonKTileSize, plan.elementBytes};
}

// Lower a proven mask plan to the scalar condition surrounding UTCWarmupOp.
Value materializeGuard(OpBuilder &builder, Location loc,
                       const GuardPlanPtr &plan) {
  switch (plan->kind) {
  case GuardPlan::Kind::True:
    return arith::ConstantIntOp::create(builder, loc, 1, 1);
  case GuardPlan::Kind::Scalar:
    return plan->scalar;
  case GuardPlan::Kind::And:
    return arith::AndIOp::create(builder, loc,
                                 materializeGuard(builder, loc, plan->lhs),
                                 materializeGuard(builder, loc, plan->rhs));
  case GuardPlan::Kind::FullTile: {
    Value tileBase =
        asI64(builder, loc, materializeExpr(builder, loc, plan->tileBase));
    Value extent = asI64(builder, loc, plan->extent);
    assert(tileBase && extent && "validated mask indices must be integers");
    Value tileEnd = arith::AddIOp::create(
        builder, loc, tileBase,
        arith::ConstantIntOp::create(builder, loc, plan->tileSize, 64));
    return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                 tileEnd, extent);
  }
  }
  llvm_unreachable("unknown deferred guard");
}

Value stripScalarSplat(Value value) {
  while (true) {
    if (auto splat = value.getDefiningOp<tt::SplatOp>()) {
      value = splat.getSrc();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    return value;
  }
}

bool isConstantInt(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.getSExtValue() == expected;
}

bool isConstantIntLike(Value value, int64_t expected) {
  value = stripScalarSplat(value);
  APInt scalar;
  if (matchPattern(value, m_ConstantInt(&scalar)))
    return scalar.getSExtValue() == expected;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                        : DenseIntElementsAttr{};
  return dense && dense.isSplat() &&
         dense.getSplatValue<APInt>().getSExtValue() == expected;
}

// Verify that the loop-carried payload pointer advances by exactly one BK
// panel. This connects the first-iteration affine proof to every later load.
bool hasCanonicalKAdvance(tt::LoadOp load, scf::ForOp loop, int64_t blockK) {
  auto iterArg = dyn_cast<BlockArgument>(load.getPtr());
  if (!iterArg || iterArg.getOwner() != loop.getBody() ||
      iterArg.getArgNumber() == 0)
    return false;
  unsigned initIndex = iterArg.getArgNumber() - 1;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (initIndex >= yield.getNumOperands())
    return false;
  auto advance = yield.getOperand(initIndex).getDefiningOp<tt::AddPtrOp>();
  return advance && advance.getPtr() == iterArg &&
         isConstantIntLike(advance.getOffset(), blockK);
}

// Peel only injective integer extensions. Truncation and index casts may lose
// high bits, so accepting them would not prove that the matrix K index is the
// reduction induction variable.
Value stripScalarIntegerExtensions(Value value) {
  while (true) {
    if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
      value = ext.getIn();
      continue;
    }
    if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
      value = ext.getIn();
      continue;
    }
    return value;
  }
}

// A matrix_load keeps one scalar index per matrix dimension instead of a
// loop-carried tensor of pointers. Require its K index to describe exactly the
// current reduction iteration, so the warmup request can safely start at K=0.
bool hasCanonicalMatrixKIndex(tt::MatrixLoadOp load, scf::ForOp loop,
                              unsigned kDimension, int64_t blockK) {
  if (load.getIndices().size() != 2)
    return false;
  Value kIndex = stripScalarIntegerExtensions(load.getIndices()[kDimension]);
  APInt step;
  if (!matchPattern(loop.getStep(), m_ConstantInt(&step)))
    return false;
  if (step.getSExtValue() == blockK)
    return kIndex == loop.getInductionVar();
  if (step.getSExtValue() != 1)
    return false;

  auto multiply = kIndex.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;
  Value lhs = stripScalarIntegerExtensions(multiply.getLhs());
  Value rhs = stripScalarIntegerExtensions(multiply.getRhs());
  return (lhs == loop.getInductionVar() && isConstantIntLike(rhs, blockK)) ||
         (rhs == loop.getInductionVar() && isConstantIntLike(lhs, blockK));
}

// Matrix-load operands expose the GEMM base, matrix strides and tile indices
// directly. Rebuild the first tile address from the non-K index; the K index
// has already been proven to start at zero by hasCanonicalMatrixKIndex.
FailureOr<WarmupOperandPlan>
analyzeMatrixWarmupOperand(tt::MatrixLoadOp load, scf::ForOp loop,
                           Value kExtent, int64_t nonKTileSize,
                           unsigned nonKDimension) {
  auto loadType = dyn_cast<RankedTensorType>(load.getType());
  if (!loadType || load.getShape().size() != 2 ||
      load.getStrides().size() != 2 || load.getIndices().size() != 2)
    return failure();

  unsigned kDimension = 1 - nonKDimension;
  if (load.getShape()[kDimension] != kExtent ||
      !loop.isDefinedOutsideOfLoop(load.getBase()) ||
      !loop.isDefinedOutsideOfLoop(load.getStrides()[nonKDimension]) ||
      !loop.isDefinedOutsideOfLoop(load.getStrides()[kDimension]) ||
      !loop.isDefinedOutsideOfLoop(load.getIndices()[nonKDimension]))
    return failure();

  auto nonKIndex = analyzeOffset(load.getIndices()[nonKDimension], loop);
  if (failed(nonKIndex) || !nonKIndex->ranges.empty() ||
      !nonKIndex->uniform)
    return failure();

  ScalarExprPtr nonKStride = makeValueExpr(load.getStrides()[nonKDimension]);
  ScalarExprPtr kStride = makeValueExpr(load.getStrides()[kDimension]);
  ScalarExprPtr index = extendIntegerExprTo(std::move(nonKIndex->uniform),
                                           nonKStride->type);
  if (!index)
    return failure();
  ScalarExprPtr tileBaseOffset =
      multiplyExpr(std::move(index), nonKStride);

  unsigned elementBits = loadType.getElementTypeBitWidth();
  if (elementBits != kBit8Width && elementBits != kBit16Width)
    return failure();
  return WarmupOperandPlan{load.getBase(), std::move(tileBaseOffset),
                           std::move(nonKStride), std::move(kStride),
                           nonKTileSize, elementBits / kBitsPerByte};
}

// Matrix-load boundary_check is expressed as dimension numbers rather than a
// tensor mask. K-tail safety is already provided by the exact runtime K and
// complete-page truncation. A checked M/N dimension needs the same full-tile
// guard used for tt.load.
FailureOr<GuardPlanPtr>
analyzeMatrixBoundaryGuard(tt::MatrixLoadOp load, scf::ForOp loop,
                           int64_t nonKTileSize,
                           unsigned nonKDimension) {
  bool checkNonKBoundary = false;
  for (int32_t dimension : load.getBoundaryCheck()) {
    if (dimension < 0 || dimension >= 2)
      return failure();
    checkNonKBoundary |= dimension == static_cast<int32_t>(nonKDimension);
  }

  auto plan = std::make_shared<GuardPlan>();
  if (!checkNonKBoundary) {
    plan->kind = GuardPlan::Kind::True;
    return plan;
  }

  auto index = analyzeOffset(load.getIndices()[nonKDimension], loop);
  if (failed(index) || !index->ranges.empty() || !index->uniform ||
      !loop.isDefinedOutsideOfLoop(load.getShape()[nonKDimension]))
    return failure();
  plan->kind = GuardPlan::Kind::FullTile;
  plan->tileBase = std::move(index->uniform);
  plan->extent = load.getShape()[nonKDimension];
  plan->tileSize = nonKTileSize;
  return plan;
}

// Compute the inclusive address span of one K-major GEMM panel and convert it
// to complete 2 MiB pages. Partial final pages are deliberately omitted so the
// extra dword requests remain inside the proven panel; hardware needs at most
// the first 16 translations for this optimization.
Value buildPageCount(OpBuilder &builder, Location loc,
                     const WarmupOperand &operand, Value kExtent) {
  Value k64 = asI64(builder, loc, kExtent);
  Value nonKStride = asI64(builder, loc, operand.nonKStrideElements);
  Value kStride = asI64(builder, loc, operand.kStrideElements);
  Value nonKSizeMinusOne =
      arith::ConstantIntOp::create(builder, loc, operand.nonKTileSize - 1, 64);
  Value kMinusOne = arith::SubIOp::create(
      builder, loc, k64, arith::ConstantIntOp::create(builder, loc, 1, 64));
  Value bytes =
      arith::ConstantIntOp::create(builder, loc, operand.elementBytes, 64);
  Value page = arith::ConstantIntOp::create(builder, loc, kUTCPageBytes, 64);
  // Inclusive affine span of one K-major GEMM operand tile. nonKTileSize is
  // BM for A and BN for B; nonKStride is stride_am or stride_bn:
  //   (nonKTileSize - 1) * nonKStride + (K - 1) * kStride + 1.
  // Non-unit K strides are rejected before this page model is built.
  Value operandSpanElements = arith::AddIOp::create(
      builder, loc,
      arith::AddIOp::create(
          builder, loc,
          arith::MulIOp::create(builder, loc, nonKSizeMinusOne, nonKStride),
          arith::MulIOp::create(builder, loc, kMinusOne, kStride)),
      arith::ConstantIntOp::create(builder, loc, 1, 64));
  Value operandSpanBytes =
      arith::MulIOp::create(builder, loc, operandSpanElements, bytes);
  Value fullPages =
      arith::DivUIOp::create(builder, loc, operandSpanBytes, page);
  Value maxPages = arith::ConstantIntOp::create(
      builder, loc, kMaxUTCWarmupPages, 64);
  Value exceedsLimit = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ugt, fullPages, maxPages);
  return arith::SelectOp::create(builder, loc, exceedsLimit, maxPages,
                                 fullPages);
}

Value splat(OpBuilder &builder, Location loc, Value scalar,
            RankedTensorType type) {
  return tt::SplatOp::create(builder, loc, type, scalar);
}

// Lanes [0, pages) touch distinct base-relative 2 MiB offsets. Other lanes
// repeat page zero so every hardware thread still issues one dword request,
// matching the backend's one-result-per-thread lowering contract.
Value createWarmupRequest(OpBuilder &builder, Location loc,
                          const WarmupOperand &operand, Value pages, Value lane,
                          RankedTensorType i32TensorType,
                          IntegerAttr warmupId) {
  Value pages32 = arith::TruncIOp::create(builder, loc, builder.getI32Type(),
                                          pages);
  Value pagesTensor = splat(builder, loc, pages32, i32TensorType);
  Value hasUniquePage = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, lane, pagesTensor);
  Value zeroTensor = splat(
      builder, loc, arith::ConstantIntOp::create(builder, loc, 0, 32),
      i32TensorType);
  // Let lanes [0, pages) touch distinct 2 MiB pages. The remaining lanes
  // repeat page zero, avoiding a dynamic modulo while still issuing one
  // request per hardware thread. The enclosing scalar guard guarantees
  // pages > 0, and buildPageCount caps pages at kMaxUTCWarmupPages.
  Value pageIndex = arith::SelectOp::create(builder, loc, hasUniquePage, lane,
                                            zeroTensor);
  Value pageBytes =
      splat(builder, loc,
            arith::ConstantIntOp::create(builder, loc, kUTCPageBytes, 32),
            i32TensorType);
  Value byteOffsets = arith::MulIOp::create(builder, loc, pageIndex, pageBytes);

  Type i32Ptr = tt::PointerType::get(
      builder.getI32Type(), tt::getAddressSpace(operand.basePtr.getType()));
  Value dwordBase =
      tt::BitcastOp::create(builder, loc, i32Ptr, operand.basePtr);
  auto warmup = tta::UTCWarmupOp::create(builder, loc, i32TensorType, dwordBase,
                                         byteOffsets);
  warmup->setAttr(kUTCWarmupIdAttr, warmupId);
  return warmup.getResult();
}

// The footprint and boundary conditions are runtime values for dynamic GEMMs.
// Keep the request itself conditional while returning a uniform result shape
// that can stay live across the payload loop in either branch.
Value createGuardedWarmupRequest(OpBuilder &builder, Location loc,
                                 Value condition,
                                 const WarmupOperand &operand, Value pages,
                                 Value lane, RankedTensorType i32TensorType,
                                 IntegerAttr warmupId) {
  auto ifOp = scf::IfOp::create(builder, loc, i32TensorType, condition,
                                /*withElseRegion=*/true);
  OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
  Value request = createWarmupRequest(thenBuilder, loc, operand, pages, lane,
                                      i32TensorType, warmupId);
  scf::YieldOp::create(thenBuilder, loc, request);

  OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
  Value zero = splat(
      elseBuilder, loc,
      arith::ConstantIntOp::create(elseBuilder, loc, 0, 32), i32TensorType);
  scf::YieldOp::create(elseBuilder, loc, zero);
  return ifOp.getResult(0);
}

// Analyze A/B independently, then commit all IR changes for the operands that
// passed. For tt.load, the private ID also selects its payload descriptor
// stride. Matrix loads receive the translation request only. Consumers after
// the loop prevent reuse of an outstanding destination VGPR in either case.
LogicalResult insertWarmup(Candidate candidate, int64_t warmupIdValue) {
  Location loc = candidate.loop.getLoc();
  Value kExtent = candidate.kExtent;

  SmallVector<PreparedWarmupOperand, kNumOperands> prepared;
  auto prepare = [&](unsigned index, int64_t nonKTileSize,
                     unsigned nonKDimension, ArrayRef<int64_t> kMaskShape,
                     ArrayRef<int64_t> nonKMaskShape) {
    Operation *payloadLoad = candidate.loads[index];
    if (!payloadLoad) {
      LDBG("skip operand " << index << ": no unique source load");
      return;
    }

    FailureOr<WarmupOperandPlan> operand = failure();
    FailureOr<GuardPlanPtr> maskGuard = failure();
    if (auto load = dyn_cast<tt::LoadOp>(payloadLoad)) {
      operand = analyzeWarmupOperand(load, candidate.loop, nonKTileSize,
                                     nonKDimension);
      if (succeeded(operand) &&
          !hasCanonicalKAdvance(load, candidate.loop, candidate.blockK)) {
        LDBG("skip operand " << index
                             << ": non-canonical K pointer advance");
        return;
      }
      maskGuard = analyzeLoadMask(load.getMask(), candidate.loop, kExtent,
                                  kMaskShape, nonKMaskShape, nonKTileSize);
    } else if (auto load = dyn_cast<tt::MatrixLoadOp>(payloadLoad)) {
      unsigned kDimension = 1 - nonKDimension;
      if (!hasCanonicalMatrixKIndex(load, candidate.loop, kDimension,
                                    candidate.blockK)) {
        LDBG("skip operand " << index
                             << ": non-canonical matrix-load K index");
        return;
      }
      operand = analyzeMatrixWarmupOperand(load, candidate.loop, kExtent,
                                           nonKTileSize, nonKDimension);
      maskGuard = analyzeMatrixBoundaryGuard(load, candidate.loop, nonKTileSize,
                                              nonKDimension);
    }
    if (failed(operand)) {
      LDBG("skip operand " << index
                           << ": cannot prove affine GEMM operand tile");
      return;
    }
    // Warm each dot operand independently, but only when its reduction
    // dimension is K-major. Triton specializes integer arguments equal to
    // one, so a unit K stride is statically visible in normal JIT launches.
    // An unknown or non-unit K stride is not a UTC warmup candidate.
    if (!isConstantExpr(operand->kStrideElements, 1)) {
      LDBG("skip operand " << index << ": K stride is not constant one");
      return;
    }
    if (failed(maskGuard)) {
      LDBG("skip operand " << index << ": unsupported boundary mask");
      return;
    }
    prepared.push_back(
        PreparedWarmupOperand{index, payloadLoad, *operand, *maskGuard});
  };
  prepare(kOperandA, candidate.blockM, /*nonKDimension=*/0,
          {1, candidate.blockK}, {candidate.blockM, 1});
  prepare(kOperandB, candidate.blockN, /*nonKDimension=*/1,
          {candidate.blockK, 1}, {1, candidate.blockN});
  if (prepared.empty())
    return failure();

  // Everything above is analysis-only. Start mutating the IR only after at
  // least one operand has passed the complete address, stride, advance and
  // mask proof, so a rejected candidate cannot leave dead helper operations.
  OpBuilder builder(candidate.loop);
  Value k64 = asI64(builder, loc, kExtent);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value minPages = arith::ConstantIntOp::create(
      builder, loc, kMinUTCWarmupPages, 64);
  Value kIsPositive = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, k64, zero);
  auto buildOperandEligibility = [&](Value pages, const WarmupOperand &operand,
                                     Value maskGuard) {
    Value nonKStride = asI64(builder, loc, operand.nonKStrideElements);
    Value kStride = asI64(builder, loc, operand.kStrideElements);
    Value positiveStrides = arith::AndIOp::create(
        builder, loc,
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                              nonKStride, zero),
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, kStride,
                              zero));
    return arith::AndIOp::create(
        builder, loc,
        arith::AndIOp::create(
            builder, loc, kIsPositive,
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                  pages, minPages)),
        arith::AndIOp::create(builder, loc, positiveStrides, maskGuard));
  };

  ModuleOp module = candidate.loop->getParentOfType<ModuleOp>();
  unsigned numWarps = ttg::lookupNumWarps(candidate.loop);
  unsigned waveSize = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int64_t numThreads = static_cast<int64_t>(numWarps) * waveSize;
  auto encoding = ttg::getDefaultBlockedEncoding(
      builder.getContext(), {numThreads}, numWarps, waveSize, /*numCTAs=*/1);
  auto i32TensorType =
      RankedTensorType::get({numThreads}, builder.getI32Type(), encoding);
  Value thread =
      tt::MakeRangeOp::create(builder, loc, i32TensorType, 0, numThreads);
  Value wave = splat(builder, loc,
                     arith::ConstantIntOp::create(builder, loc, waveSize, 32),
                     i32TensorType);
  Value lane = arith::RemUIOp::create(builder, loc, thread, wave);

  SmallVector<Value, kNumOperands> warmupResults;
  for (const PreparedWarmupOperand &entry : prepared) {
    WarmupOperand operand =
        materializeWarmupOperand(builder, loc, entry.operand);
    Value maskGuard = materializeGuard(builder, loc, entry.maskGuard);
    Value pages = buildPageCount(builder, loc, operand, kExtent);
    Value condition = buildOperandEligibility(pages, operand, maskGuard);
    auto warmupId =
        builder.getI64IntegerAttr(warmupIdValue + entry.index);
    warmupResults.push_back(
        createGuardedWarmupRequest(builder, loc, condition, operand, pages,
                                   lane, i32TensorType, warmupId));
    // Matrix loads have a separate descriptor lowering path. Do not attach the
    // payload association attribute: MLS receives warmup without the cache-
    // swizzle stride override requested for ordinary tt.load payloads.
    if (auto load = dyn_cast<tt::LoadOp>(entry.payloadLoad))
      load->setAttr(kUTCWarmupIdAttr, warmupId);
  }

  // The buffer request is asynchronous. Keep its backend-allocated virtual
  // destination register live across the complete payload loop, then consume
  // it with an instruction-free compiler barrier. Reusing a dead destination
  // while the request is outstanding can corrupt live values; waiting here is
  // late enough that it cannot serialize the first matrix loads.
  builder.setInsertionPointAfter(candidate.loop);
  for (Value result : warmupResults)
    tta::UTCWarmupConsumeOp::create(builder, loc, result);
  return success();
}

struct TritonHCUUTCWarmupPass
    : impl::TritonHCUUTCWarmupBase<TritonHCUUTCWarmupPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto arch = getAMDArch(module);
    if (!arch) {
      LDBG("skip: target architecture is missing from the module");
      return;
    }
    if (!tt::AMD::supportsBufferCacheSwizzle(*arch)) {
      LDBG("skip unsupported target " << *arch);
      return;
    }
    bool alreadyApplied = false;
    module.walk([&](tta::UTCWarmupOp) { alreadyApplied = true; });
    if (alreadyApplied) {
      LDBG("skip: UTC warmup pass was already applied");
      return;
    }
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    int64_t nextWarmupId = 0;
    for (scf::ForOp loop : loops) {
      auto candidate = matchCandidate(loop);
      if (succeeded(candidate))
        if (succeeded(insertWarmup(*candidate, nextWarmupId)))
          nextWarmupId += kNumOperands;
    }
  }
};

} // namespace
} // namespace mlir
