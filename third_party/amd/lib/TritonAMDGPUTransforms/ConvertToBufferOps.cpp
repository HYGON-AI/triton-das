#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>
#include <optional>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {
bool verifyNonNegativeByAssumption(Value expr,
                                   const DenseSet<Value> &assumptions) {
  for (Value assume : assumptions) {
    LDBG("Assumption:" << assume);
    if (auto cmpOp = assume.getDefiningOp<arith::CmpIOp>()) {
      bool isGreaterThan = (cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
                            cmpOp.getPredicate() == arith::CmpIPredicate::sgt);
      APInt cst;
      if (isGreaterThan && (cmpOp.getLhs() == expr) &&
          matchPattern(cmpOp.getRhs(), m_ConstantInt(&cst))) {
        return cst.isNonNegative();
      }
    }
    // HGCG: add ConvertToBufferOp with data from memory assumptions support.
    else if (auto blockArg= mlir::dyn_cast<BlockArgument>(assume)) {
      if (blockArg.getOwner()->isEntryBlock() && isa<tt::PointerType>(blockArg.getType())) {
        return true;
      }
    }

  }
  return false;
}

bool verifyNonNegativeExpr(Value expr, const DenseSet<Value> &assumptions) {

  // Check if the expression is contained in any assumption
  if (verifyNonNegativeByAssumption(expr, assumptions)) {
    LDBG("Non negative by assumption");
    return true;
  }

  // Recurse if the operation is defined
  Operation *op = expr.getDefiningOp();
  if (!op)
    return false;

  bool nonNegative =
      llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
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
            // max(a,b) >= 0 iff a>=0 || b>=0
            bool nnLhs = verifyNonNegativeExpr(maxOp.getLhs(), assumptions);
            bool nnRhs = verifyNonNegativeExpr(maxOp.getRhs(), assumptions);
            return nnLhs || nnRhs;
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions);
          })
          .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
              // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
              // OP != sub
              [&](Operation *binOp) {
                bool nnLhs =
                    verifyNonNegativeExpr(binOp->getOperand(0), assumptions);
                bool nnRhs =
                    verifyNonNegativeExpr(binOp->getOperand(1), assumptions);
                return nnLhs && nnRhs;
              })
          .Case<triton::LoadOp>([&](auto loadOp) {
            // HGCG: add ConvertToBufferOp with data from memory assumptions support.
            return verifyNonNegativeExpr(loadOp->getOperand(0), assumptions);
          })
          .Default([&](Operation *op) {
            // Conservatively assume that the expression is negative
            return false;
          });
  return nonNegative;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr, const DenseSet<Value> &assumptions) {
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

  // 3. Check if the offset is non-negative
  if (!verifyNonNegativeExpr(offset, assumptions))
    return false;

  LDBG("Non-negative");
  return true;
}
} // namespace

struct ConvertTritonLoadToBufferLoad
    : public mlir::OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(mlir::MLIRContext *context,
                                DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::LoadOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (op.getCache() != triton::CacheModifier::NONE)
      return failure();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();

      auto bufferLoadOp = rewriter.create<triton::amdgpu::BufferLoadOp>(
          op->getLoc(), op.getType(), basePtr, tensorOffset, maybeMask,
          maybeOther);

      // Propagate `OpIdxAttr` if the currently processed `tt.LoadOp` was
      // labeled it. The attribute needs to be preserved for custom instruction
      // scheduling.
      if (auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
              triton::amdgpu::OpIdxAttr::getMnemonic())) {
        bufferLoadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(),
                              opIdxAttr);
      }
      rewriter.replaceOp(op, bufferLoadOp);

      return success();
    }
    LDBG("Failed to convert: " << op);
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(mlir::MLIRContext *context,
                                  DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (op.getCache() != triton::CacheModifier::NONE)
      return failure();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, maybeMask);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

class TritonAMDGPUConvertToBufferOpsPass
    : public TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {

public:
  TritonAMDGPUConvertToBufferOpsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();
    // Collect assumptions in the function
    DenseSet<Value> assumptions;
    m.walk([&](LLVM::AssumeOp op) {
      if (op->getOperand(0).getDefiningOp<arith::CmpIOp>()) {
        assumptions.insert(op->getOperand(0));

        /*  HGCG: add ConvertToBufferOp with data from memory assumptions support.
         *  This is a trick method to support addPtr with offset from memory load.
         *        In triton kernel if user write below code, it means that the
         *  sorted_token_ids_ptr is a pointer to the memory, and all value in the
         *  memory(like offs_token) is assumed to be non-negative.
         *        Due to the propagation of assume, the a_ptrs is fit for using
         *  buffer load ops.
         *
         *  Eg:
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
        auto cmpIOp = op->getOperand(0).getDefiningOp<arith::CmpIOp>();
        bool isGe = (cmpIOp.getPredicate() == arith::CmpIPredicate::sge);
        if (isGe && cmpIOp->getOperand(0).getDefiningOp<triton::PtrToIntOp>()) {
          auto ptrToIntOp = cmpIOp->getOperand(0).getDefiningOp<triton::PtrToIntOp>();
          if (isa<BlockArgument>(ptrToIntOp->getOperand(0))
              && ptrToIntOp->getBlock()->isEntryBlock()) {
              assumptions.insert(ptrToIntOp->getOperand(0));
          }
        }
      }
    });
    LDBG("Number of assumptions found: " << assumptions.size());

    patterns.add<ConvertTritonLoadToBufferLoad>(context, assumptions);
    patterns.add<ConvertTritonStoreToBufferStore>(context, assumptions);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUConvertToBufferOpsPass() {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>();
}
