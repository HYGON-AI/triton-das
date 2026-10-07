#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "TritonHCUToLLVM/Passes.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

#define GEN_PASS_DEF_ALLOCATESCALEBUFFER
#include "TritonHCUToLLVM/Passes.h.inc"

// slot = 16 rows x 16 cols = 256 bytes
static constexpr int64_t kScaleBufferSlotRows = 16;

// Module attribute name used to record the number of scale-buffer slots
// required by this module, analogous to "ttg.shared".
static constexpr llvm::StringLiteral kScaleBufferAttrName =
    "hcu.scale_buffer";

static int64_t getScaleBufferNumRows(DotScaledOp dotOp) {
  auto aScaleShape = dotOp.getAScale().getType().getShape();
  auto bScaleShape = dotOp.getBScale().getType().getShape();

  // 16 cols per row. Since repeat mapping of the second half lanes,
  // valid rows *= 2.
  int64_t totalRows = (aScaleShape[0] / 16) * aScaleShape[1] * 2;
  totalRows += (bScaleShape[0] / 16) * bScaleShape[1] * 2;

  return totalRows;
}

struct AllocateScaleBufferPass
    : public impl::AllocateScaleBufferBase<AllocateScaleBufferPass> {
  using AllocateScaleBufferBase::AllocateScaleBufferBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    int64_t numRows = 0;
    moduleOp.walk([&](DotScaledOp op) {
      numRows += getScaleBufferNumRows(op);
    });

    int64_t numSlots =
        llvm::divideCeil(numRows, kScaleBufferSlotRows);

    assert(numSlots <= 32 && "Problem size of tt.dot_scaled exceeds the maximum scale buffer limit (32 slots).");

    moduleOp->setAttr(
        kScaleBufferAttrName,
        IntegerAttr::get(IntegerType::get(ctx, 32),
                         static_cast<int32_t>(numSlots)));
  }
};

} // namespace

namespace mlir::triton::HCU {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateScaleBuffer() {
  return std::make_unique<AllocateScaleBufferPass>();
}

}
