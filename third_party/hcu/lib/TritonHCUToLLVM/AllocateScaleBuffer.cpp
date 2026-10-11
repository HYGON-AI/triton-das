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
  if (!dotOp.getAScale() && !dotOp.getBScale())
    return 0;
  constexpr int64_t kRowsPerWave64ByteCopy = 4;
  constexpr int64_t kElementsPerScaleGroup = 32;
  auto mma = cast<AMDMfmaEncodingAttr>(dotOp.getType().getEncoding());
  int64_t instrK = mma.getInstrShape()[2];
  // Each M16/N16 scale tile covers instrK / 32 scale columns, while a full
  // Wave64 byte-mode copy writes 4 buffer rows (including repeated values).
  // Rows per scale column = 4 / (instrK / 32) = (4 * 32) / instrK:
  // 2 for K64 and 4 for K32. This is allocation footprint, not MMAC read size.
  int64_t rowsPerScale =
      (kRowsPerWave64ByteCopy * kElementsPerScaleGroup) / instrK;
  int64_t totalRows = 0;
  for (auto scale : {dotOp.getAScale(), dotOp.getBScale()}) {
    if (!scale)
      continue;
    auto shape = scale.getType().getShape();
    totalRows += (shape[0] / 16) * shape[1] * rowsPerScale;
  }

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

    if (numRows) {
      int64_t numSlots =
          llvm::divideCeil(numRows, kScaleBufferSlotRows);

      assert(numSlots <= 32 && "Problem size of tt.dot_scaled exceeds the maximum scale buffer limit (32 slots).");

      moduleOp->setAttr(
          kScaleBufferAttrName,
          IntegerAttr::get(IntegerType::get(ctx, 32),
                          static_cast<int32_t>(numSlots)));
    }
  }
};

} // namespace

namespace mlir::triton::HCU {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateScaleBuffer() {
  return std::make_unique<AllocateScaleBufferPass>();
}

}
