#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_UTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_UTILITY_H_

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace mlir {
struct MfmaIntrinsic;
} // namespace mlir

namespace mlir::triton::HCU {

// Static K views alias the scale_write allocation. kOffset counts scale
// columns (one per 32 data elements), not physical Scale Buffer rows.
struct ScaleBufferView {
  amdgpu::ScaleWriteOp allocation;
  int64_t kOffset = 0;
};
// Resolve the scale_write root and accumulate K offsets through subslices.
// Return a null allocation if the chain cannot be resolved; slice legality is
// checked by the operation verifiers.
ScaleBufferView getScaleBufferView(Value descriptor);
// Collect dot scale-operand uses through static K views; reject other uses.
FailureOr<SmallVector<OpOperand *>> getScaleDotUses(Value descriptor);

gpu::MmacLayout getMmacLayoutDefault(llvm::StringRef archGen);

// Resolves the MmacLayout recorded in the AMDMfmaEncodingAttr created by the
// accelerate-matmul pass. `forcedOrDefault` is the --mmac-layout-force mode;
// the MFMA value is a sentinel meaning "not forced", in which case the
// default mapping is used. The result is clamped to a physically realizable
// mode for the given instruction shape (see the implementation).
gpu::MmacLayout resolveMmacLayout(gpu::MmacLayout forcedOrDefault,
                                  bool logicalIsTransposed,
                                  bool supportMmacLayout,
                                  llvm::ArrayRef<unsigned> instrShape);

// Diagnose layout bits that the selected instruction cannot represent.
LogicalResult validateMmacLayout(Operation *op, gpu::MmacLayout layout,
                                 const MfmaIntrinsic &intrinsic);

FailureOr<std::pair<DotOpInterface, unsigned>>
getDotOpIdxFromMatrixLoad(MatrixLoadOp matrixOp);

} // namespace mlir::triton::HCU

#endif
