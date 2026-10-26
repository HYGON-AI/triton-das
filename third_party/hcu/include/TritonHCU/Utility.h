#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_UTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_UTILITY_H_

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
