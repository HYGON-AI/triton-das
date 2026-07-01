#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_UTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCU_UTILITY_H_

#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace mlir::triton::HCU {

gpu::MmacLayout getMmacLayoutDefault(llvm::StringRef archGen);

gpu::MmacLayout
getMfmaMappedMmacLayout(bool logicalIsTransposed, bool supportMmacLayout);

FailureOr<std::pair<DotOpInterface, unsigned>>
getDotOpIdxFromMatrixLoad(MatrixLoadOp matrixOp);

} // namespace mlir::triton::HCU

#endif
