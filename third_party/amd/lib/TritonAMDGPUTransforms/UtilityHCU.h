#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITYHCU_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITYHCU_H_

#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

using namespace mlir;

mlir::triton::gpu::MmacLayout getMmacLayoutDefault(llvm::StringRef archGen);

mlir::triton::gpu::MmacLayout
getMfmaMappedMmacLayout(bool logicalIsTransposed, bool supportMmacLayout);

FailureOr<std::pair<triton::DotOpInterface, unsigned>>
getDotOpIdxFromMatrixLoad(triton::MatrixLoadOp matrixOp);

#endif
