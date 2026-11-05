// Modified by Hygon Information Technology Co., Ltd., 2026.

#ifndef TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/CTAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/AttrDefs.h.inc"

namespace mlir::triton::gpu {
// Query base MMAC configuration without discarding the tensor's full encoding.
// Only layouts implementing the wrapper interface are traversed.
AMDMfmaEncodingAttr getUnderlyingMmacEncoding(Attribute encoding);
DotOperandEncodingAttr getUnderlyingDotOperandEncoding(Attribute encoding);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_ATTRIBUTES_H_
