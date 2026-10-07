#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUTOLLVM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::triton::HCU {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateScaleBuffer();

#define GEN_PASS_REGISTRATION
#include "TritonHCUToLLVM/Passes.h.inc"

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUTOLLVM_PASSES_H_
