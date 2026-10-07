#ifndef TRITONAMDGPU_CONVERSION_PASSES
#define TRITONAMDGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateScaleBuffer : Pass<"tritonhcu-allocate-scale-buffer", "mlir::ModuleOp"> {
  let summary = "Allocate the scale buffer size required by tt.dot_scaled ops";

  let description = [{
    Scans every `tt.dot_scaled` op in the module, sums the scale buffer rows of
    their `a_scale` / `b_scale` operands, converts that total into a
    number of fixed-size 256-byte "scale buffer" slots, and records the
    result as a module attribute named "hcu.scale_buffer" (an `i32`
    integer attribute holding the slot count).
  }];
  let constructor = "mlir::triton::HCU::createAllocateScaleBuffer()";
  let dependentDialects = [
    "mlir::triton::TritonDialect",
  ];
}

#endif
