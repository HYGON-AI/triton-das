#ifndef TRITONHCU_CONVERSION_PASSES
#define TRITONHCU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateScaleBuffer : Pass<"tritonhcu-allocate-scale-buffer", "mlir::ModuleOp"> {
  let summary = "Assign Scale Buffer rows and record kernel reservation";

  let description = [{
    Validate the fixed Scale Buffer layout against the LDS source and MMAC
    consumer. Assign each amdg.scale_write an 8-row-aligned row_base, preserving
    the layout's A/B SRAM-bank polarity. The resulting storage descriptor is
    consumed by amdg.dot_scaled; allocation does not reselect copy geometry.

    In a single-function, straight-line scope with tracked scale accesses,
    reuse regions whose write-to-dot SSA lifetimes do not overlap. Loops, CFG,
    predicates and untracked accesses conservatively retain allocations. Scale
    membar analysis synchronizes reuse of the same physical rows.

    Record the peak reservation in the i32 module attribute hcu.scale_buffer,
    in 16-row (256-byte) slots. Reject reservations exceeding 32 slots.
    compiler.py forwards this count to the LLVM kernel attribute required by
    the backend/driver; this pass does not perform LLVM instruction lowering.
  }];
  let constructor = "mlir::triton::HCU::createAllocateScaleBuffer()";
  let dependentDialects = [
    "mlir::triton::amdgpu::TritonAMDGPUDialect",
  ];
}

#endif
