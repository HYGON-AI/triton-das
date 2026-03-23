#ifndef TRITONAMDGPU_CONVERSION_PASSES
#define TRITONAMDGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def OptimizeAMDLDSUsage : Pass<"optimize-amd-lds-usage", "mlir::ModuleOp"> {
    let summary = "Minimize LDS usage";
    let constructor = "mlir::triton::AMD::createOptimizeLDSUsagePass(\"\")";

    let options = [
        Option<"targetArch", "target-arch", "std::string", /*default*/"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"customLDSLimit", "lds-limit", "int", /*default*/"0",
               "custom limit of LDS consumption, if not provided, maximum LDS size is used">,
    ];
}

def AllocateAMDGPUSharedMemory : Pass<"allocate-amdgpu-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

def ConvertTritonAMDGPUToLLVM : Pass<"convert-triton-amdgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonAMDGPUToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def ConvertBuiltinFuncToLLVM : Pass<"convert-builtin-func-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Builtin Func to LLVM";
    let constructor = "mlir::triton::createConvertBuiltinFuncToLLVMPass(/*ftz=*/true)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];

    let options = [
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def TritonAMDGPUInsertInstructionSchedHints : Pass<"triton-amdgpu-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Insert instruction scheduling hints after the dot ops in the main loop";
    let constructor = "mlir::triton::createTritonAMDGPUInsertInstructionSchedHintsPass(/*variant=*/\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::amdgpu::TritonAMDGPUDialect"];

    let options = [
        Option<"variant", "variant", "std::string", /*default*/"\"none\"",
               "instruction scheduling variant">,
    ];
}

def TritonAMDGPULowerInstructionSchedHints : Pass<"triton-amdgpu-lower-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Lower instruction scheduling hints to LLVM intrinsics";
    let constructor = "mlir::triton::createTritonAMDGPULowerInstructionSchedHintsPass(/*arch=*/\"\",/*numStages=*/2)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::amdgpu::TritonAMDGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"numStages", "num_stages", "int32_t", /*default*/"2",
                "number of pipeline stages">,
    ];
}

def AMDGPUConvertWarpSpecializeToLLVM : Pass<"triton-amdgpu-convert-warp-specialize-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower `ttg.warp_specialize` to LLVM";
  let constructor = "mlir::triton::createAMDGPUConvertWarpSpecializeToLLVM(\"\", 4, 8, true, 32, 32, 32)";
  let description = [{
    The `convert-warp-specialize-to-llvm` pass performs codegen for warp
    specialization. It is a function-level transformation that rewrites
    warp-specialized kernels by using shared memory and barriers to communicate
    states between the default warpgroup and the worker warps.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                          "mlir::ROCDL::ROCDLDialect",
                          "mlir::triton::amdgpu::TritonAMDGPUDialect"];
  let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"waspNumLoadWarps", "wasp_num_load_warps", "int", /*default*/"4",
               "total number of warps to do load">,
        Option<"waspNumMmaWarps", "wasp_num_mma_warps", "int", /*default*/"8",
               "total number of warps to do mma">,
        Option<"wdraEnabled", "wdra-enabled", "bool", /*default*/"true",
               "flag to indicate wdra is enabled">,
        Option<"wdraNumLoadRegs", "wdra_num_load_regs", "int", /*default*/"0",
               "max number of registers allocated to per wave in load_partition">,
        Option<"wdraNumMmaRegsMain", "wdra_num_mma_regs_main", "int", /*default*/"0",
               "max number of registers allocated to per wave in main mma partition">,
        Option<"wdraNumMmaRegsTail", "wdra_num_mma_regs_tail", "int", /*default*/"",
               "max number of registers allocated to per wave in tail mma partition; omit to leave unset (e.g. inherit main mma or backend default)">,
    ];
}

#endif
