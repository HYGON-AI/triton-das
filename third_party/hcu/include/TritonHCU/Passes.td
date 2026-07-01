// HCU-specific pass declarations
include "mlir/Pass/PassBase.td"


def TritonHCUAccelerateMatmul : Pass<"tritonhcu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul for HCU";
  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];
  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">,
    Option<"kPack", "kPack",
           "int32_t", /*default*/"1",
           "KWidth / kBase">,
    Option<"mmacLayoutForce", "mmac-layout-force",
           "int32_t", /*default*/"-1",
           "force mmac layout">,
  ];
}

def TritonHCUMlsStreamPipeline : Pass<"tritonhcu-mls-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "Pipeline MatrixLoad to shared memory async";
  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];
  let options = [
    Option<"numStages", "num_stages", "int32_t", /*default*/"2", "Number of Pipeline stages">,
    Option<"globalPrefetch", "global_prefetch", "int32_t", /*default*/"0", "Set global prefetch stage count">,
    Option<"asyncCopySingleBuffer", "async_copy_single_buffer", "int32_t", /*default*/"1", "Use AsyncCopyGlobalToLocal">,
  ];
}

def TritonHCUMlsEncodingInsertion: Pass<"tritonhcu-mls-encoding-insertion", "mlir::ModuleOp"> {
  let summary = "Insert MlsEncoding to the matrix_load instruction";
  let dependentDialects = [];
}

def TritonHCUMlsLowering: Pass<"tritonhcu-mls-lowering", "mlir::ModuleOp"> {
  let summary = "Lower matrix_load to the matrix_load_to_local instruction";
  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];
}

def HCUConvertWarpSpecializeToLLVM : Pass<"triton-hcu-convert-warp-specialize-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower `ttg.warp_specialize` to LLVM";
  let constructor = "mlir::triton::createHCUConvertWarpSpecializeToLLVM(\"\", 4, 8, true, 32, 32, 32)";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::ROCDL::ROCDLDialect", "mlir::triton::amdgpu::TritonAMDGPUDialect"];
  let options = [
    Option<"arch", "arch", "std::string", /*default*/"\"\"", "gfx target device architecture">,
    Option<"waspNumLoadWarps", "wasp_num_load_warps", "int", /*default*/"4", "total number of warps to do load">,
    Option<"waspNumMmaWarps", "wasp_num_mma_warps", "int", /*default*/"8", "total number of warps to do mma">,
    Option<"wdraEnabled", "wdra-enabled", "bool", /*default*/"true", "flag to indicate wdra is enabled">,
    Option<"wdraNumLoadRegs", "wdra_num_load_regs", "int", /*default*/"0", "max registers for load partition">,
    Option<"wdraNumMmaRegsMain", "wdra_num_mma_regs_main", "int", /*default*/"0", "max registers for main mma partition">,
    Option<"wdraNumMmaRegsTail", "wdra_num_mma_regs_tail", "int", /*default*/"0", "max registers for tail mma partition">,
  ];
}

def ConvertTritonHCUToLLVM : Pass<"convert-triton-hcu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonGPU to LLVM (HCU)";
  let constructor = "mlir::triton::createConvertTritonHCUToLLVMPass(\"\", true)";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::ROCDL::ROCDLDialect",
  ];
  let options = [
    Option<"arch", "arch", "std::string", /*default*/"""",
           "gfx target device architecture, e.g., gfx942">,
    Option<"ftz", "ftz", "bool", /*default*/"true",
           "flush denorms for math functions">,
  ];
}
