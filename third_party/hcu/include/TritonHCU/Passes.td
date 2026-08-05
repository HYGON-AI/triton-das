// HCU-specific pass declarations
include "mlir/Pass/PassBase.td"


def TritonHCUApplyFuncHintsAndClean : Pass<"tritonhcu-apply-func-hints-and-clean", "mlir::ModuleOp"> {
  let summary = "Apply known tt.hint.func.* effects and clean tt.hint.* (HCU)";
  let description = [{
    HCU pipeline only, before TTGIR -> LLVMIR.

    For each `tt.func`:
      - `func.args_ptr.no_alias` (explicit true) → `llvm.noalias` on ptr args
      - `func.attr.*` (true) → FuncOp `passthrough` with the LLVM function
        attr name (suffix after `func.attr.`, e.g. `optsize`, `noinline`)
      - then drop remaining `tt.hint.*` on that FuncOp
    Then drop ModuleOp `tt.hint.mod.flag.*` (these attrs live on the module,
    not on FuncOp). This pass does not interpret mod.flag values; any
    consumer must run earlier in the pipeline.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::LLVM::LLVMDialect"];
}

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

def TritonHCUPrepareWdraSplitPlan
    : Pass<"tritonhcu-prepare-wdra-split-plan", "mlir::ModuleOp"> {
  let summary = "Annotate warp-specialized loops with HCU WDRA split plans";
  let options = [
    Option<"numConsumerGroups", "num-consumer-groups", "int32_t",
           /*default=*/"2",
           "Number of MMA consumer groups that will receive split data.">
  ];
}

def TritonHCUConsumerPingpong : Pass<"tritonhcu-consumer-pingpong", "mlir::ModuleOp"> {
  let summary = "Stagger WDRA MMA consumers so they do not contend for matrix units";
  let description = [{
    WDRA OnePTwoC / TwoPTwoC runs two MMA consumer warp groups that otherwise
    lockstep through local_load + dot and compete for the same SIMD matrix
    pipes. This pass mirrors `tritonamdgpu-block-pingpong`:

      * split each consumer K-loop into a Memory cluster (wait / ds_read /
        empty) and a Dot cluster (`tt.dot` guarded by `s_setprio`)
      * insert a shared `rocdl.s.barrier {hcu.consumer_pingpong}` between
        clusters (later lowered to one ebarrier with both MMA groups)
      * offset the two groups like AMD `cond_barrier`: MMA_tail waits once
        before the loop, MMA_main waits once after, so they stay one cluster
        apart (one computes while the other loads)
  }];
  let dependentDialects = [
    "mlir::ROCDL::ROCDLDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonHCUMlsStreamPipeline : Pass<"tritonhcu-mls-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "Pipeline MatrixLoad to shared memory async";
  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];
  let options = [
    Option<"numStages", "num_stages", "int32_t", /*default*/"2", "Number of Pipeline stages">,
    Option<"globalPrefetch", "global_prefetch", "int32_t", /*default*/"0", "Set global prefetch stage count">,
    Option<"useAsyncCopy", "use_async_copy", "bool", /*default*/"false", "Use multi-buffer MLS async copy">,
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

def TritonHCUUpdateAsyncWaitCount : Pass<"tritonhcu-update-async-wait-count", "mlir::ModuleOp"> {
  let summary = "Compute async wait count for HCU async copy and MLS";
  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];
  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">
  ];
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
