// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938 mmac-layout-force=2" | FileCheck %s --check-prefix=K8-LIT
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938 mmac-layout-force=4" | FileCheck %s --check-prefix=K8-BOTH
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946" | FileCheck %s --check-prefix=LTS
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=0" | FileCheck %s --check-prefix=LTS
// RUN: not triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=2" 2>&1 | FileCheck %s --check-prefix=LIT-ERROR
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=3" | FileCheck %s --check-prefix=LTS
// RUN: not triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=4" 2>&1 | FileCheck %s --check-prefix=LIT-LTS-ERROR
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=1" | FileCheck %s --check-prefix=LEGACY
// K8-LIT: #mma = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 8]{{.*}}mmacLayout = 2
// K8-BOTH: #mma = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 8]{{.*}}mmacLayout = 4
// Automatic selection uses a supported LTS encoding. Explicit requests are
// honored if supported and diagnosed otherwise; no silent replacement.
// LIT-ERROR: error: MMAC layout interleave is not supported by rocdl.mmac.16x16x4.f32
// LIT-LTS-ERROR: error: MMAC layout interleave_transpose is not supported by rocdl.mmac.16x16x4.f32
// LTS: #mma = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 4]{{.*}}mmacLayout = 3
// LEGACY: #mma = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 4]{{.*}}mmacLayout = 1
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp32_lts(%a: tensor<64x64xf32, #blocked>, %b: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #blocked>
    %av = ttg.convert_layout %a : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c, inputPrecision = ieee : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}
