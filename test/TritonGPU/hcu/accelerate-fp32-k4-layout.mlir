// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938" | FileCheck %s --check-prefix=LTS
// RUN: not triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938 mmac-layout-force=2" 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938 mmac-layout-force=3" | FileCheck %s --check-prefix=LTS
// On the same target, selecting K4 instead of K8 changes modifier support.
// LTS: #mma = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 4]{{.*}}mmacLayout = 3
// ERROR: error: MMAC layout interleave is not supported by rocdl.mmac.16x16x4.f32
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp32_k4(%a: tensor<64x4xf32, #blocked>, %b: tensor<4x64xf32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #blocked>
    %av = ttg.convert_layout %a : tensor<64x4xf32, #blocked> -> tensor<64x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<4x64xf32, #blocked> -> tensor<4x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c, inputPrecision = ieee : tensor<64x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<4x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}
