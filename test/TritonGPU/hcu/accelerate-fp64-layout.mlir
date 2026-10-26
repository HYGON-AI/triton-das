// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946" | FileCheck %s --check-prefix=LEGACY
// RUN: not triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=2" 2>&1 | FileCheck %s --check-prefix=ERROR
// Preserve the existing FP64 accumulator-layout workaround in acceleration.
// LEGACY: #mma = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 4]{{.*}}mmacLayout = 1
// ERROR: error: MMAC layout interleave is not supported by rocdl.mmac.16x16x4.f64
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fp32_lts(%a: tensor<64x4xf64, #blocked>, %b: tensor<4x64xf64, #blocked>) -> tensor<64x64xf64, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf64, #blocked>
    %av = ttg.convert_layout %a : tensor<64x4xf64, #blocked> -> tensor<64x4xf64, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<4x64xf64, #blocked> -> tensor<4x64xf64, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c, inputPrecision = ieee : tensor<64x4xf64, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<4x64xf64, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf64, #blocked>
    tt.return %d : tensor<64x64xf64, #blocked>
  }
}
