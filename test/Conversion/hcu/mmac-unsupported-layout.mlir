// RUN: not triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" 2>&1 | FileCheck %s
// An already-encoded dot can bypass layout selection. Lowering must reject
// unsupported LIT rather than omit its bit and silently miscompile the dot.
// CHECK: error: MMAC layout interleave is not supported by rocdl.mmac.16x16x4.f32
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 2}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @invalid_lit(%a: tensor<16x4xf32, #dot_a>, %b: tensor<4x16xf32, #dot_b>) -> tensor<16x16xf32, #mma> {
    %c = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<16x4xf32, #dot_a> * tensor<4x16xf32, #dot_b> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----
// An already-encoded dot can bypass layout selection. Lowering must reject
// unsupported LIT rather than omit its bit and silently miscompile the dot.
// CHECK: error: MMAC layout interleave is not supported by rocdl.mmac.16x16x4.f64
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = false, elementBitWidth = 64, mmacLayout = 2}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @invalid_f64_lit(%a: tensor<16x4xf64, #dot_a>, %b: tensor<4x16xf64, #dot_b>) -> tensor<16x16xf64, #mma> {
    %c = arith.constant dense<0.0> : tensor<16x16xf64, #mma>
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<16x4xf64, #dot_a> * tensor<4x16xf64, #dot_b> -> tensor<16x16xf64, #mma>
    tt.return %d : tensor<16x16xf64, #mma>
  }
}
