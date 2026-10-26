// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" --canonicalize | FileCheck %s
// FP32 K4 supports LTS but has no LIT operand. Check the emitted LTS flag,
// including the explicit legacy path, independently of acceleration.
// CHECK-LABEL: llvm.func internal @supported_lts(
// CHECK: %[[FLAG_3:.*]] = llvm.mlir.constant(true) : i1
// CHECK: rocdl.mmac.16x16x4.f32 {{.*}}, %[[FLAG_3]] : (f32, f32, vector<4xf32>, i32, i1) -> vector<4xf32>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 3}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @supported_lts(%a: tensor<16x4xf32, #dot_a>, %b: tensor<4x16xf32, #dot_b>) -> tensor<16x16xf32, #mma> {
    %c = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<16x4xf32, #dot_a> * tensor<4x16xf32, #dot_b> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----
// CHECK-LABEL: llvm.func internal @supported_legacy(
// CHECK: %[[FLAG_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: rocdl.mmac.16x16x4.f32 {{.*}}, %[[FLAG_1]] : (f32, f32, vector<4xf32>, i32, i1) -> vector<4xf32>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 1}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @supported_legacy(%a: tensor<16x4xf32, #dot_a>, %b: tensor<4x16xf32, #dot_b>) -> tensor<16x16xf32, #mma> {
    %c = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<16x4xf32, #dot_a> * tensor<4x16xf32, #dot_b> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----
// FP64 K4 supports LTS but has no LIT operand. Check the emitted LTS flag,
// including the explicit legacy path, independently of acceleration.
// CHECK-LABEL: llvm.func internal @supported_f64_lts(
// CHECK: %[[FLAG_3:.*]] = llvm.mlir.constant(true) : i1
// CHECK: rocdl.mmac.16x16x4.f64 {{.*}}, %[[FLAG_3]] : (f64, f64, vector<4xf64>, i32, i1) -> vector<4xf64>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = false, elementBitWidth = 64, mmacLayout = 3}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @supported_f64_lts(%a: tensor<16x4xf64, #dot_a>, %b: tensor<4x16xf64, #dot_b>) -> tensor<16x16xf64, #mma> {
    %c = arith.constant dense<0.0> : tensor<16x16xf64, #mma>
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<16x4xf64, #dot_a> * tensor<4x16xf64, #dot_b> -> tensor<16x16xf64, #mma>
    tt.return %d : tensor<16x16xf64, #mma>
  }
}

// -----
// CHECK-LABEL: llvm.func internal @supported_f64_legacy(
// CHECK: %[[FLAG_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK: rocdl.mmac.16x16x4.f64 {{.*}}, %[[FLAG_1]] : (f64, f64, vector<4xf64>, i32, i1) -> vector<4xf64>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = false, elementBitWidth = 64, mmacLayout = 1}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @supported_f64_legacy(%a: tensor<16x4xf64, #dot_a>, %b: tensor<4x16xf64, #dot_b>) -> tensor<16x16xf64, #mma> {
    %c = arith.constant dense<0.0> : tensor<16x16xf64, #mma>
    %d = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<16x4xf64, #dot_a> * tensor<4x16xf64, #dot_b> -> tensor<16x16xf64, #mma>
    tt.return %d : tensor<16x16xf64, #mma>
  }
}
