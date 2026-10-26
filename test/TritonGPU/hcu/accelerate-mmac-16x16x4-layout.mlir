// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946" | FileCheck %s --check-prefix=DEFAULT
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=2" | FileCheck %s --check-prefix=FORCE2
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=3" | FileCheck %s --check-prefix=FORCE3
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=4" | FileCheck %s --check-prefix=FORCE4
//
// The 16x16x4 HCU MMAC instructions (mmac_f32_16x16x4f32 /
// mmac_f64_16x16x4f64) have no interleave ("lit") hardware mode: their C
// fragment only distinguishes LEGACY (lts=0) from TRANSPOSE (lts=1).
// resolveMmacLayout must therefore clamp the not-physically-realizable
// modes at encoding creation (INTERLEAVE -> LEGACY,
// INTERLEAVE_TRANSPOSE -> TRANSPOSE), no matter whether the requested mode
// came from the default mapping or from mmac-layout-force, so the encoding
// never names a mode the instruction cannot produce. Instructions with K>=8
// keep every mode unchanged.
//
// gfx946 has the MMAC_LAYOUT feature, so both the default isTransposed
// mapping and mmac-layout-force apply.
//
// MmacLayout values: 0=MFMA, 1=LEGACY, 2=INTERLEAVE, 3=TRANSPOSE,
// 4=INTERLEAVE_TRANSPOSE.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // f32 x f32 with a K=4 tile selects mmac_f32_16x16x4f32 on gfx946 (no
  // MMAC_F32_K8 feature): the interleave bit is clamped away.
  tt.func public @fp32_k4(%a: tensor<16x4xf32, #blocked_a>, %b: tensor<4x16xf32, #blocked_b>) -> tensor<16x16xf32, #blocked_d> {
    %c = arith.constant dense<0.0> : tensor<16x16xf32, #blocked_d>
    %av = ttg.convert_layout %a : tensor<16x4xf32, #blocked_a> -> tensor<16x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked_d, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<4x16xf32, #blocked_b> -> tensor<4x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked_d, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<16x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked_d, kWidth = 0}>> * tensor<4x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked_d, kWidth = 0}>> -> tensor<16x16xf32, #blocked_d>
    tt.return %d : tensor<16x16xf32, #blocked_d>
  }

  // f16 x f16 with a K=16 tile selects mmac_f32_16x16x16f16: K>=8 supports
  // all modes, so the clamp must not over-demote (guard against clamping on
  // any 16x16 instruction rather than on the K=4 ones).
  tt.func public @f16_k16(%a: tensor<16x16xf16, #blocked_a>, %b: tensor<16x16xf16, #blocked_b>) -> tensor<16x16xf32, #blocked_d> {
    %c = arith.constant dense<0.0> : tensor<16x16xf32, #blocked_d>
    %av = ttg.convert_layout %a : tensor<16x16xf16, #blocked_a> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_d, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<16x16xf16, #blocked_b> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_d, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_d, kWidth = 0}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_d, kWidth = 0}>> -> tensor<16x16xf32, #blocked_d>
    tt.return %d : tensor<16x16xf32, #blocked_d>
  }
}

// The pass rewrites each dot's accumulator to a #ttg.amd_mfma encoding that
// carries the resolved mmacLayout. Check the two module-scope definitions in
// print order: the 16x16x4 one (fp32_k4) first, then the 16x16x16 one
// (f16_k16).
//
// 16x16x4: the default mapping resolves to INTERLEAVE (2) here and is
// clamped to LEGACY (1); FORCE2/3/4 request 2/3/4 and resolve to
// 1/3/3 (TRANSPOSE is realizable and passes through,
// INTERLEAVE_TRANSPOSE is clamped to TRANSPOSE).
// DEFAULT: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 1}>
// FORCE2: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 1}>
// FORCE3: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 3}>
// FORCE4: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 3}>
//
// 16x16x16: all modes are physically realizable, so the request is kept
// unchanged (default INTERLEAVE = 2).
// DEFAULT: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2}>
// FORCE2: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2}>
// FORCE3: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 3}>
// FORCE4: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 4}>
