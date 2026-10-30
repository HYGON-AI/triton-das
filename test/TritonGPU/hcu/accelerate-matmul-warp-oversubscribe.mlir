// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946" | FileCheck %s --check-prefix=M1 --check-prefix=M2 --check-prefix=M3
//
// When num_warps exceeds the number of output tiles a warp partition can
// give each warp a unique tile of, the surplus warps re-issue the MMAC
// tiles their siblings already compute (aliasing).  The fix:
//
//   1. Keep the ds_read_m 32-column panel partition eligible even when
//      there are more warps than 16x32 panels.
//   2. Over-subscribed dots fill N panels exactly and put surplus warps
//      on M, minimizing replication without duplicating B panels.
//
// Scenarios (MFMA encodings print in first-use order, before the modules):
//   M1 64x32 / 8 warps : [2, 4] + [1, 2] replicated 4x; [8, 1] +
//      [1, 2] replicates 2x and retains matrix-read staging.
//   M2 64x32 / 4 warps : panel fits exactly, keeps [4, 1] + [1, 2]
//      and matrix-read staging (the production tuned layout).
//   M3 32x64 / 8 warps : keeps [4, 2] + [1, 2] and matrix-read staging;
//      2x replication is unavoidable with eight warps and four panels.
// M1: #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
// M2: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
// M3: #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
// M1: "ttg.num-warps" = 8 : i32
// M1-NEXT: tt.func public @gdn_fwd_h_8_warps
// M1: ttg.local_load
// M2: "ttg.num-warps" = 4 : i32
// M2-NEXT: tt.func public @gdn_fwd_h_4_warps
// M2: ttg.local_load
// M3: "ttg.num-warps" = 8 : i32
// M3-NEXT: tt.func public @panel_over_subscribed_matrix_read
// M3: ttg.local_load

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked_c = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @gdn_fwd_h_8_warps(%a: tensor<64x64xf16, #blocked_a>, %B: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b0: i32 {tt.divisibility = 16 : i32}) -> tensor<64x32xf32, #blocked_c> {
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_b}>>
    %offs_n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>>
    %rm = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_b}>> -> tensor<64x1xi32, #blocked_b>
    %rn = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>> -> tensor<1x32xi32, #blocked_b>
    %sm = tt.splat %stride_b0 : i32 -> tensor<64x1xi32, #blocked_b>
    %m0 = arith.muli %rm, %sm : tensor<64x1xi32, #blocked_b>
    %bfull = tt.splat %B : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked_b>
    %m0b = tt.broadcast %m0 : tensor<64x1xi32, #blocked_b> -> tensor<64x32xi32, #blocked_b>
    %rn2 = tt.broadcast %rn : tensor<1x32xi32, #blocked_b> -> tensor<64x32xi32, #blocked_b>
    %boff = arith.addi %m0b, %rn2 : tensor<64x32xi32, #blocked_b>
    %bptr = tt.addptr %bfull, %boff : tensor<64x32x!tt.ptr<f16>, #blocked_b>, tensor<64x32xi32, #blocked_b>
    %b = tt.load %bptr : tensor<64x32x!tt.ptr<f16>, #blocked_b>
    %c = arith.constant dense<0.0> : tensor<64x32xf32, #blocked_c>
    %av = ttg.convert_layout %a : tensor<64x64xf16, #blocked_a> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_c, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<64x32xf16, #blocked_b> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_c, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_c, kWidth = 0}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_c, kWidth = 0}>> -> tensor<64x32xf32, #blocked_c>
    tt.return %d : tensor<64x32xf32, #blocked_c>
  }
}

#blocked_a2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_c2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @gdn_fwd_h_4_warps(%a: tensor<64x64xf16, #blocked_a2>, %B: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b0: i32 {tt.divisibility = 16 : i32}) -> tensor<64x32xf32, #blocked_c2> {
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_b2}>>
    %offs_n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_b2}>>
    %rm = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_b2}>> -> tensor<64x1xi32, #blocked_b2>
    %rn = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_b2}>> -> tensor<1x32xi32, #blocked_b2>
    %sm = tt.splat %stride_b0 : i32 -> tensor<64x1xi32, #blocked_b2>
    %m0 = arith.muli %rm, %sm : tensor<64x1xi32, #blocked_b2>
    %bfull = tt.splat %B : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked_b2>
    %m0b = tt.broadcast %m0 : tensor<64x1xi32, #blocked_b2> -> tensor<64x32xi32, #blocked_b2>
    %rn2 = tt.broadcast %rn : tensor<1x32xi32, #blocked_b2> -> tensor<64x32xi32, #blocked_b2>
    %boff = arith.addi %m0b, %rn2 : tensor<64x32xi32, #blocked_b2>
    %bptr = tt.addptr %bfull, %boff : tensor<64x32x!tt.ptr<f16>, #blocked_b2>, tensor<64x32xi32, #blocked_b2>
    %b = tt.load %bptr : tensor<64x32x!tt.ptr<f16>, #blocked_b2>
    %c = arith.constant dense<0.0> : tensor<64x32xf32, #blocked_c2>
    %av = ttg.convert_layout %a : tensor<64x64xf16, #blocked_a2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_c2, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<64x32xf16, #blocked_b2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_c2, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_c2, kWidth = 0}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_c2, kWidth = 0}>> -> tensor<64x32xf32, #blocked_c2>
    tt.return %d : tensor<64x32xf32, #blocked_c2>
  }
}

#blocked_a3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked_b3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked_c3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @panel_over_subscribed_matrix_read(%a: tensor<32x64xf16, #blocked_a3>, %B: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %stride_b0: i32 {tt.divisibility = 16 : i32}) -> tensor<32x64xf32, #blocked_c3> {
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_b3}>>
    %offs_n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_b3}>>
    %rm = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_b3}>> -> tensor<64x1xi32, #blocked_b3>
    %rn = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_b3}>> -> tensor<1x64xi32, #blocked_b3>
    %sm = tt.splat %stride_b0 : i32 -> tensor<64x1xi32, #blocked_b3>
    %m0 = arith.muli %rm, %sm : tensor<64x1xi32, #blocked_b3>
    %bfull = tt.splat %B : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked_b3>
    %m0b = tt.broadcast %m0 : tensor<64x1xi32, #blocked_b3> -> tensor<64x64xi32, #blocked_b3>
    %rn2 = tt.broadcast %rn : tensor<1x64xi32, #blocked_b3> -> tensor<64x64xi32, #blocked_b3>
    %boff = arith.addi %m0b, %rn2 : tensor<64x64xi32, #blocked_b3>
    %bptr = tt.addptr %bfull, %boff : tensor<64x64x!tt.ptr<f16>, #blocked_b3>, tensor<64x64xi32, #blocked_b3>
    %b = tt.load %bptr : tensor<64x64x!tt.ptr<f16>, #blocked_b3>
    %c = arith.constant dense<0.0> : tensor<32x64xf32, #blocked_c3>
    %av = ttg.convert_layout %a : tensor<32x64xf16, #blocked_a3> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_c3, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<64x64xf16, #blocked_b3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_c3, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked_c3, kWidth = 0}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked_c3, kWidth = 0}>> -> tensor<32x64xf32, #blocked_c3>
    tt.return %d : tensor<32x64xf32, #blocked_c3>
  }
}
