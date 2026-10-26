// RUN: triton-opt %s -tritongpu-partition-scheduling-hcu | FileCheck %s
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// The stable scale is carried unchanged: dependency traversal must terminate
// even when a region iter_arg yields itself.
// A scaled partial dot is not the loop-carried accumulator. Its pointwise
// reduction and scale decoding must belong to the consumer, not root.
// CHECK-LABEL: tt.func public @post_dot_scales
// CHECK: tt.load {{.*}}ttg.partition = array<i32: 2>
// CHECK: tt.load {{.*}}ttg.partition = array<i32: 2>
// CHECK: tt.load {{.*}}ttg.partition = array<i32: 2>
// CHECK: tt.load {{.*}}ttg.partition = array<i32: 2>
// CHECK: arith.extui {{.*}}ttg.partition = array<i32: 1>
// CHECK: arith.shli {{.*}}ttg.partition = array<i32: 1>
// CHECK: tt.bitcast {{.*}}ttg.partition = array<i32: 1>
// CHECK: tt.dot {{.*}}ttg.partition = array<i32: 1>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: 1>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: 1>
// CHECK: arith.mulf {{.*}}ttg.partition = array<i32: 1>
// CHECK: arith.addf {{.*}}ttg.partition = array<i32: 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true, mmacLayout = 0, tilesPerWarp = [1, 1]}>
#slice_m = #ttg.slice<{dim = 1, parent = #mma}>
#slice_n = #ttg.slice<{dim = 0, parent = #mma}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx946"} {
  tt.func public @post_dot_scales(%A: tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>, %B: tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>, %AS: tensor<64x!tt.ptr<f32>, #slice_m>, %BS: tensor<64x!tt.ptr<i8>, #slice_n>, %end: i32) -> tensor<64x64xf32, #mma> {
    %zero = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %one = arith.constant dense<1.0> : tensor<64x64xf32, #mma>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c23 = arith.constant dense<23> : tensor<64xi32, #slice_n>
    %acc:2 = scf.for %iv = %c0 to %end step %c1 iter_args(%carry = %zero, %stable = %one) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) : i32 {
      %a = tt.load %A : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
      %b = tt.load %B : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
      %sa = tt.load %AS : tensor<64x!tt.ptr<f32>, #slice_m>
      %sb = tt.load %BS : tensor<64x!tt.ptr<i8>, #slice_n>
      %exponent = arith.extui %sb : tensor<64xi8, #slice_n> to tensor<64xi32, #slice_n>
      %bits = arith.shli %exponent, %c23 : tensor<64xi32, #slice_n>
      %sbf = tt.bitcast %bits : tensor<64xi32, #slice_n> -> tensor<64xf32, #slice_n>
      %av = ttg.convert_layout %a : tensor<64x64xf8E4M3FN, #blocked> -> tensor<64x64xf8E4M3FN, #dot_a>
      %bv = ttg.convert_layout %b : tensor<64x64xf8E4M3FN, #blocked> -> tensor<64x64xf8E4M3FN, #dot_b>
      %partial = tt.dot %av, %bv, %zero : tensor<64x64xf8E4M3FN, #dot_a> * tensor<64x64xf8E4M3FN, #dot_b> -> tensor<64x64xf32, #mma>
      %sae = tt.expand_dims %sa {axis = 1 : i32} : tensor<64xf32, #slice_m> -> tensor<64x1xf32, #mma>
      %sab = tt.broadcast %sae : tensor<64x1xf32, #mma> -> tensor<64x64xf32, #mma>
      %scaled_a = arith.mulf %partial, %sab : tensor<64x64xf32, #mma>
      %sbe = tt.expand_dims %sbf {axis = 0 : i32} : tensor<64xf32, #slice_n> -> tensor<1x64xf32, #mma>
      %sbb = tt.broadcast %sbe : tensor<1x64xf32, #mma> -> tensor<64x64xf32, #mma>
      %scaled_b = arith.mulf %scaled_a, %sbb : tensor<64x64xf32, #mma>
      %scaled_invariant = arith.mulf %scaled_b, %stable : tensor<64x64xf32, #mma>
      %next = arith.addf %carry, %scaled_invariant : tensor<64x64xf32, #mma>
      scf.yield %next, %stable : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    } {tt.warp_specialize}
    tt.return %acc#0 : tensor<64x64xf32, #mma>
  }
}
