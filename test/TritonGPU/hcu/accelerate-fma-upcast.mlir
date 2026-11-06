// Modified by Hygon Information Technology Co., Ltd., 2026.
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx936" | FileCheck %s
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938" | FileCheck %s
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946" | FileCheck %s

// K=1 requires FMA legalization. Its exact upcasts must not carry rounding;
// narrowing the accumulator back to the requested output must retain RTNE.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @f8E4M3FN_to_f32(
  // CHECK: tt.fp_to_fp %arg0 : tensor<16x1xf8E4M3FN, {{.*}}> -> tensor<16x1xf32, {{.*}}>
  // CHECK: tt.fp_to_fp %arg1 : tensor<1x16xf8E4M3FN, {{.*}}> -> tensor<1x16xf32, {{.*}}>
  // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, {{.*}}>
  // CHECK: tt.return
  tt.func public @f8E4M3FN_to_f32(%a: tensor<16x1xf8E4M3FN, #dot0>, %b: tensor<1x16xf8E4M3FN, #dot1>, %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<16x1xf8E4M3FN, #dot0> * tensor<1x16xf8E4M3FN, #dot1> -> tensor<16x16xf32, #blocked>
    tt.return %d : tensor<16x16xf32, #blocked>
  }

  // CHECK-LABEL: tt.func public @f8E5M2_to_f32(
  // CHECK: tt.fp_to_fp %arg0 : tensor<16x1xf8E5M2, {{.*}}> -> tensor<16x1xf32, {{.*}}>
  // CHECK: tt.fp_to_fp %arg1 : tensor<1x16xf8E5M2, {{.*}}> -> tensor<1x16xf32, {{.*}}>
  // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, {{.*}}>
  // CHECK: tt.return
  tt.func public @f8E5M2_to_f32(%a: tensor<16x1xf8E5M2, #dot0>, %b: tensor<1x16xf8E5M2, #dot1>, %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<16x1xf8E5M2, #dot0> * tensor<1x16xf8E5M2, #dot1> -> tensor<16x16xf32, #blocked>
    tt.return %d : tensor<16x16xf32, #blocked>
  }

  // CHECK-LABEL: tt.func public @f8E4M3FN_to_bf16(
  // CHECK: tt.fp_to_fp %arg0 : tensor<16x1xf8E4M3FN, {{.*}}> -> tensor<16x1xf32, {{.*}}>
  // CHECK: tt.fp_to_fp %arg1 : tensor<1x16xf8E4M3FN, {{.*}}> -> tensor<1x16xf32, {{.*}}>
  // CHECK: tt.fp_to_fp %arg2 : tensor<16x16xbf16, {{.*}}> -> tensor<16x16xf32, {{.*}}>
  // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, {{.*}}>
  // CHECK: tt.fp_to_fp {{.*}}, rounding = rtne : tensor<16x16xf32, {{.*}}> -> tensor<16x16xbf16, {{.*}}>
  // CHECK: tt.return
  tt.func public @f8E4M3FN_to_bf16(%a: tensor<16x1xf8E4M3FN, #dot0>, %b: tensor<1x16xf8E4M3FN, #dot1>, %c: tensor<16x16xbf16, #blocked>) -> tensor<16x16xbf16, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<16x1xf8E4M3FN, #dot0> * tensor<1x16xf8E4M3FN, #dot1> -> tensor<16x16xbf16, #blocked>
    tt.return %d : tensor<16x16xbf16, #blocked>
  }

  // CHECK-LABEL: tt.func public @bf16_to_bf16(
  // CHECK: tt.fp_to_fp %arg0 : tensor<16x1xbf16, {{.*}}> -> tensor<16x1xf32, {{.*}}>
  // CHECK: tt.fp_to_fp %arg1 : tensor<1x16xbf16, {{.*}}> -> tensor<1x16xf32, {{.*}}>
  // CHECK: tt.fp_to_fp %arg2 : tensor<16x16xbf16, {{.*}}> -> tensor<16x16xf32, {{.*}}>
  // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, {{.*}}>
  // CHECK: tt.fp_to_fp {{.*}}, rounding = rtne : tensor<16x16xf32, {{.*}}> -> tensor<16x16xbf16, {{.*}}>
  // CHECK: tt.return
  tt.func public @bf16_to_bf16(%a: tensor<16x1xbf16, #dot0>, %b: tensor<1x16xbf16, #dot1>, %c: tensor<16x16xbf16, #blocked>) -> tensor<16x16xbf16, #blocked> {
    %d = tt.dot %a, %b, %c : tensor<16x1xbf16, #dot0> * tensor<1x16xbf16, #dot1> -> tensor<16x16xbf16, #blocked>
    tt.return %d : tensor<16x16xbf16, #blocked>
  }

}
