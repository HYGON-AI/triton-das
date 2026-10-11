// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=1" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>

// gfx946 supports 16-bit accumulators, but these dots request FP32. Both
// instruction-selection paths must retain the native FP8/FP32 instruction.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @mls_fp8_fp32
  // CHECK: tt.matrix_load
  // CHECK: tt.matrix_load
  // CHECK-NOT: tt.dot_scaled
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @mls_fp8_fp32(%a: !tt.ptr<f8E4M3FN>, %b: !tt.ptr<f8E4M3FN>) -> tensor<64x64xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #blocked>
    %av = tt.matrix_load %a, shape = [%c64, %c128], strides = [%s128, %s1], tensorShape = [64, 128], indices = [%c0, %c0] : <f8E4M3FN> -> tensor<64x128xf8E4M3FN, #blocked1>
    %bv = tt.matrix_load %b, shape = [%c128, %c64], strides = [%s1, %s128], tensorShape = [128, 64], indices = [%c0, %c0] : <f8E4M3FN> -> tensor<128x64xf8E4M3FN, #blocked>
    %d = tt.dot_scaled %av, %bv, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #blocked1> * tensor<128x64xf8E4M3FN, #blocked> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }

  // CHECK-LABEL: tt.func public @plain_fp8_fp32
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @plain_fp8_fp32(%a: tensor<64x128xf8E4M3FN, #blocked1>, %b: tensor<128x64xf8E4M3FN, #blocked>) -> tensor<64x64xf32, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xf8E4M3FN, #blocked> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }

  // The instruction accumulates in FP32; restore the requested output type.
  // CHECK-LABEL: tt.func public @plain_fp8_f16
  // CHECK: arith.extf {{.*}} to tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: arith.truncf {{.*}} to tensor<64x64xf16, #blocked{{[0-9]*}}>
  // CHECK: tt.return {{.*}} : tensor<64x64xf16, #blocked{{[0-9]*}}>
  tt.func public @plain_fp8_f16(%a: tensor<64x128xf8E4M3FN, #blocked1>, %b: tensor<128x64xf8E4M3FN, #blocked>) -> tensor<64x64xf16, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf16, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xf8E4M3FN, #blocked> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf16, #blocked>
    tt.return %d : tensor<64x64xf16, #blocked>
  }

  // The instruction accumulates in FP32; restore the requested output type.
  // CHECK-LABEL: tt.func public @plain_fp8_bf16
  // CHECK: arith.extf {{.*}} to tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: arith.truncf {{.*}} to tensor<64x64xbf16, #blocked{{[0-9]*}}>
  // CHECK: tt.return {{.*}} : tensor<64x64xbf16, #blocked{{[0-9]*}}>
  tt.func public @plain_fp8_bf16(%a: tensor<64x128xf8E4M3FN, #blocked1>, %b: tensor<128x64xf8E4M3FN, #blocked>) -> tensor<64x64xbf16, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xbf16, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xf8E4M3FN, #blocked> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xbf16, #blocked>
    tt.return %d : tensor<64x64xbf16, #blocked>
  }

  // CHECK-LABEL: tt.func public @f16_acc_f32
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @f16_acc_f32(%a: tensor<64x128xf16, #blocked1>, %b: tensor<128x64xf16, #blocked>) -> tensor<64x64xf32, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }

  // CHECK-LABEL: tt.func public @f16_acc_f16
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf16, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @f16_acc_f16(%a: tensor<64x128xf16, #blocked1>, %b: tensor<128x64xf16, #blocked>) -> tensor<64x64xf16, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf16, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf16, #blocked>
    tt.return %d : tensor<64x64xf16, #blocked>
  }

  // CHECK-LABEL: tt.func public @f16_acc_bf16
  // CHECK: tt.dot {{.*}} -> tensor<64x64xbf16, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @f16_acc_bf16(%a: tensor<64x128xf16, #blocked1>, %b: tensor<128x64xf16, #blocked>) -> tensor<64x64xbf16, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xbf16, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xbf16, #blocked>
    tt.return %d : tensor<64x64xbf16, #blocked>
  }

  // CHECK-LABEL: tt.func public @fp64_acc_fp64
  // CHECK: tt.dot {{.*}} -> tensor<64x64xf64, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @fp64_acc_fp64(%a: tensor<64x16xf64, #blocked1>, %b: tensor<16x64xf64, #blocked>) -> tensor<64x64xf64, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf64, #blocked>
    %av = ttg.convert_layout %a : tensor<64x16xf64, #blocked1> -> tensor<64x16xf64, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<16x64xf64, #blocked> -> tensor<16x64xf64, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x16xf64, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<16x64xf64, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xf64, #blocked>
    tt.return %d : tensor<64x64xf64, #blocked>
  }

  // CHECK-LABEL: tt.func public @int8_acc_int32
  // CHECK: tt.dot {{.*}} -> tensor<64x64xi32, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @int8_acc_int32(%a: tensor<64x128xi8, #blocked1>, %b: tensor<128x64xi8, #blocked>) -> tensor<64x64xi32, #blocked> {
    %c = arith.constant dense<0> : tensor<64x64xi32, #blocked>
    %av = ttg.convert_layout %a : tensor<64x128xi8, #blocked1> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x128xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<128x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x64xi32, #blocked>
    tt.return %d : tensor<64x64xi32, #blocked>
  }

  // CHECK-LABEL: tt.func public @fp4_acc_f32
  // CHECK: tt.dot_scaled {{.*}} -> tensor<64x64xf32, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @fp4_acc_f32(%a: tensor<64x64xi8, #blocked1>, %b: tensor<64x64xi8, #blocked>) -> tensor<64x64xf32, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf32, #blocked>
    %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #blocked1> * tensor<64x64xi8, #blocked> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }

  // CHECK-LABEL: tt.func public @fp4_acc_f16
  // CHECK: tt.dot_scaled {{.*}} -> tensor<64x64xf16, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @fp4_acc_f16(%a: tensor<64x64xi8, #blocked1>, %b: tensor<64x64xi8, #blocked>) -> tensor<64x64xf16, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xf16, #blocked>
    %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #blocked1> * tensor<64x64xi8, #blocked> -> tensor<64x64xf16, #blocked>
    tt.return %d : tensor<64x64xf16, #blocked>
  }

  // CHECK-LABEL: tt.func public @fp4_acc_bf16
  // CHECK: tt.dot_scaled {{.*}} -> tensor<64x64xbf16, #mma{{[0-9]*}}>
  // CHECK: tt.return
  tt.func public @fp4_acc_bf16(%a: tensor<64x64xi8, #blocked1>, %b: tensor<64x64xi8, #blocked>) -> tensor<64x64xbf16, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x64xbf16, #blocked>
    %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #blocked1> * tensor<64x64xi8, #blocked> -> tensor<64x64xbf16, #blocked>
    tt.return %d : tensor<64x64xbf16, #blocked>
  }
}
