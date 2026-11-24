// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=2" --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" --canonicalize -o %t.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.mlir -o %t.ll
// RUN: opt -passes=verify -disable-output %t.ll
// RUN: FileCheck %s --input-file=%t.ll

// Check the integer-backed BF16 accumulator ABI through LLVM translation.
// Instruction selection alone does not detect a BF16/i16 type mismatch.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: define internal {{.*}} @fp4_acc_bf16(
  // CHECK: %[[ACC:.*]] = bitcast <4 x bfloat> {{.*}} to <4 x i16>
  // CHECK: %[[RESULT:.*]] = call <4 x i16> @llvm.hcu.mmac.bf16.16x16x64.fp4({{.*}}<4 x i16> %[[ACC]],
  // CHECK: bitcast <4 x i16> {{.*}} to <4 x bfloat>
  tt.func private @fp4_acc_bf16(%a: tensor<64x64xi8, #blocked1>, %b: tensor<64x64xi8, #blocked>, %c: tensor<64x64xbf16, #blocked>) -> tensor<64x64xbf16, #blocked> {
    %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #blocked1> * tensor<64x64xi8, #blocked> -> tensor<64x64xbf16, #blocked>
    tt.return %d : tensor<64x64xbf16, #blocked>
  }

  // CHECK-LABEL: define internal {{.*}} @fp8_fp4_acc_bf16(
  // CHECK: %[[ACC:.*]] = bitcast <4 x bfloat> {{.*}} to <4 x i16>
  // CHECK: %[[RESULT:.*]] = call <4 x i16> @llvm.hcu.mmac.bf16.16x16x32.f8f6f4({{.*}}<4 x i16> %[[ACC]],
  // CHECK: bitcast <4 x i16> {{.*}} to <4 x bfloat>
  tt.func private @fp8_fp4_acc_bf16(%a: tensor<64x128xf8E4M3FN, #blocked1>, %b: tensor<64x64xi8, #blocked>, %c: tensor<64x64xbf16, #blocked>) -> tensor<64x64xbf16, #blocked> {
    %d = tt.dot_scaled %a, %b, %c lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<64x128xf8E4M3FN, #blocked1> * tensor<64x64xi8, #blocked> -> tensor<64x64xbf16, #blocked>
    tt.return %d : tensor<64x64xbf16, #blocked>
  }
}
