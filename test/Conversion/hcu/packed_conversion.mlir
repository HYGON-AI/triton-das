// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx938" | FileCheck %s --check-prefix=GFX938
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx936" | FileCheck %s --check-prefix=GFX936

#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // The FpToFp and arith.truncf forms both consume adjacent F32 lanes.
  // GFX938-LABEL: f32_to_bf16
  // GFX938: llvm.call_intrinsic "llvm.hcu.cvt.pk.bf16.f32"
  // GFX936-LABEL: f32_to_bf16
  // GFX936-NOT: llvm.call_intrinsic "llvm.hcu.cvt.pk.bf16.f32"
  tt.func private @f32_to_bf16(%arg0: tensor<128xf32, #blocked2>) -> tensor<128xbf16, #blocked2> {
    %0 = arith.truncf %arg0 : tensor<128xf32, #blocked2> to tensor<128xbf16, #blocked2>
    tt.return %0 : tensor<128xbf16, #blocked2>
  }

  // GFX938-LABEL: f32_to_f16
  // GFX938: llvm.call_intrinsic "llvm.hcu.cvt.pk.f16.f32"
  // GFX936-LABEL: f32_to_f16
  // GFX936-NOT: llvm.call_intrinsic "llvm.hcu.cvt.pk.f16.f32"
  tt.func private @f32_to_f16(%arg0: tensor<128xf32, #blocked2>) -> tensor<128xf16, #blocked2> {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<128xf32, #blocked2> -> tensor<128xf16, #blocked2>
    tt.return %0 : tensor<128xf16, #blocked2>
  }
}
