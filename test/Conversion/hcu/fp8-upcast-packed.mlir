// RUN: triton-opt %s --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func internal @fp8_to_bf16
  // CHECK: %[[PACK:.*]] = llvm.bitcast {{.*}} : vector<4xi8> to i32
  // CHECK: rocdl.hcu.cvt.scale.pk.bf16.fp8 %[[PACK]][false]
  // CHECK: rocdl.hcu.cvt.scale.pk.bf16.fp8 %[[PACK]][true]
  tt.func private @fp8_to_bf16(%input: tensor<256xf8E4M3FN, #blocked>) -> tensor<256xbf16, #blocked> {
    %output = tt.fp_to_fp %input : tensor<256xf8E4M3FN, #blocked> -> tensor<256xbf16, #blocked>
    tt.return %output : tensor<256xbf16, #blocked>
  }
  // CHECK-LABEL: llvm.func internal @fp8_to_f16
  // CHECK: %[[PACK:.*]] = llvm.bitcast {{.*}} : vector<4xi8> to i32
  // CHECK: rocdl.hcu.cvt.scale.pk.f16.fp8 %[[PACK]][false]
  // CHECK: rocdl.hcu.cvt.scale.pk.f16.fp8 %[[PACK]][true]
  tt.func private @fp8_to_f16(%input: tensor<256xf8E4M3FN, #blocked>) -> tensor<256xf16, #blocked> {
    %output = tt.fp_to_fp %input : tensor<256xf8E4M3FN, #blocked> -> tensor<256xf16, #blocked>
    tt.return %output : tensor<256xf16, #blocked>
  }
  // CHECK-LABEL: llvm.func internal @bf8_to_f16
  // CHECK: %[[PACK:.*]] = llvm.bitcast {{.*}} : vector<4xi8> to i32
  // CHECK: rocdl.hcu.cvt.scale.pk.f16.bf8 %[[PACK]][false]
  // CHECK: rocdl.hcu.cvt.scale.pk.f16.bf8 %[[PACK]][true]
  tt.func private @bf8_to_f16(%input: tensor<256xf8E5M2, #blocked>) -> tensor<256xf16, #blocked> {
    %output = tt.fp_to_fp %input : tensor<256xf8E5M2, #blocked> -> tensor<256xf16, #blocked>
    tt.return %output : tensor<256xf16, #blocked>
  }
}
