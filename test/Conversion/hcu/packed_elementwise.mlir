// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx938" | FileCheck %s --check-prefix=GFX938
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx936" | FileCheck %s --check-prefix=GFX936
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx942" | FileCheck %s --check-prefix=GFX942

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX938-LABEL: f32_add
  // GFX938: llvm.fadd {{.*}} : vector<2xf32>
  // GFX936-LABEL: f32_add
  // GFX936: llvm.fadd {{.*}} : vector<2xf32>
  tt.func private @f32_add(%a: tensor<128xf32, #blocked>, %b: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    %r = arith.addf %a, %b : tensor<128xf32, #blocked>
    tt.return %r : tensor<128xf32, #blocked>
  }

  // GFX938-LABEL: f16_fma
  // GFX938: llvm.intr.fma{{.*}}vector<2xf16>
  // GFX936-LABEL: f16_fma
  // GFX936: llvm.intr.fma{{.*}}vector<2xf16>
  // GFX942-LABEL: f16_fma
  // GFX942: llvm.intr.fma{{.*}}vector<2xf16>
  tt.func private @f16_fma(%a: tensor<128xf16, #blocked>, %b: tensor<128xf16, #blocked>, %c: tensor<128xf16, #blocked>) -> tensor<128xf16, #blocked> {
    %r = math.fma %a, %b, %c : tensor<128xf16, #blocked>
    tt.return %r : tensor<128xf16, #blocked>
  }

  // GFX938-LABEL: bf16_mul
  // GFX938: llvm.fmul {{.*}} : vector<2xbf16>
  // GFX936-LABEL: bf16_mul
  // GFX936: llvm.fmul {{.*}} : f32
  tt.func private @bf16_mul(%a: tensor<128xbf16, #blocked>, %b: tensor<128xbf16, #blocked>) -> tensor<128xbf16, #blocked> {
    %r = arith.mulf %a, %b : tensor<128xbf16, #blocked>
    tt.return %r : tensor<128xbf16, #blocked>
  }
}
