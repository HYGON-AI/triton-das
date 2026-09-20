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

  // Packed BF16 sub exists on gfx938/gfx946 via negate + V_PK_ADD_BF16.
  // GFX938-LABEL: bf16_sub
  // GFX938: llvm.fadd {{.*}} : vector<2xbf16>
  // GFX936-LABEL: bf16_sub
  // GFX936-NOT: vector<2xbf16>
  tt.func private @bf16_sub(%a: tensor<128xbf16, #blocked>, %b: tensor<128xbf16, #blocked>) -> tensor<128xbf16, #blocked> {
    %r = arith.subf %a, %b : tensor<128xbf16, #blocked>
    tt.return %r : tensor<128xbf16, #blocked>
  }

  // FSub: packed like FAdd/FMul (f32 needs the v_pk_*_f32 archs, f16 always).
  // GFX938-LABEL: f32_sub
  // GFX938: llvm.fsub {{.*}} : vector<2xf32>
  // GFX936-LABEL: f32_sub
  // GFX936: llvm.fsub {{.*}} : vector<2xf32>
  // GFX942-LABEL: f32_sub
  // GFX942: llvm.fsub {{.*}} : f32
  tt.func private @f32_sub(%a: tensor<128xf32, #blocked>, %b: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    %r = arith.subf %a, %b : tensor<128xf32, #blocked>
    tt.return %r : tensor<128xf32, #blocked>
  }

  // GFX938-LABEL: f16_sub
  // GFX938: llvm.fsub {{.*}} : vector<2xf16>
  // GFX936-LABEL: f16_sub
  // GFX936: llvm.fsub {{.*}} : vector<2xf16>
  // GFX942-LABEL: f16_sub
  // GFX942: llvm.fsub {{.*}} : vector<2xf16>
  tt.func private @f16_sub(%a: tensor<128xf16, #blocked>, %b: tensor<128xf16, #blocked>) -> tensor<128xf16, #blocked> {
    %r = arith.subf %a, %b : tensor<128xf16, #blocked>
    tt.return %r : tensor<128xf16, #blocked>
  }

  // Min/max: packed pair exposed to instruction selection; NaN semantics
  // (fminimum/fmaximum) are preserved by the destination op.
  // GFX938-LABEL: f32_min
  // GFX938: llvm.intr.minimum{{.*}}-> vector<2xf32>
  // GFX936-LABEL: f32_min
  // GFX936: llvm.intr.minimum{{.*}}-> vector<2xf32>
  // GFX942-LABEL: f32_min
  // GFX942: llvm.intr.minimum{{.*}}-> f32
  tt.func private @f32_min(%a: tensor<128xf32, #blocked>, %b: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    %r = arith.minimumf %a, %b : tensor<128xf32, #blocked>
    tt.return %r : tensor<128xf32, #blocked>
  }

  // GFX938-LABEL: f32_max
  // GFX938: llvm.intr.maximum{{.*}}-> vector<2xf32>
  // GFX936-LABEL: f32_max
  // GFX936: llvm.intr.maximum{{.*}}-> vector<2xf32>
  // GFX942-LABEL: f32_max
  // GFX942: llvm.intr.maximum{{.*}}-> f32
  tt.func private @f32_max(%a: tensor<128xf32, #blocked>, %b: tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked> {
    %r = arith.maximumf %a, %b : tensor<128xf32, #blocked>
    tt.return %r : tensor<128xf32, #blocked>
  }

  // fminnum (tl.minimum default) lowers to llvm.minnum.
  // GFX938-LABEL: f16_minnum
  // GFX938: llvm.intr.minnum{{.*}}-> vector<2xf16>
  // GFX936-LABEL: f16_minnum
  // GFX936: llvm.intr.minnum{{.*}}-> vector<2xf16>
  // GFX942-LABEL: f16_minnum
  // GFX942: llvm.intr.minnum{{.*}}-> vector<2xf16>
  tt.func private @f16_minnum(%a: tensor<128xf16, #blocked>, %b: tensor<128xf16, #blocked>) -> tensor<128xf16, #blocked> {
    %r = arith.minnumf %a, %b : tensor<128xf16, #blocked>
    tt.return %r : tensor<128xf16, #blocked>
  }

  // GFX938-LABEL: f16_max
  // GFX938: llvm.intr.maximum{{.*}}-> vector<2xf16>
  // GFX936-LABEL: f16_max
  // GFX936: llvm.intr.maximum{{.*}}-> vector<2xf16>
  // GFX942-LABEL: f16_max
  // GFX942: llvm.intr.maximum{{.*}}-> vector<2xf16>
  tt.func private @f16_max(%a: tensor<128xf16, #blocked>, %b: tensor<128xf16, #blocked>) -> tensor<128xf16, #blocked> {
    %r = arith.maximumf %a, %b : tensor<128xf16, #blocked>
    tt.return %r : tensor<128xf16, #blocked>
  }

  // Packed bf16 min/max pair exposed on gfx936/gfx938/gfx946
  // (v_pk_min/max_num_bf16); gfx942 has no packed bf16 min/max.
  // Note: the frontend promotes tl.minimum/tl.maximum on bf16 to
  // f32, so this path only fires for direct bf16 minnum/maxnum.
  // GFX938-LABEL: bf16_min
  // GFX938: llvm.intr.minimum{{.*}}-> vector<2xbf16>
  // GFX936-LABEL: bf16_min
  // GFX936: llvm.intr.minimum{{.*}}-> vector<2xbf16>
  // GFX942-LABEL: bf16_min
  // GFX942: llvm.intr.minimum{{.*}}-> bf16
  tt.func private @bf16_min(%a: tensor<128xbf16, #blocked>, %b: tensor<128xbf16, #blocked>) -> tensor<128xbf16, #blocked> {
    %r = arith.minimumf %a, %b : tensor<128xbf16, #blocked>
    tt.return %r : tensor<128xbf16, #blocked>
  }

  // GFX938-LABEL: bf16_max
  // GFX938: llvm.intr.maximum{{.*}}-> vector<2xbf16>
  // GFX936-LABEL: bf16_max
  // GFX936: llvm.intr.maximum{{.*}}-> vector<2xbf16>
  // GFX942-LABEL: bf16_max
  // GFX942: llvm.intr.maximum{{.*}}-> bf16
  tt.func private @bf16_max(%a: tensor<128xbf16, #blocked>, %b: tensor<128xbf16, #blocked>) -> tensor<128xbf16, #blocked> {
    %r = arith.maximumf %a, %b : tensor<128xbf16, #blocked>
    tt.return %r : tensor<128xbf16, #blocked>
  }
}
