// Modified by Hygon Information Technology Co., Ltd., 2026.
// RUN: triton-opt %s -split-input-file --convert-triton-hcu-to-llvm="arch=gfx938 ftz=True" | FileCheck %s --check-prefixes=COMMON,FTZ
// RUN: triton-opt %s -split-input-file --convert-triton-hcu-to-llvm="arch=gfx938 ftz=False" | FileCheck %s --check-prefixes=COMMON,NO_FTZ

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: llvm.func @sqrt_rn
  // COMMON-NOT: llvm.call
  // COMMON: llvm.intr.sqrt
  // COMMON-NOT: llvm.intr.fma
  // COMMON: llvm.return
  tt.func public @sqrt_rn(%x: tensor<64xf32, #blocked>) {
    %r = tt.precise_sqrt %x : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: llvm.func @rsqrt
  // FTZ: llvm.call @llvm.amdgcn.rsq.f32
  // NO_FTZ: llvm.call @__ocml_rsqrt_f32
  tt.func public @rsqrt(%x: tensor<64xf32, #blocked>) {
    %r = math.rsqrt %x : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: llvm.func @sqrt
  // FTZ-NOT: llvm.select
  // FTZ: llvm.call @llvm.amdgcn.sqrt.f32
  // NO_FTZ: %[[SCALED:.*]] = llvm.select
  // NO_FTZ: llvm.call @llvm.amdgcn.sqrt.f32(%[[SCALED]])
  tt.func public @sqrt(%x: tensor<64xf32, #blocked>) {
    %r = math.sqrt %x : tensor<64xf32, #blocked>
    tt.return
  }
}
