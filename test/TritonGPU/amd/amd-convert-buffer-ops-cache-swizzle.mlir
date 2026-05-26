// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="analyze-small-tensor-ofst=true buffer-cache-swizzle=true" | FileCheck %s --check-prefixes=AUTO,GFX942
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="buffer-cache-swizzle=true" | FileCheck %s --check-prefixes=WIDE
// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="analyze-small-tensor-ofst=true" | FileCheck %s --check-prefixes=OPT_OUT

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @auto_narrow_default_8k(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
      -> tensor<64xf16, #blocked1> {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
    // AUTO-LABEL: auto_narrow_default_8k
    // AUTO: amdgpu.buffer_load %{{.*}}[%{{.*}}] stride = %c8192_i32 : i32
    // OPT_OUT-LABEL: auto_narrow_default_8k
    // OPT_OUT: amdgpu.buffer_load %{{.*}}[%{{.*}}] : tensor
    // OPT_OUT-NOT: stride =
    %3 = tt.load %2 : tensor<64x!tt.ptr<f16>, #blocked1>
    tt.return %3 : tensor<64xf16, #blocked1>
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @auto_wide_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) -> tensor<256xf32, #blocked0> {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // AUTO-LABEL: auto_wide_1d
    // WIDE-LABEL: auto_wide_1d
    // WIDE: amdgpu.buffer_load %{{.*}}[%{{.*}}] : tensor<256xf32, #blocked>
    %7 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return %7 : tensor<256xf32, #blocked0>
  }
}

// -----

#blocked_wide = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @auto_non_uniform_wide_layout(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
      -> tensor<1024xf16, #blocked_wide> {
    %c64_i32 = arith.constant 64 : i32
    %c25664_i32 = arith.constant 25664 : i32
    %c0_i64 = arith.constant 0 : i64
    %base_int = tt.ptr_to_int %arg0 : !tt.ptr<f16> -> i64
    %base_non_negative = arith.cmpi sge, %base_int, %c0_i64 : i64
    llvm.intr.assume %base_non_negative : i1
    %pid = tt.get_program_id x : i32
    %pid_off = arith.muli %pid, %c25664_i32 : i32
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked_wide>
    %1 = tt.splat %c64_i32 : i32 -> tensor<1024xi32, #blocked_wide>
    %x0 = arith.divsi %0, %1 : tensor<1024xi32, #blocked_wide>
    %x1 = arith.remsi %0, %1 : tensor<1024xi32, #blocked_wide>
    %x0_scaled = arith.muli %x0, %1 : tensor<1024xi32, #blocked_wide>
    %tile_off = tt.splat %pid_off : i32 -> tensor<1024xi32, #blocked_wide>
    %linearized = arith.addi %x1, %x0_scaled : tensor<1024xi32, #blocked_wide>
    %offset = arith.addi %linearized, %tile_off : tensor<1024xi32, #blocked_wide>
    %ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked_wide>
    %addptr = tt.addptr %ptr, %offset : tensor<1024x!tt.ptr<f16>, #blocked_wide>, tensor<1024xi32, #blocked_wide>
    // AUTO-LABEL: auto_non_uniform_wide_layout
    // AUTO: amdgpu.buffer_load %{{.*}}[%{{.*}}] stride = %c8192_i32 : i32
    %load = tt.load %addptr : tensor<1024x!tt.ptr<f16>, #blocked_wide>
    tt.return %load : tensor<1024xf16, #blocked_wide>
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @auto_gfx942_unsupported(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
      -> tensor<64xf16, #blocked1> {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
    // GFX942-LABEL: auto_gfx942_unsupported
    // GFX942: amdgpu.buffer_load %{{.*}}[%{{.*}}] : tensor
    // GFX942-NOT: stride =
    %3 = tt.load %2 : tensor<64x!tt.ptr<f16>, #blocked1>
    tt.return %3 : tensor<64xf16, #blocked1>
  }
}
