// RUN: triton-opt %s --allocate-amdgpu-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 32768 : i32
// CHECK-NOT: ttg.shared = 65536 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @mfma_to_blocked
  // CHECK: allocation.offset = 0 : i32
  tt.func @mfma_to_blocked(%arg0: tensor<256x256xf16, #mma>) {
    %0 = ttg.convert_layout %arg0 : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked>
    tt.return
  }
}
