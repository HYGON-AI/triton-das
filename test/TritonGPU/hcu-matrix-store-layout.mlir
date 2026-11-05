// RUN: triton-opt %s -split-input-file --verify-diagnostics

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 8], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#value = #mma
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @ordinary(%base: !tt.ptr<bf16>, %value: tensor<32x32xbf16, #value>) {
  amdg.matrix_store_from_reg %value, %base, shape = [], strides = [], tensorShape = [32, 32], indices = [] : tensor<32x32xbf16, #value>, !tt.ptr<bf16>
  tt.return
 }
}

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 8], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#value = #ttg.hcu_mmac_interleave2<parent = #mma>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @interleave(%base: !tt.ptr<bf16>, %value: tensor<32x32xbf16, #value>) {
  amdg.matrix_store_from_reg %value, %base, shape = [], strides = [], tensorShape = [32, 32], indices = [] : tensor<32x32xbf16, #value>, !tt.ptr<bf16>
  tt.return
 }
}

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 1]}>
#value = #ttg.hcu_mmac_interleave2<parent = #mma>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @interleave_32(%base: !tt.ptr<bf16>, %value: tensor<32x32xbf16, #value>) {
  amdg.matrix_store_from_reg %value, %base, shape = [], strides = [], tensorShape = [32, 32], indices = [] : tensor<32x32xbf16, #value>, !tt.ptr<bf16>
  tt.return
 }
}

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 8], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#value = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @blocked(%base: !tt.ptr<bf16>, %value: tensor<32x32xbf16, #value>) {
  // expected-error@+1 {{matrix store requires a rank-two MMAC result layout}}
  amdg.matrix_store_from_reg %value, %base, shape = [], strides = [], tensorShape = [32, 32], indices = [] : tensor<32x32xbf16, #value>, !tt.ptr<bf16>
  tt.return
 }
}
