// RUN: triton-opt %s -split-input-file --tritonhcu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// MatrixLoadToLocal participates in the same async commit/wait accounting as
// AsyncCopyGlobalToLocal. Count it from MlsEncoding + MlsGroup lowering info.

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: matrix_load_to_local_waitcnt
  tt.func public @matrix_load_to_local_waitcnt(
      %base: !tt.ptr<f16>,
      %m: i32,
      %n: i32,
      %stride_m: i64,
      %stride_n: i64,
      %i: i32,
      %j: i32,
      %dst0: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>,
      %dst1: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) {
    %0 = amdg.matrix_load_to_local %base, %dst0, shape = [%m, %n], strides = [%stride_m, %stride_n], tensorShape = [128, 64], indices = [%i, %j] {MlsEncoding = #amdg<MlsEncoding opIdx = 0, mlsTile = [64, 64], elemBitWidth = 16, elemBitTyKind = 0, alt2Kind = 0, version = 2, order = [0, 1], warpsPerCTA = [1, 1]>} : <f16> -> <128x64xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = amdg.matrix_load_to_local %base, %dst1, shape = [%m, %n], strides = [%stride_m, %stride_n], tensorShape = [128, 64], indices = [%i, %j] {MlsEncoding = #amdg<MlsEncoding opIdx = 0, mlsTile = [64, 64], elemBitWidth = 16, elemBitTyKind = 0, alt2Kind = 0, version = 2, order = [0, 1], warpsPerCTA = [1, 1]>} : <f16> -> <128x64xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // shape=[128,64], mlsTile=[64,64], warpsPerCTA=[1,1],
    // matrix_load_64x16_b16 => numReps=[2,1], instrsPerWarp=[4,1].
    // CHECK: amdg.async_wait {{.*}} {num_inst = 8
    %4 = ttg.async_wait %1 {num = 0 : i32}
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %5 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}
