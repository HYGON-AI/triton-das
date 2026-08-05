// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=2 use_async_copy=0" | FileCheck %s --check-prefix=SINGLE
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=2 use_async_copy=1" | FileCheck %s --check-prefix=ASYNC

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // SINGLE-LABEL: tt.func public @mls_pipeline
  // ASYNC-LABEL: tt.func public @mls_pipeline
  tt.func public @mls_pipeline(
      %a_ptr: !tt.ptr<f16>,
      %b_ptr: !tt.ptr<f16>,
      %m: i32,
      %n: i32,
      %k: i32,
      %stride_am: i64,
      %stride_bn: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %accumulator = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>

    // SINGLE: ttg.local_alloc
    // SINGLE-SAME: !ttg.memdesc<1x32x64xf16
    // SINGLE: ttg.local_alloc
    // SINGLE-SAME: !ttg.memdesc<1x64x32xf16
    // SINGLE: scf.for
    // SINGLE:   ttg.async_wait
    // SINGLE:   ttg.local_load %{{[^ ]+}} :
    // SINGLE:   ttg.local_load %{{[^ ]+}} :
    // SINGLE:   amdg.matrix_load_to_local
    // SINGLE:   ttg.async_commit_group
    // SINGLE:   amdg.matrix_load_to_local
    // SINGLE:   ttg.async_commit_group
    // SINGLE:   tt.dot

    // ASYNC: ttg.local_alloc
    // ASYNC-SAME: !ttg.memdesc<2x32x64xf16
    // ASYNC: ttg.local_alloc
    // ASYNC-SAME: !ttg.memdesc<2x64x32xf16
    // ASYNC: scf.for
    // ASYNC:   ttg.async_wait %{{[^ ]+}}
    // ASYNC:   amdg.matrix_load_to_local
    // ASYNC:   ttg.async_commit_group
    // ASYNC:   ttg.local_load %{{[^ ]+}} token
    %result = scf.for %iv = %c0_i32 to %k step %c64_i32
        iter_args(%acc = %accumulator) -> (tensor<32x32xf32, #mma>) : i32 {
      %a = tt.matrix_load %a_ptr,
          shape = [%m, %k],
          strides = [%stride_am, %c1_i64],
          tensorShape = [32, 64],
          indices = [%c0_i32, %iv]
          {MlsEncoding = #amdg<MlsEncoding opIdx = 0, mlsTile = [16, 32], elemBitWidth = 16, elemBitTyKind = 0, alt2Kind = 0, version = 1, order = [1, 0], warpsPerCTA = [2, 2]>}
          : <f16> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %b = tt.matrix_load %b_ptr,
          shape = [%k, %n],
          strides = [%c1_i64, %stride_bn],
          tensorShape = [64, 32],
          indices = [%iv, %c0_i32]
          {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [32, 16], elemBitWidth = 16, elemBitTyKind = 0, alt2Kind = 0, version = 1, order = [0, 1], warpsPerCTA = [2, 2]>}
          : <f16> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %next = tt.dot %a, %b, %acc, inputPrecision = tf32
          : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
          * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
          -> tensor<32x32xf32, #mma>
      scf.yield %next : tensor<32x32xf32, #mma>
    }
    tt.return
  }
}
