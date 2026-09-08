// Modified by Hygon Information Technology Co., Ltd., 2026.
// RUN: triton-opt %s -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline="use_async_copy=1" | FileCheck %s --check-prefix=UNIFIED -DWAIT=0 --implicit-check-not="tt.matrix_load " --implicit-check-not="ttg.pipeline.hcu_mls"
// RUN: triton-opt %s -tritonamdgpu-schedule-loops="num_stages=3" -tritonamdgpu-pipeline="use_async_copy=1" | FileCheck %s --check-prefix=UNIFIED -DWAIT=2 --implicit-check-not="tt.matrix_load "
// RUN: triton-opt %s -tritonamdgpu-schedule-loops="num_stages=4" -tritonamdgpu-pipeline="use_async_copy=1" | FileCheck %s --check-prefix=UNIFIED -DWAIT=4 --implicit-check-not="tt.matrix_load "
// RUN: triton-opt %s -tritonamdgpu-schedule-loops="num_stages=3" -tritonamdgpu-pipeline="use_async_copy=1" --convert-scf-to-cf --convert-index-to-llvm --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx938" | FileCheck %s --check-prefix=LLVM-UNIFIED
// With async copy disabled, the preceding MLS pass consumes the matrix loads.
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=3 use_async_copy=0" | FileCheck %s --check-prefix=MULTI -DWAIT=2 --implicit-check-not="tt.matrix_load "
// Non-pipelineable MLS remains for fallback lowering. Inspect before cleanup DCE.
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=1 use_async_copy=0" -tritonamdgpu-schedule-loops="num_stages=1" | FileCheck %s --check-prefix=FALLBACK --implicit-check-not="amdg.matrix_load_to_local" --implicit-check-not="ttg.async_" --implicit-check-not="loop.stage"
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=2 use_async_copy=0" | FileCheck %s --check-prefix=SINGLE
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=2 use_async_copy=1" | FileCheck %s --check-prefix=ASYNC
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=2 use_async_copy=1" --tritonhcu-update-async-wait-count=arch-generation-name=gfx938 --convert-scf-to-cf --convert-index-to-llvm --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx938" | FileCheck %s --check-prefix=LLVM-ASYNC
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=3 use_async_copy=0" -tritonamdgpu-schedule-loops=num_stages=3 -tritonamdgpu-pipeline="use_async_copy=0" --tritonhcu-update-async-wait-count=arch-generation-name=gfx938 | FileCheck %s --check-prefix=MULTI -DWAIT=2
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=4 use_async_copy=0" -tritonamdgpu-schedule-loops=num_stages=4 -tritonamdgpu-pipeline="use_async_copy=0" --tritonhcu-update-async-wait-count=arch-generation-name=gfx938 | FileCheck %s --check-prefix=MULTI -DWAIT=4
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=3 use_async_copy=1" | FileCheck %s --check-prefix=MULTI -DWAIT=2
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=4 use_async_copy=1" | FileCheck %s --check-prefix=MULTI -DWAIT=4
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=3 use_async_copy=0" --tritonhcu-update-async-wait-count=arch-generation-name=gfx938 --convert-scf-to-cf --convert-index-to-llvm --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx938" | FileCheck %s --check-prefix=LLVM-MULTI -DWAIT=2
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=4 use_async_copy=0" --tritonhcu-update-async-wait-count=arch-generation-name=gfx938 --convert-scf-to-cf --convert-index-to-llvm --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx938" | FileCheck %s --check-prefix=LLVM-MULTI -DWAIT=4

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // FALLBACK-LABEL: tt.func public @mls_pipeline
  // FALLBACK: scf.for
  // FALLBACK-COUNT-2: tt.matrix_load
  // FALLBACK: tt.dot
  // FALLBACK: scf.yield
  // UNIFIED: tt.func public @mls_pipeline
  // UNIFIED: amdg.matrix_load_to_local {{.*}} pred =
  // UNIFIED-NEXT: {{.*}}ttg.async_commit_group
  // UNIFIED: scf.for
  // UNIFIED: ttg.async_wait {{.*}} {num = [[WAIT]] : i32}
  // UNIFIED: ttg.local_load {{.*}} token
  // UNIFIED: scf.yield
  // UNIFIED: ttg.async_wait {{.*}} {num = 0 : i32}
  // LLVM-UNIFIED-LABEL: llvm.func @mls_pipeline
  // The false edge must bypass the entire copy; both edges reach the mark.
  // Check all four prologue copies (A/B for two prefetched iterations).
  // LLVM-UNIFIED-NOT: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.cond_br {{.*}}, ^[[BODY:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
  // LLVM-UNIFIED-NEXT: ^[[BODY]]:
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.br ^[[CONT]]
  // LLVM-UNIFIED-NEXT: ^[[CONT]]:
  // LLVM-UNIFIED-NEXT: rocdl.asyncmark
  // LLVM-UNIFIED-NOT: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.cond_br {{.*}}, ^[[BODY:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
  // LLVM-UNIFIED-NEXT: ^[[BODY]]:
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.br ^[[CONT]]
  // LLVM-UNIFIED-NEXT: ^[[CONT]]:
  // LLVM-UNIFIED-NEXT: rocdl.asyncmark
  // LLVM-UNIFIED-NOT: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.cond_br {{.*}}, ^[[BODY:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
  // LLVM-UNIFIED-NEXT: ^[[BODY]]:
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.br ^[[CONT]]
  // LLVM-UNIFIED-NEXT: ^[[CONT]]:
  // LLVM-UNIFIED-NEXT: rocdl.asyncmark
  // LLVM-UNIFIED-NOT: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.cond_br {{.*}}, ^[[BODY:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
  // LLVM-UNIFIED-NEXT: ^[[BODY]]:
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: rocdl.matrix.load
  // LLVM-UNIFIED-NOT: rocdl.asyncmark
  // LLVM-UNIFIED: llvm.br ^[[CONT]]
  // LLVM-UNIFIED-NEXT: ^[[CONT]]:
  // LLVM-UNIFIED-NEXT: rocdl.asyncmark
  // LLVM-UNIFIED: rocdl.wait.asyncmark 2
  // LLVM-UNIFIED: rocdl.wait.asyncmark 0
  // SINGLE-LABEL: tt.func public @mls_pipeline
  // ASYNC-LABEL: tt.func public @mls_pipeline
  // LLVM-ASYNC-LABEL: llvm.func @mls_pipeline
  // LLVM-ASYNC: rocdl.matrix.load.32x16.b16
  // LLVM-ASYNC-NEXT: rocdl.asyncmark
  // LLVM-ASYNC: rocdl.matrix.load.32x16.b16
  // LLVM-ASYNC-NEXT: rocdl.asyncmark
  // LLVM-ASYNC: rocdl.wait.asyncmark 0
  // LLVM-ASYNC-NOT: amdg.async_wait
  // Each prefetched A/B pair contributes two commit groups. Wait for both
  // current tiles while leaving the newer pairs in flight, even when the
  // generic async-copy pipeline is disabled. Drain all groups in the epilogue.
  // MULTI: tt.func public @mls_pipeline
  // MULTI: scf.for
  // MULTI: ttg.async_wait %{{[^ ,]+}}, %{{[^ ,]+}} {num = [[WAIT]] : i32}
  // MULTI: scf.yield
  // MULTI: ttg.async_wait {{.*}} {num = 0 : i32}
  // MULTI-NOT: ttg.async_wait
  // MULTI: tt.return
  // LLVM-MULTI: llvm.func @mls_pipeline
  // LLVM-MULTI-NOT: rocdl.wait.asyncmark
  // LLVM-MULTI: rocdl.wait.asyncmark [[WAIT]]
  // LLVM-MULTI: rocdl.wait.asyncmark 0
  // LLVM-MULTI-NOT: rocdl.wait.asyncmark
  // LLVM-MULTI: llvm.return
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
