// HCU membar regression tests.
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory -test-tritonamdgpu-membar | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// MatrixLoadToLocal follows the same membar filtering contract as
// AsyncCopyGlobalToLocal. Single-buffer MLS local_loads intentionally do not use
// an AsyncWait token, so the true WAR against the next MLS write is not filtered.
// CHECK-LABEL: matrix_load_to_local_single_buffer_true_war
tt.func @matrix_load_to_local_single_buffer_true_war(%base: !tt.ptr<f16>, %m: i32, %n: i32, %stride_m: i64, %stride_n: i64, %i: i32, %j: i32) {
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %0 = amdg.matrix_load_to_local %base, %alloc, shape = [%m, %n], strides = [%stride_m, %stride_n], tensorShape = [16, 16], indices = [%i, %j] : <f16> -> <16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %1 = ttg.async_commit_group tokens %0
  %2 = ttg.async_wait %1 {num = 0 : i32}
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  %3 = ttg.local_load %alloc : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  // CHECK: ttg.local_load
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: amdg.matrix_load_to_local
  %4 = amdg.matrix_load_to_local %base, %alloc, shape = [%m, %n], strides = [%stride_m, %stride_n], tensorShape = [16, 16], indices = [%i, %j] : <f16> -> <16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

// Double-buffer MLS local_loads may carry the AsyncWait token, matching the
// AsyncCopyGlobalToLocal path: membar keeps only the wait barrier and filters the
// false WAR against the next-buffer MLS prefetch.
// CHECK-LABEL: matrix_load_to_local_double_buffer_false_war
tt.func @matrix_load_to_local_double_buffer_false_war(%base: !tt.ptr<f16>, %m: i32, %n: i32, %stride_m: i64, %stride_n: i64, %i: i32, %j: i32) {
  %index_0 = arith.constant 0 : i32
  %index_1 = arith.constant 1 : i32
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile0 = ttg.memdesc_index %alloc[%index_0] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %tile1 = ttg.memdesc_index %alloc[%index_1] : !ttg.memdesc<2x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>

  %0 = amdg.matrix_load_to_local %base, %tile0, shape = [%m, %n], strides = [%stride_m, %stride_n], tensorShape = [16, 16], indices = [%i, %j] : <f16> -> <16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %1 = ttg.async_commit_group tokens %0
  %2 = ttg.async_wait %1 {num = 0 : i32}
  %3 = ttg.local_load %tile0 token %2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> tensor<16x16xf16, #AL>
  %4 = amdg.matrix_load_to_local %base, %tile1, shape = [%m, %n], strides = [%stride_m, %stride_n], tensorShape = [16, 16], indices = [%i, %j] : <f16> -> <16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK-NOT: ttg.local_barrier
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.return
  tt.return
}
}
