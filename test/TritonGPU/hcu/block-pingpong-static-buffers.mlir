// RUN: triton-opt %s -tritonhcu-finalize-block-pingpong -canonicalize --allocate-shared-memory -test-tritonamdgpu-membar | FileCheck %s

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared_a = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // Model the main loop immediately after late unroll=2.  Slot parity is
  // static within this two-BK body: consume slot0, publish slot1, consume
  // slot1, publish slot0, then carry slot0 around the loop backedge.
  // CHECK-LABEL: tt.func public @late_unrolled_static_slots
  tt.func public @late_unrolled_static_slots(
      %next_a: tensor<256x32xf16, #blocked_a>,
      %next_b: tensor<32x256xf16, #blocked_b>, %dynamic_slot: i32,
      %limit: i32)
      -> tensor<256x256xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable>
    %slot0_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
    %slot0_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
    %cold_a = ttg.memdesc_index %alloc_a[%dynamic_slot] : !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
    %cold_b = ttg.memdesc_index %alloc_b[%dynamic_slot] : !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>

    %result:4 = scf.for %iv = %c0_i32 to %limit step %c64_i32
        iter_args(%acc = %zero, %current_a = %slot0_a, %current_b = %slot0_b,
                  %slot_index = %c0_i32)
        -> (tensor<256x256xf32, #mma>,
            !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>,
            !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>, i32) : i32 {
      %slot1_increment = arith.addi %slot_index, %c1_i32 : i32
      %slot1_in_range = arith.cmpi slt, %slot1_increment, %c2_i32 : i32
      %slot1_index = arith.select %slot1_in_range, %slot1_increment, %c0_i32 : i32
      %slot1_a = ttg.memdesc_index %alloc_a[%slot1_index] : !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
      %slot1_b = ttg.memdesc_index %alloc_b[%slot1_index] : !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
      %a0 = ttg.local_load %current_a : !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable> -> tensor<256x32xf16, #dot_a>
      %b0 = ttg.local_load %current_b : !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable> -> tensor<32x256xf16, #dot_b>
      rocdl.sched.barrier 0
      ttg.local_store %next_a, %slot1_a : tensor<256x32xf16, #blocked_a> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
      ttg.local_store %next_b, %slot1_b : tensor<32x256xf16, #blocked_b> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
      gpu.barrier
      %acc0 = tt.dot %a0, %b0, %acc, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>

      %slot0_increment = arith.addi %slot1_index, %c1_i32 : i32
      %slot0_in_range = arith.cmpi slt, %slot0_increment, %c2_i32 : i32
      %slot0_index = arith.select %slot0_in_range, %slot0_increment, %c0_i32 : i32
      %next_slot0_a = ttg.memdesc_index %alloc_a[%slot0_index] : !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
      %next_slot0_b = ttg.memdesc_index %alloc_b[%slot0_index] : !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
      %a1 = ttg.local_load %slot1_a : !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable> -> tensor<256x32xf16, #dot_a>
      %b1 = ttg.local_load %slot1_b : !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable> -> tensor<32x256xf16, #dot_b>
      rocdl.sched.barrier 0
      ttg.local_store %next_a, %next_slot0_a : tensor<256x32xf16, #blocked_a> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
      ttg.local_store %next_b, %next_slot0_b : tensor<32x256xf16, #blocked_b> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
      gpu.barrier
      %acc1 = tt.dot %a1, %b1, %acc0, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>
      scf.yield %acc1, %next_slot0_a, %next_slot0_b, %slot0_index
          : tensor<256x256xf32, #mma>,
            !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>,
            !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>, i32
    } {hcu.block_pingpong.applied}

    // Independent allocs prove that current-slot reads and next-slot writes
    // do not alias.  Membar must not add a local_barrier between them.
    // CHECK-COUNT-4: ttg.local_alloc
    // CHECK-NOT: memdesc<2x
    // A cold remainder may still choose a slot dynamically.  Canonicalization
    // turns the generated scf.if into arith.select; alias analysis must join
    // both independent slots instead of asserting on an unknown memdesc op.
    // CHECK: arith.select {{.*}} : !ttg.memdesc<256x32xf16
    // CHECK: arith.select {{.*}} : !ttg.memdesc<32x256xf16
    // CHECK: scf.for
    // CHECK: ttg.local_load
    // CHECK: ttg.local_load
    // CHECK-NEXT: rocdl.sched.barrier 0
    // CHECK-NOT: ttg.local_barrier
    // CHECK: ttg.local_store
    // CHECK: ttg.local_store
    // CHECK: gpu.barrier
    // CHECK: tt.dot
    // CHECK: ttg.local_load
    // CHECK: ttg.local_load
    // CHECK-NEXT: rocdl.sched.barrier 0
    // CHECK-NOT: ttg.local_barrier
    // CHECK: ttg.local_store
    // CHECK: ttg.local_store
    // CHECK: gpu.barrier
    // CHECK: tt.dot
    // CHECK: } {hcu.block_pingpong.applied,
    // CHECK-SAME: hcu.block_pingpong.static_buffers
    %cold_load_a = ttg.local_load %cold_a : !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable> -> tensor<256x32xf16, #dot_a>
    %cold_load_b = ttg.local_load %cold_b : !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable> -> tensor<32x256xf16, #dot_b>
    %final = tt.dot %cold_load_a, %cold_load_b, %result#0, inputPrecision = tf32
        : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
        -> tensor<256x256xf32, #mma>
    tt.return %final : tensor<256x256xf32, #mma>
  }
}
