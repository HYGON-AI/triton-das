// RUN: triton-opt %s -tritonhcu-block-pingpong | FileCheck %s
// RUN: triton-opt %s -tritonhcu-block-pingpong | FileCheck %s --check-prefix=PANEL
// RUN: triton-opt %s -tritonhcu-block-pingpong | FileCheck %s --check-prefix=BIT8
// RUN: triton-opt %s -tritonhcu-block-pingpong | FileCheck %s --check-prefix=BIT8-GENERIC

#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared_a = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#shared_panel_b = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 0], [8, 0], [16, 0], [0, 32], [0, 64], [0, 128]]}, alignment = 16>
#blocked_a_bit8 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b_col_bit8 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked_b_row_bit8 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma_bit8 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot_a_bit8 = #ttg.dot_op<{opIdx = 0, parent = #mma_bit8, kWidth = 8}>
#dot_b_bit8 = #ttg.dot_op<{opIdx = 1, parent = #mma_bit8, kWidth = 8}>
#shared_a_bit8 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared_b_bit8 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#shared_panel_b_bit8 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 16], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [1, 48], [2, 64], [0, 128]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @buffer_load_block_pingpong
  tt.func public @buffer_load_block_pingpong(
      %a_ptrs: tensor<256x32x!tt.ptr<f16>, #blocked_a>,
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b>) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable>
    %slot_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
    %slot_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x256xf16, #shared_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
    %init_a = tt.load %a_ptrs : tensor<256x32x!tt.ptr<f16>, #blocked_a>
    %init_b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>

    // CHECK: gpu.barrier
    // CHECK-NEXT: rocdl.s.setprio 1
    // CHECK: %[[TID:.+]] = rocdl.workitem.id.x
    // CHECK: %[[WAVE:.+]] = arith.shrui %[[TID]]
    // CHECK: %[[UNIFORM_WAVE:.+]] = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%[[WAVE]])
    // CHECK: %[[LOW:.+]] = arith.cmpi ult, %[[UNIFORM_WAVE]]
    // CHECK: %[[HIGH:.+]] = arith.cmpi uge, %[[UNIFORM_WAVE]]
    // CHECK: amdg.cond_barrier %[[LOW]]
    // CHECK: scf.for
    %result:5 = scf.for %iv = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %zero, %current_a = %slot_a, %current_b = %slot_b,
                  %prefetch_a = %init_a, %prefetch_b = %init_b)
        -> (tensor<256x256xf32, #mma>,
            !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>,
            !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>,
            tensor<256x32xf16, #blocked_a>, tensor<32x256xf16, #blocked_b>) : i32 {
      %anchor = arith.addi %iv, %c0_i32 : i32
      %global_a = tt.load %a_ptrs : tensor<256x32x!tt.ptr<f16>, #blocked_a>
      %global_b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %local_a = ttg.local_load %current_a : !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable> -> tensor<256x32xf16, #dot_a>
      %local_b = ttg.local_load %current_b : !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable> -> tensor<32x256xf16, #dot_b>
      %next = tt.dot %local_a, %local_b, %acc, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>
      ttg.local_store %prefetch_a, %current_a : tensor<256x32xf16, #blocked_a> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
      ttg.local_store %prefetch_b, %current_b : tensor<32x256xf16, #blocked_b> -> !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>
      scf.yield %next, %current_a, %current_b, %global_a, %global_b
          : tensor<256x256xf32, #mma>,
            !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>,
            !ttg.memdesc<32x256xf16, #shared_b, #smem, mutable>,
            tensor<256x32xf16, #blocked_a>, tensor<32x256xf16, #blocked_b>
    } {hcu.block_pingpong.buffer_load_stage3,
       ttg.pipeline.hcu_local_store_stage = 1 : i32,
       ttg.pipeline.hcu_num_buffers = 2 : i32}

    // M0: K-half0 LDS reads, A publish/issue, control-only barrier.
    // CHECK: %[[A0:.+]] = ttg.local_load {{.+}} -> tensor<256x16xf16
    // CHECK: %[[B0:.+]] = ttg.local_load {{.+}} -> tensor<16x256xf16
    // CHECK: ttg.local_store
    // CHECK: tt.load
    // CHECK: rocdl.sched.barrier 0
    // CHECK-NEXT: rocdl.s.barrier
    // CHECK-NEXT: rocdl.sched.barrier 0

    // D0: first K16 half, kept as one uninterrupted dot cluster.
    // Generic/swizzled B keeps half1 in M1; unlike panel B, it must not extend
    // the B1 register live range across D0.
    // CHECK-NOT: ttg.local_load {{.+}} -> tensor<16x256xf16
    // CHECK: rocdl.sched.barrier 0
    // CHECK-NEXT: rocdl.s.setprio 0
    // CHECK-NEXT: %[[DOT0:.+]] = tt.dot %[[A0]], %[[B0]], %{{.+}}
    // CHECK-NEXT: rocdl.s.setprio 1
    // CHECK-NEXT: rocdl.sched.barrier 0
    // CHECK: rocdl.s.barrier

    // M1: K-half1 LDS reads, B publish/issue, the only full loop barrier.
    // CHECK: %[[A1:.+]] = ttg.local_load {{.+}} -> tensor<256x16xf16
    // CHECK: %[[B1:.+]] = ttg.local_load {{.+}} -> tensor<16x256xf16
    // CHECK: ttg.local_store
    // CHECK: tt.load
    // CHECK: rocdl.sched.barrier 0
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: rocdl.sched.barrier 0

    // D1: second K16 half chains from D0 and ends at a control barrier.
    // CHECK: rocdl.sched.barrier 0
    // CHECK-NEXT: rocdl.s.setprio 0
    // CHECK-NEXT: %[[DOT1:.+]] = tt.dot %[[A1]], %[[B1]], %[[DOT0]]
    // CHECK-NEXT: rocdl.s.setprio 1
    // CHECK-NEXT: rocdl.sched.barrier 0
    // CHECK: rocdl.s.barrier
    // CHECK: } {hcu.block_pingpong.applied
    // CHECK-SAME: tt.loop_unroll_factor = 2 : i32
    // CHECK: amdg.cond_barrier %[[HIGH]]

    tt.return
  }

  // PANEL-LABEL: tt.func public @buffer_load_block_pingpong_panel_b
  tt.func public @buffer_load_block_pingpong_panel_b(
      %a_ptrs: tensor<256x32x!tt.ptr<f16>, #blocked_a>,
      %b_ptrs: tensor<32x256x!tt.ptr<f16>, #blocked_b>) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x256xf16, #shared_panel_b, #smem, mutable>
    %slot_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x256x32xf16, #shared_a, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
    %slot_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x32x256xf16, #shared_panel_b, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared_panel_b, #smem, mutable>
    %init_a = tt.load %a_ptrs : tensor<256x32x!tt.ptr<f16>, #blocked_a>
    %init_b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>

    %result:5 = scf.for %iv = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %zero, %current_a = %slot_a, %current_b = %slot_b,
                  %prefetch_a = %init_a, %prefetch_b = %init_b)
        -> (tensor<256x256xf32, #mma>,
            !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>,
            !ttg.memdesc<32x256xf16, #shared_panel_b, #smem, mutable>,
            tensor<256x32xf16, #blocked_a>, tensor<32x256xf16, #blocked_b>) : i32 {
      %anchor = arith.addi %iv, %c0_i32 : i32
      %global_a = tt.load %a_ptrs : tensor<256x32x!tt.ptr<f16>, #blocked_a>
      %global_b = tt.load %b_ptrs : tensor<32x256x!tt.ptr<f16>, #blocked_b>
      %local_a = ttg.local_load %current_a : !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable> -> tensor<256x32xf16, #dot_a>
      %local_b = ttg.local_load %current_b : !ttg.memdesc<32x256xf16, #shared_panel_b, #smem, mutable> -> tensor<32x256xf16, #dot_b>
      %next = tt.dot %local_a, %local_b, %acc, inputPrecision = tf32
          : tensor<256x32xf16, #dot_a> * tensor<32x256xf16, #dot_b>
          -> tensor<256x256xf32, #mma>
      ttg.local_store %prefetch_a, %current_a : tensor<256x32xf16, #blocked_a> -> !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>
      ttg.local_store %prefetch_b, %current_b : tensor<32x256xf16, #blocked_b> -> !ttg.memdesc<32x256xf16, #shared_panel_b, #smem, mutable>
      scf.yield %next, %current_a, %current_b, %global_a, %global_b
          : tensor<256x256xf32, #mma>,
            !ttg.memdesc<256x32xf16, #shared_a, #smem, mutable>,
            !ttg.memdesc<32x256xf16, #shared_panel_b, #smem, mutable>,
            tensor<256x32xf16, #blocked_a>, tensor<32x256xf16, #blocked_b>
    } {hcu.block_pingpong.buffer_load_stage3,
       ttg.pipeline.hcu_local_store_stage = 1 : i32,
       ttg.pipeline.hcu_num_buffers = 2 : i32}

    // Panel B half1 must be read before D0.  Its two ds_read_m operations then
    // overlap the 32 half0 MMACs instead of extending M1's critical path.
    // PANEL: %[[PANEL_A0:.+]] = ttg.local_load {{.+}} -> tensor<256x16xf16
    // PANEL: %[[PANEL_B0:.+]] = ttg.local_load {{.+}} -> tensor<16x256xf16
    // PANEL: %[[PANEL_B1:.+]] = ttg.local_load {{.+}} -> tensor<16x256xf16
    // PANEL: rocdl.s.barrier
    // PANEL: rocdl.s.setprio 0
    // PANEL-NEXT: %[[PANEL_DOT0:.+]] = tt.dot %[[PANEL_A0]], %[[PANEL_B0]], %{{.+}}
    // PANEL: rocdl.s.barrier
    // PANEL: %[[PANEL_A1:.+]] = ttg.local_load {{.+}} -> tensor<256x16xf16
    // PANEL-NOT: ttg.local_load {{.+}} -> tensor<16x256xf16
    // PANEL: rocdl.s.setprio 0
    // PANEL-NEXT: %[[PANEL_DOT1:.+]] = tt.dot %[[PANEL_A1]], %[[PANEL_B1]], %[[PANEL_DOT0]]
    // PANEL: } {hcu.block_pingpong.applied

    tt.return
  }

  // BIT8-GENERIC-LABEL: tt.func public @bit8_generic_schedule
  tt.func public @bit8_generic_schedule(
      %a_ptrs: tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_bit8>,
      %b_ptrs: tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_col_bit8>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma_bit8>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable>
    %slot_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable> -> !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>
    %slot_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable> -> !ttg.memdesc<64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable>
    %init_a = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_bit8>
    %init_b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_col_bit8>

    %result:5 = scf.for %iv = %c0_i32 to %c128_i32 step %c64_i32
        iter_args(%acc = %zero, %current_a = %slot_a, %current_b = %slot_b,
                  %prefetch_a = %init_a, %prefetch_b = %init_b)
        -> (tensor<256x256xf32, #mma_bit8>,
            !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>,
            !ttg.memdesc<64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable>,
            tensor<256x64xf8E4M3FN, #blocked_a_bit8>,
            tensor<64x256xf8E4M3FN, #blocked_b_col_bit8>) : i32 {
      %anchor = arith.addi %iv, %c0_i32 : i32
      %global_a = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_bit8>
      %global_b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_col_bit8>
      %local_a = ttg.local_load %current_a : !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable> -> tensor<256x64xf8E4M3FN, #dot_a_bit8>
      %local_b = ttg.local_load %current_b : !ttg.memdesc<64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable> -> tensor<64x256xf8E4M3FN, #dot_b_bit8>
      %next = tt.dot %local_a, %local_b, %acc, inputPrecision = tf32
          : tensor<256x64xf8E4M3FN, #dot_a_bit8> * tensor<64x256xf8E4M3FN, #dot_b_bit8>
          -> tensor<256x256xf32, #mma_bit8>
      ttg.local_store %prefetch_a, %current_a : tensor<256x64xf8E4M3FN, #blocked_a_bit8> -> !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>
      ttg.local_store %prefetch_b, %current_b : tensor<64x256xf8E4M3FN, #blocked_b_col_bit8> -> !ttg.memdesc<64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable>
      scf.yield %next, %current_a, %current_b, %global_a, %global_b
          : tensor<256x256xf32, #mma_bit8>,
            !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>,
            !ttg.memdesc<64x256xf8E4M3FN, #shared_b_bit8, #smem, mutable>,
            tensor<256x64xf8E4M3FN, #blocked_a_bit8>,
            tensor<64x256xf8E4M3FN, #blocked_b_col_bit8>
    } {hcu.block_pingpong.buffer_load_stage3,
       ttg.pipeline.hcu_local_store_stage = 1 : i32,
       ttg.pipeline.hcu_num_buffers = 2 : i32}

    // B1 and the future B load are issued in M0 for every generic Bit8 dot and
    // carried across D0; A1 remains in M1.
    // BIT8-GENERIC: %[[B0:.+]] = ttg.local_load {{.+}} -> tensor<32x256xf8E4M3FN
    // BIT8-GENERIC: %[[A0:.+]] = ttg.local_load {{.+}} -> tensor<256x32xf8E4M3FN
    // BIT8-GENERIC: rocdl.sched.barrier 0
    // BIT8-GENERIC-NEXT: %[[FUTURE_A:.+]] = tt.load {{.+}} : tensor<256x64x!tt.ptr<f8E4M3FN>
    // BIT8-GENERIC-NEXT: %[[FUTURE_B:.+]] = tt.load {{.+}} : tensor<64x256x!tt.ptr<f8E4M3FN>
    // BIT8-GENERIC: %[[B1:.+]] = ttg.local_load {{.+}} -> tensor<32x256xf8E4M3FN
    // BIT8-GENERIC-NOT: ttg.local_load {{.+}} -> tensor<256x32xf8E4M3FN
    // BIT8-GENERIC: rocdl.s.barrier
    // BIT8-GENERIC: %[[DOT0:.+]] = tt.dot %[[A0]], %[[B0]], %{{.+}}
    // BIT8-GENERIC: %[[A1:.+]] = ttg.local_load {{.+}} -> tensor<256x32xf8E4M3FN
    // BIT8-GENERIC: ttg.local_store {{.+}} : tensor<256x64xf8E4M3FN
    // BIT8-GENERIC: ttg.local_store {{.+}} : tensor<64x256xf8E4M3FN
    // BIT8-GENERIC: gpu.barrier
    // BIT8-GENERIC: tt.dot %[[A1]], %[[B1]], %[[DOT0]]
    // BIT8-GENERIC: } {hcu.block_pingpong.applied

    tt.return
  }

  // BIT8-LABEL: tt.func public @bit8_panel_schedule
  tt.func public @bit8_panel_schedule(
      %a_ptrs: tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_bit8>,
      %b_ptrs: tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_row_bit8>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma_bit8>
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable>
    %slot_a = ttg.memdesc_index %alloc_a[%c0_i32] : !ttg.memdesc<2x256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable> -> !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>
    %slot_b = ttg.memdesc_index %alloc_b[%c0_i32] : !ttg.memdesc<2x64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable> -> !ttg.memdesc<64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable>
    %init_a = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_bit8>
    %init_b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_row_bit8>

    %result:5 = scf.for %iv = %c0_i32 to %c128_i32 step %c64_i32
        iter_args(%acc = %zero, %current_a = %slot_a, %current_b = %slot_b,
                  %prefetch_a = %init_a, %prefetch_b = %init_b)
        -> (tensor<256x256xf32, #mma_bit8>,
            !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>,
            !ttg.memdesc<64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable>,
            tensor<256x64xf8E4M3FN, #blocked_a_bit8>,
            tensor<64x256xf8E4M3FN, #blocked_b_row_bit8>) : i32 {
      %anchor = arith.addi %iv, %c0_i32 : i32
      %global_a = tt.load %a_ptrs : tensor<256x64x!tt.ptr<f8E4M3FN>, #blocked_a_bit8>
      %global_b = tt.load %b_ptrs : tensor<64x256x!tt.ptr<f8E4M3FN>, #blocked_b_row_bit8>
      %local_a = ttg.local_load %current_a : !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable> -> tensor<256x64xf8E4M3FN, #dot_a_bit8>
      %local_b = ttg.local_load %current_b : !ttg.memdesc<64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable> -> tensor<64x256xf8E4M3FN, #dot_b_bit8>
      %next = tt.dot %local_a, %local_b, %acc, inputPrecision = tf32
          : tensor<256x64xf8E4M3FN, #dot_a_bit8> * tensor<64x256xf8E4M3FN, #dot_b_bit8>
          -> tensor<256x256xf32, #mma_bit8>
      ttg.local_store %prefetch_a, %current_a : tensor<256x64xf8E4M3FN, #blocked_a_bit8> -> !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>
      ttg.local_store %prefetch_b, %current_b : tensor<64x256xf8E4M3FN, #blocked_b_row_bit8> -> !ttg.memdesc<64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable>
      scf.yield %next, %current_a, %current_b, %global_a, %global_b
          : tensor<256x256xf32, #mma_bit8>,
            !ttg.memdesc<256x64xf8E4M3FN, #shared_a_bit8, #smem, mutable>,
            !ttg.memdesc<64x256xf8E4M3FN, #shared_panel_b_bit8, #smem, mutable>,
            tensor<256x64xf8E4M3FN, #blocked_a_bit8>,
            tensor<64x256xf8E4M3FN, #blocked_b_row_bit8>
    } {hcu.block_pingpong.buffer_load_stage3,
       ttg.pipeline.hcu_local_store_stage = 1 : i32,
       ttg.pipeline.hcu_num_buffers = 2 : i32}

    // BIT8: scf.for
    // BIT8: rocdl.sched.barrier 0
    // BIT8-NEXT: %[[FUTURE_B:.+]] = tt.load {{.+}} : tensor<64x256x!tt.ptr<f8E4M3FN>
    // BIT8-NEXT: %[[FUTURE_A:.+]] = tt.load {{.+}} : tensor<256x64x!tt.ptr<f8E4M3FN>
    // BIT8: ttg.local_load {{.+}} -> tensor<32x256xf8E4M3FN
    // BIT8-NEXT: ttg.local_store
    // BIT8: rocdl.s.barrier
    // BIT8: tt.dot
    // BIT8: %[[A1:.+]] = ttg.local_load {{.+}} -> tensor<256x32xf8E4M3FN
    // BIT8-NEXT: rocdl.sched.barrier 0
    // BIT8: ttg.local_store {{.+}} : tensor<64x256xf8E4M3FN
    // BIT8-NEXT: rocdl.sched.barrier 0
    // BIT8-NEXT: gpu.barrier
    // BIT8-NEXT: rocdl.sched.barrier 0
    // BIT8: tt.dot %[[A1]]
    // BIT8: } {hcu.block_pingpong.applied

    tt.return
  }
}
