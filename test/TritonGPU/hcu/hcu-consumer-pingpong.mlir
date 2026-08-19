// RUN: triton-opt %s -split-input-file --tritonhcu-consumer-pingpong | FileCheck %s

// Two MMA consumers (wdra48 / OnePTwoC): offset MMA_tail before the K-loop and
// omit its final post-dot barrier, so both consumers can independently enter
// their epilogues. Insert Memory|Dot cluster barriers + setprio around tt.dot.
// Load partition is left alone.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: @wdra48_consumer_pingpong
// CHECK: ttg.warp_specialize
// CHECK: partition0
// CHECK-NOT: hcu.consumer_pingpong
// CHECK: ttg.warp_return
// MMA_main: no pre-loop or post-loop barrier; cluster barriers + setprio in the loop.
// CHECK: partition1
// CHECK-NOT: rocdl.s.barrier
// CHECK: scf.for
// CHECK: ttg.local_load
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK-SAME: hcu.consumer_pingpong
// CHECK: rocdl.s.setprio 1
// CHECK: tt.dot
// CHECK: rocdl.s.setprio 0
// CHECK: rocdl.s.barrier
// CHECK-SAME: hcu.consumer_pingpong
// CHECK: scf.yield
// CHECK-NOT: rocdl.s.barrier
// CHECK: ttg.warp_return
// MMA_tail: guarded pre-loop barrier and guarded non-final post-dot barrier.
// CHECK: partition2
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: rocdl.s.barrier
// CHECK-SAME: hcu.consumer_pingpong
// CHECK: scf.for
// CHECK: rocdl.s.setprio 1
// CHECK: tt.dot
// CHECK: rocdl.s.setprio 0
// CHECK: arith.addi
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: rocdl.s.barrier
// CHECK-SAME: hcu.consumer_pingpong
module attributes {hcu.wdra_topo = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 12 : i32} {
  tt.func public @wdra48_consumer_pingpong(%a: !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable>, %k: i32) {
    ttg.warp_specialize(%a, %b, %k) attributes {requestedRegisters = array<i32: 88, 88, 88>, warpGroupStartIds = array<i32: 0, 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable>, %arg2: i32) num_warps(4) {
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable>, %arg2: i32) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
      %acc = scf.for %i = %c0 to %arg2 step %c1 iter_args(%c = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
        %av = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
        %bv = ttg.memdesc_index %arg1[%c0] : !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
        %al = ttg.local_load %av : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %bl = ttg.local_load %bv : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %d = tt.dot %al, %bl, %c : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
        scf.yield %d : tensor<64x64xf32, #mma>
      }
      ttg.warp_return
    }
    partition2(%arg0: !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>, %arg1: !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable>, %arg2: i32) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
      %acc = scf.for %i = %c0 to %arg2 step %c1 iter_args(%c = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
        %av = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
        %bv = ttg.memdesc_index %arg1[%c0] : !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared1, #smem, mutable>
        %al = ttg.local_load %av : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %bl = ttg.local_load %bv : !ttg.memdesc<32x64xf16, #shared1, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %d = tt.dot %al, %bl, %c : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
        scf.yield %d : tensor<64x64xf32, #mma>
      }
      ttg.warp_return
    } : (!ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>, !ttg.memdesc<2x32x64xf16, #shared1, #smem, mutable>, i32) -> ()
    tt.return
  }
}

// -----

// Without a two-consumer WDRA topology the pass is a no-op.
// CHECK-LABEL: @skip_without_wdra_topo
// CHECK-NOT: hcu.consumer_pingpong
// CHECK-NOT: rocdl.s.setprio
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @skip_without_wdra_topo() {
    tt.return
  }
}
