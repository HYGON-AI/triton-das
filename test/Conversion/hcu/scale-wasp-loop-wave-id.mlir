// RUN: triton-opt %s --convert-scf-to-cf --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --cse | FileCheck %s
// Native scale copies must hoist wave ID to each partition entry, never
// capture a function-level value or recompute readfirstlane in the K loop.
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#scale = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @scale_wasp_loop_wave_id
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  tt.func public @scale_wasp_loop_wave_id(%n: i32) {
    ttg.warp_specialize(%n) attributes {warpGroupStartIds = array<i32: 4>}
    // CHECK: default {
    // CHECK-NOT: llvm.br
    // CHECK-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: llvm.br ^bb
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: rocdl.ds.scale.copy.ds2buf
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: rocdl.ds.scale.copy.ds2buf
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: ttg.warp_yield
    default {
      %src = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %n step %c1 : i32 {
        %s0 = amdg.scale_write %src {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x8xi8, #scale, #amdg.scale_memory>
        %s1 = amdg.scale_write %src {mode = 2 : i32, row_base = 64 : i32} : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x8xi8, #scale, #amdg.scale_memory>
      }
      ttg.warp_yield
    }
    // CHECK: partition0(
    // CHECK-NOT: llvm.br
    // CHECK-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: llvm.br ^bb
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: rocdl.ds.scale.copy.ds2buf
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: rocdl.ds.scale.copy.ds2buf
    // CHECK-NOT: llvm.amdgcn.readfirstlane
    // CHECK: ttg.warp_return
    partition0(%count: i32) num_warps(4) {
      %src = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %count step %c1 : i32 {
        %s0 = amdg.scale_write %src {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x8xi8, #scale, #amdg.scale_memory>
        %s1 = amdg.scale_write %src {mode = 2 : i32, row_base = 64 : i32} : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<128x8xi8, #scale, #amdg.scale_memory>
      }
      ttg.warp_return
    }
    : (i32) -> ()
    tt.return
  }
}
