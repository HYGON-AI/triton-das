// RUN: triton-opt %s --tritonhcu-mls-lowering --convert-scf-to-cf --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s
// RUN: triton-opt %s --tritonhcu-mls-lowering --convert-scf-to-cf --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s --check-prefix=BPS
// RUN: sed 's/indices = \[%c0, %c0\]/indices = [%c0, %c0] boundaryCheck = [0, 1]/g' %s | triton-opt --tritonhcu-mls-lowering --convert-scf-to-cf --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s --check-prefix=BPS

// Expected placement for both buffer-to-LDS and MLS lowering:
//   partition0 {                 partition1 {
//     wave_id = ...                wave_id = ...
//     for (...) { use wave_id }    for (...) { use wave_id }
//   }                            }
// Move out of the loop, not out of its partition. Each partition owns its
// wave-ID SSA value; no isolated region captures a value from the function.
// SCF lowering makes the loop a separate CFG block. Check readfirstlane before
// the first branch, and reject another readfirstlane throughout the loop.
// These are lowering tests; synchronization/runtime coverage lives in
// python/test/unit/language/test_hcu_gluon_pipeline.py.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#copy = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0]], lane = [[0, 16], [0, 32], [0, 64], [1, 0], [8, 0], [16, 0]], warp = [[2, 0], [4, 0]], block = []}>
#wrap_linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [8, 0], [16, 0], [32, 0], [64, 0], [2, 0], [4, 0]]}, alignment = 16>
#wrapped = #ttg.hcu_buffer_lds_shared<{linearComponent = #wrap_linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 16}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2}>
#dot = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#regs = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @wasp_loop_wave_id
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: default {
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_yield
  // CHECK: partition0(
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_return
  // CHECK: partition1(
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_return
  tt.func public @wasp_loop_wave_id(%base: !tt.ptr<f32>, %n: i32) {
    ttg.warp_specialize(%base, %n) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      %dst = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %offset = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
      scf.for %i = %c0 to %n step %c1 : i32 {
        %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<256xi32, #blocked>] -> <256xf32, #shared, #smem, mutable>
      }
      ttg.warp_yield
    }
    partition0(%p: !tt.ptr<f32>, %count: i32) num_warps(4) {
      %dst = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %offset = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
      scf.for %i = %c0 to %count step %c1 : i32 {
        %token = amdg.buffer_load_to_local %p[%offset] into %dst : <f32>[tensor<256xi32, #blocked>] -> <256xf32, #shared, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%p: !tt.ptr<f32>, %count: i32) num_warps(4) {
      %dst = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %offset = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #blocked>
      scf.for %i = %c0 to %count step %c1 : i32 {
        %token = amdg.buffer_load_to_local %p[%offset] into %dst : <f32>[tensor<256xi32, #blocked>] -> <256xf32, #shared, #smem, mutable>
      }
      ttg.warp_return
    }
    : (!tt.ptr<f32>, i32) -> ()
    tt.return
  }

  // Wrapped buffer loads must also support the WASP default region.
  // CHECK-LABEL: llvm.func @wasp_wrapped_loop_wave_id
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: default {
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_yield
  // CHECK: partition0(
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_return
  // CHECK: partition1(
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_return
  tt.func public @wasp_wrapped_loop_wave_id(%base: !tt.ptr<f8E4M3FN>, %offsets: tensor<128x128xi32, #copy>, %n: i32) {
    ttg.warp_specialize(%base, %offsets, %n) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      %dst = ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN, #wrapped, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %n step %c1 : i32 {
        %token = amdg.buffer_load_to_local %base[%offsets] into %dst {contiguity = 16 : i32} : <f8E4M3FN>[tensor<128x128xi32, #copy>] -> <128x128xf8E4M3FN, #wrapped, #smem, mutable>
      }
      ttg.warp_yield
    }
    partition0(%p: !tt.ptr<f8E4M3FN>, %offset: tensor<128x128xi32, #copy>, %count: i32) num_warps(4) {
      %dst = ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN, #wrapped, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %count step %c1 : i32 {
        %token = amdg.buffer_load_to_local %p[%offset] into %dst {contiguity = 16 : i32} : <f8E4M3FN>[tensor<128x128xi32, #copy>] -> <128x128xf8E4M3FN, #wrapped, #smem, mutable>
      }
      ttg.warp_return
    }
    partition1(%p: !tt.ptr<f8E4M3FN>, %offset: tensor<128x128xi32, #copy>, %count: i32) num_warps(4) {
      %dst = ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN, #wrapped, #smem, mutable>
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %count step %c1 : i32 {
        %token = amdg.buffer_load_to_local %p[%offset] into %dst {contiguity = 16 : i32} : <f8E4M3FN>[tensor<128x128xi32, #copy>] -> <128x128xf8E4M3FN, #wrapped, #smem, mutable>
      }
      ttg.warp_return
    }
    : (!tt.ptr<f8E4M3FN>, tensor<128x128xi32, #copy>, i32) -> ()
    tt.return
  }

  // Start from tt.matrix_load with an explicit MLS encoding: the MLS lowering
  // materializes amdg.matrix_load_to_local and its shared-memory layout.
  // CHECK-LABEL: llvm.func @mls_loop_wave_id
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: llvm.return
  tt.func public @mls_loop_wave_id(%p: !tt.ptr<f8E4M3FN>, %out: !tt.ptr<f8E4M3FN>, %count: i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %size = arith.constant 128 : i32
      %s1 = arith.constant 1 : i64
      %s128 = arith.constant 128 : i64
      %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %cols = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #regs}>>
      %stride = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %row_offsets = arith.muli %rows, %stride : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %r = tt.expand_dims %row_offsets {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>> -> tensor<128x1xi32, #regs>
      %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #regs}>> -> tensor<1x128xi32, #regs>
      %rr = tt.broadcast %r : tensor<128x1xi32, #regs> -> tensor<128x128xi32, #regs>
      %cc = tt.broadcast %c : tensor<1x128xi32, #regs> -> tensor<128x128xi32, #regs>
      %offsets = arith.addi %rr, %cc : tensor<128x128xi32, #regs>
      %bases = tt.splat %out : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>
      %ptrs = tt.addptr %bases, %offsets : tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>, tensor<128x128xi32, #regs>
      scf.for %i = %c0 to %count step %c1 : i32 {
        %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128], tensorShape = [128, 128], indices = [%c0, %c0] {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>} : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #dot>
        %values = ttg.convert_layout %tile : tensor<128x128xf8E4M3FN, #dot> -> tensor<128x128xf8E4M3FN, #regs>
        tt.store %ptrs, %values : tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>
      }
    tt.return
  }

  // CHECK-LABEL: llvm.func @wasp_mls_loop_wave_id
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: partition0(
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_return
  // CHECK: partition1(
  // CHECK-NOT: llvm.br
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.readfirstlane"
  // CHECK-NOT: ^bb
  // CHECK: llvm.br ^bb
  // CHECK-NOT: llvm.amdgcn.readfirstlane
  // CHECK: ttg.warp_return
  tt.func public @wasp_mls_loop_wave_id(%base: !tt.ptr<f8E4M3FN>, %out0: !tt.ptr<f8E4M3FN>, %out1: !tt.ptr<f8E4M3FN>, %n: i32) {
    // The caller supplies separate output tiles for the two partitions.
    ttg.warp_specialize(%base, %out0, %out1, %n) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%p: !tt.ptr<f8E4M3FN>, %out: !tt.ptr<f8E4M3FN>, %unused: !tt.ptr<f8E4M3FN>, %count: i32) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %size = arith.constant 128 : i32
      %s1 = arith.constant 1 : i64
      %s128 = arith.constant 128 : i64
      %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %cols = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #regs}>>
      %stride = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %row_offsets = arith.muli %rows, %stride : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %r = tt.expand_dims %row_offsets {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>> -> tensor<128x1xi32, #regs>
      %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #regs}>> -> tensor<1x128xi32, #regs>
      %rr = tt.broadcast %r : tensor<128x1xi32, #regs> -> tensor<128x128xi32, #regs>
      %cc = tt.broadcast %c : tensor<1x128xi32, #regs> -> tensor<128x128xi32, #regs>
      %offsets = arith.addi %rr, %cc : tensor<128x128xi32, #regs>
      %bases = tt.splat %out : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>
      %ptrs = tt.addptr %bases, %offsets : tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>, tensor<128x128xi32, #regs>
      scf.for %i = %c0 to %count step %c1 : i32 {
        rocdl.hcu.s.abarrier.seq %c1
        %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128], tensorShape = [128, 128], indices = [%c0, %c0] {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>} : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #dot>
        %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
        %values = ttg.convert_layout %tile : tensor<128x128xf8E4M3FN, #dot> -> tensor<128x128xf8E4M3FN, #regs>
        tt.store %ptrs, %values : tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>
      }
      ttg.warp_return
    }
    partition1(%p: !tt.ptr<f8E4M3FN>, %unused: !tt.ptr<f8E4M3FN>, %out: !tt.ptr<f8E4M3FN>, %count: i32) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %size = arith.constant 128 : i32
      %s1 = arith.constant 1 : i64
      %s128 = arith.constant 128 : i64
      %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %cols = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #regs}>>
      %stride = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %row_offsets = arith.muli %rows, %stride : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>>
      %r = tt.expand_dims %row_offsets {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #regs}>> -> tensor<128x1xi32, #regs>
      %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #regs}>> -> tensor<1x128xi32, #regs>
      %rr = tt.broadcast %r : tensor<128x1xi32, #regs> -> tensor<128x128xi32, #regs>
      %cc = tt.broadcast %c : tensor<1x128xi32, #regs> -> tensor<128x128xi32, #regs>
      %offsets = arith.addi %rr, %cc : tensor<128x128xi32, #regs>
      %bases = tt.splat %out : !tt.ptr<f8E4M3FN> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>
      %ptrs = tt.addptr %bases, %offsets : tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>, tensor<128x128xi32, #regs>
      scf.for %i = %c0 to %count step %c1 : i32 {
        %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128], tensorShape = [128, 128], indices = [%c0, %c0] {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>} : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #dot>
        %values = ttg.convert_layout %tile : tensor<128x128xf8E4M3FN, #dot> -> tensor<128x128xf8E4M3FN, #regs>
        tt.store %ptrs, %values : tensor<128x128x!tt.ptr<f8E4M3FN>, #regs>
      }
      ttg.warp_return
    }
    : (!tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, !tt.ptr<f8E4M3FN>, i32) -> ()
    tt.return
  }
}

// Only partition0 opens a seq track; partition1 is WASP without a seq.
// Check BPS in both the shared-descriptor and padded
// per-instruction-descriptor lowering paths.
// BPS-LABEL: llvm.func @mls_loop_wave_id
// BPS: rocdl.matrix.load.64x16.b8 {{.*}}, false]
// BPS-NOT: rocdl.matrix.load.64x16.b8 {{.*}}, true]
// BPS-LABEL: llvm.func @wasp_mls_loop_wave_id
// BPS: partition0(
// BPS: rocdl.matrix.load.64x16.b8 {{.*}}, true]
// BPS-NOT: rocdl.matrix.load.64x16.b8 {{.*}}, false]
// BPS: partition1(
// BPS: rocdl.matrix.load.64x16.b8 {{.*}}, false]
// BPS-NOT: rocdl.matrix.load.64x16.b8 {{.*}}, true]
