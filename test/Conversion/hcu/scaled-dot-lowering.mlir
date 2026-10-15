// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse --verify-diagnostics | FileCheck %s

// One matrix dot expands into multiple native calls. No instruction-form AMD
// dot or changes to the common DotOpInterface are needed for register fragments.
// CHECK-LABEL: llvm.func @matrix_tile
// A/B share one publication barrier after both owner-wave regions.
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-SAME: "amdg.m0"
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: amdg.dot_scaled
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matrix_tile(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Compact translated views retain their byte offset. Strided views are
// rejected because the copy/bank plan assumes a compact logical pitch.
// CHECK-LABEL: llvm.func @s_contiguous_row_slice
// CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK-DAG: %[[ROW64:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: llvm.add %[[ZERO]], %[[ROW64]]
// CHECK: llvm.getelementptr
// CHECK: rocdl.ds.scale.copy.ds2buf
#blocked = #ttg.blocked<{sizePerThread=[1,4], threadsPerWarp=[4,16], warpsPerCTA=[4,1], order=[1,0]}>
#shared = #ttg.swizzled_shared<{vec=4, perPhase=1, maxPhase=1, order=[1,0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=64:i32, "ttg.num-ctas"=1:i32, ttg.target="hip:gfx946"} {
  tt.func public @s_contiguous_row_slice() {
    %src = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
    %slice = ttg.memdesc_subslice %src[64, 0] : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 128x8>
    %scale = amdg.scale_write %slice {mode = 2 : i32, row_base=0:i32} : !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 128x8> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    tt.return
  }
}

// -----

// CHECK-LABEL: llvm.func @r_contiguous_column_slice
// CHECK-DAG: %[[COL8:.*]] = llvm.mlir.constant(8 : i32)
// CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.add %[[ZERO]], %[[COL8]]
// CHECK: llvm.getelementptr
// CHECK: rocdl.ds.scale.copy.ds2buf
#blocked = #ttg.blocked<{sizePerThread=[1,4], threadsPerWarp=[4,16], warpsPerCTA=[4,1], order=[1,0]}>
#shared = #ttg.swizzled_shared<{vec=4, perPhase=1, maxPhase=1, order=[0,1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=64:i32, "ttg.num-ctas"=1:i32, ttg.target="hip:gfx946"} {
  tt.func public @r_contiguous_column_slice() {
    %src = ttg.local_alloc : () -> !ttg.memdesc<64x16xi8, #shared, #smem, mutable>
    %slice = ttg.memdesc_subslice %src[0, 8] : !ttg.memdesc<64x16xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 64x16>
    %scale = amdg.scale_write %slice {mode = 0 : i32, row_base=0:i32} : !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 64x16> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 2], rowStride = 8}>, #amdg.scale_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread=[1,4], threadsPerWarp=[4,16], warpsPerCTA=[4,1], order=[1,0]}>
#shared = #ttg.swizzled_shared<{vec=4, perPhase=1, maxPhase=1, order=[1,0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=64:i32, "ttg.num-ctas"=1:i32, ttg.target="hip:gfx946"} {
  tt.func public @reject_s_strided() {
    %src = ttg.local_alloc : () -> !ttg.memdesc<64x16xi8, #shared, #smem, mutable>
    %slice = ttg.memdesc_subslice %src[0, 8] : !ttg.memdesc<64x16xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 64x16>
    // expected-error @+1 {{scale copy requires a compact LDS view}}
    %scale = amdg.scale_write %slice {mode = 2 : i32, row_base=0:i32} : !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 64x16> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread=[1,4], threadsPerWarp=[4,16], warpsPerCTA=[4,1], order=[1,0]}>
#shared = #ttg.swizzled_shared<{vec=4, perPhase=1, maxPhase=1, order=[0,1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=64:i32, "ttg.num-ctas"=1:i32, ttg.target="hip:gfx946"} {
  tt.func public @reject_r_strided() {
    %src = ttg.local_alloc : () -> !ttg.memdesc<128x8xi8, #shared, #smem, mutable>
    %slice = ttg.memdesc_subslice %src[64, 0] : !ttg.memdesc<128x8xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 128x8>
    // expected-error @+1 {{scale copy requires a compact LDS view}}
    %scale = amdg.scale_write %slice {mode = 0 : i32, row_base=0:i32} : !ttg.memdesc<64x8xi8, #shared, #smem, mutable, 128x8> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 2], rowStride = 8}>, #amdg.scale_memory>
    tt.return
  }
}

// -----

// Separate write blocks keep the publication barrier at the common consumer.
// CHECK-LABEL: llvm.func @cross_block
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @cross_block(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    cf.br ^copy_b
  ^copy_b:
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Reuse an existing publication barrier after the two writes.
// CHECK-LABEL: llvm.func @existing_barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @existing_barrier(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    ttg.local_barrier
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Native Shared membar reuses Scale publication for a later local_load.
// The pending LDS write must not cause another barrier after the dot.
// CHECK-LABEL: llvm.func @shared_barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @shared_barrier(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>,
                    %values: tensor<64x8xi8, #blocked>,
                    %dst: tensor<64x8x!tt.ptr<i8>, #blocked>) {
    %lds = ttg.local_alloc %values : (tensor<64x8xi8, #blocked>) -> !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    %loaded = ttg.local_load %lds : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> tensor<64x8xi8, #blocked>
    tt.store %dst, %loaded : tensor<64x8x!tt.ptr<i8>, #blocked>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// A bare gpu.barrier does not complete ds2buf; keep the local barrier.
// CHECK-LABEL: llvm.func @bare_barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.barrier
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bare_barrier(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    gpu.barrier
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// A bare gpu.barrier does not complete ds2buf; keep the local barrier.
// CHECK-LABEL: llvm.func @bare_barrier_cross_block
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.barrier
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bare_barrier_cross_block(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    gpu.barrier
    cf.br ^compute
  ^compute:
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// A local_barrier with its lgkm wait removed cannot publish Scale writes.
// CHECK-LABEL: llvm.func @deferred_barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.s.barrier
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @deferred_barrier(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    ttg.local_barrier {hcu.block_pingpong.drop_lgkmcnt}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----


// A CTA-uniform conditional uses the same CFG analysis as native membar.
// CHECK-LABEL: llvm.func @uniform_guard
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: scf.if
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @uniform_guard(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>, %cond: i1) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    scf.if %cond {
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    ttg.local_barrier
    tt.return
  }
}

// -----


// Drain ds2buf at the existing WASP consumer rendezvous; do not add a second barrier.
// CHECK-LABEL: llvm.func @consumer_pingpong_barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK-NOT: rocdl.s.barrier
// CHECK: rocdl.ds.scale.copy.ds2buf
// CHECK: rocdl.s.waitcnt
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NOT: rocdl.s.barrier
// CHECK: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// CHECK-NOT: rocdl.s.barrier
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @consumer_pingpong_barrier(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32, row_base = 0 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32, row_base = 32 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    rocdl.s.barrier {hcu.consumer_pingpong}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
