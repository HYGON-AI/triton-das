// RUN: split-file %s %t
// RUN: triton-opt %t/roundtrip.mlir | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: triton-opt %t/roundtrip.mlir | triton-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: triton-opt %t/invalid.mlir -split-input-file -verify-diagnostics
// RUN: triton-opt %t/buffer-read.mlir --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s --check-prefix=BUFFER
// RUN: triton-opt %t/panel-read.mlir --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse | FileCheck %s --check-prefix=PANEL
// RUN: triton-opt %t/target-parity.mlir --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx938' --canonicalize --cse | FileCheck %s --check-prefix=PARITY
// RUN: triton-opt %t/target-parity.mlir --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse | FileCheck %s --check-prefix=PARITY

// RUN: triton-opt %t/transposed-ring.mlir --tritonhcu-convert-buffer-lds-copies | FileCheck %s --check-prefix=CONVERT
// RUN: triton-opt %t/transposed-ring.mlir --tritonhcu-convert-buffer-lds-copies --tritonhcu-synchronize-buffer-lds-copies | FileCheck %s --check-prefix=SYNC
// RUN: triton-opt %t/transposed-applied.mlir --tritonhcu-synchronize-buffer-lds-copies | FileCheck %s --check-prefix=APPLIED

// 64-bank LDS layout coverage: encoding roundtrip/transpose, invalid row-bit
// swaps, native buffer-to-LDS copies, B8 panel contracts and 32/64-bank parity.

//--- roundtrip.mlir
// Row bits 2/3 exchange producer ownership. This is a layout mapping, not
// a hardware buffer-load mode. The linear bases must describe the same swap.
#linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
#lds = #ttg.hcu_buffer_lds_shared<{linearComponent = #linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 4}>
#trans_linear = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 4], [0, 16], [0, 32], [0, 64], [0, 2], [0, 8]]}, alignment = 16>
#trans = #ttg.hcu_buffer_lds_shared<{linearComponent = #trans_linear, elementBytes = 1, contiguousDim = 0, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 4}>
#smem = #ttg.shared_memory
// ROUNDTRIP: rowBitSwap = 4
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  // ROUNDTRIP-LABEL: @row_bit_swap
  tt.func public @row_bit_swap() {
    %mem = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable>
    %c1 = arith.constant 1 : i32
    %slot = ttg.memdesc_index %mem[%c1] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    %trans = ttg.memdesc_trans %slot {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #trans, #smem, mutable>
    tt.return
  }
}

//--- invalid.mlir
#linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
// expected-error @+1 {{HCU buffer-to-LDS row bit swap must select two row bits}}
#lds = #ttg.hcu_buffer_lds_shared<{linearComponent = #linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 3}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @row_bit_swap() {
    %mem = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable>
    %c1 = arith.constant 1 : i32
    %slot = ttg.memdesc_index %mem[%c1] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    tt.return
  }
}

// -----
#linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
// expected-error @+1 {{HCU buffer-to-LDS row bit swap must select two row bits}}
#lds = #ttg.hcu_buffer_lds_shared<{linearComponent = #linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 64}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @row_bit_swap() {
    %mem = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable>
    %c1 = arith.constant 1 : i32
    %slot = ttg.memdesc_index %mem[%c1] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    tt.return
  }
}

// -----
#linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
// expected-error @+1 {{linear component does not match producer interleaving}}
#lds = #ttg.hcu_buffer_lds_shared<{linearComponent = #linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 0}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @row_bit_swap() {
    %mem = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable>
    %c1 = arith.constant 1 : i32
    %slot = ttg.memdesc_index %mem[%c1] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    tt.return
  }
}

//--- buffer-read.mlir
// FP8 [128,128], four waves, 16B/lane. The producer linear bases and the
// shared bases both exchange row bits 1/2. Wrap32 separates the B128 groups.
// Keep four native 16B buffer copies and four vector reads per wave; no
// register-staged global copy or narrower LDS reads are needed.
// BUFFER-LABEL: llvm.func @probe
// BUFFER-COUNT-4: rocdl.raw.ptr.buffer.load.async.lds
// BUFFER-COUNT-4: llvm.load {{.*}} : !llvm.ptr<3> -> vector<16xi8>
// BUFFER: rocdl.mmac.f32.16x16x32.fp8.fp8
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0]], lane = [[0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0]], warp = [[2, 0], [8, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
#shared1 = #ttg.hcu_buffer_lds_shared<{linearComponent = #shared, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 4, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @probe(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<128x1xi32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_1 = arith.constant dense<56> : tensor<128x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    %cst_2 = arith.constant dense<128> : tensor<128x1xi32, #linear>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %4 = arith.muli %3, %cst_2 : tensor<128x1xi32, #linear>
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x128xi32, #linear>
    %6 = tt.broadcast %4 : tensor<128x1xi32, #linear> -> tensor<128x128xi32, #linear>
    %7 = tt.broadcast %5 : tensor<1x128xi32, #linear> -> tensor<128x128xi32, #linear>
    %8 = arith.addi %6, %7 : tensor<128x128xi32, #linear>
    %9 = amdg.buffer_load_to_local %arg0[%8] into %2 : <f8E4M3FN>[tensor<128x128xi32, #linear>]  -> <128x128xf8E4M3FN, #shared1, #smem, mutable>
    %10 = ttg.async_commit_group
    %11 = ttg.async_wait {num = 0 : i32}
    gpu.barrier
    %12 = ttg.local_load %2 : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    %13 = tt.bitcast %cst_1 : tensor<128x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    %14 = tt.dot %12, %13, %cst_0 : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<128x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x16xf32, #mma>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %17 = tt.expand_dims %15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %18 = arith.muli %17, %cst : tensor<128x1xi32, #mma>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #mma>
    %20 = tt.addptr %19, %18 : tensor<128x1x!tt.ptr<f32>, #mma>, tensor<128x1xi32, #mma>
    %21 = tt.expand_dims %16 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x16xi32, #mma>
    %22 = tt.broadcast %20 : tensor<128x1x!tt.ptr<f32>, #mma> -> tensor<128x16x!tt.ptr<f32>, #mma>
    %23 = tt.broadcast %21 : tensor<1x16xi32, #mma> -> tensor<128x16xi32, #mma>
    %24 = tt.addptr %22, %23 : tensor<128x16x!tt.ptr<f32>, #mma>, tensor<128x16xi32, #mma>
    tt.store %24, %14 : tensor<128x16x!tt.ptr<f32>, #mma>
    tt.return
  }
}

//--- panel-read.mlir
// B8 panel contracts differ in which K bit supplies logical N bit4.
// 32-bank: K bit0; 64-bank: K bit3, because each read phase has 16 lanes.
// Both encodings remain valid inputs, and each is lowered with its own inverse.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#dot = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#old = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 16], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [1, 48], [2, 64]]}, alignment = 16>
#new = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 16], [16, 0], [32, 0], [1, 32], [2, 64]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // PANEL-LABEL: llvm.func internal @old_b8_panel
  // PANEL-COUNT-2: llvm.call_intrinsic "llvm.hcu.ds.read.m32x32.b8"
  // PANEL: llvm.return
  tt.func private @old_b8_panel() -> tensor<64x128xi8, #dot> {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<64x128xi8, #old, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<64x128xi8, #old, #smem, mutable> -> tensor<64x128xi8, #dot>
    tt.return %b : tensor<64x128xi8, #dot>
  }
  // PANEL-LABEL: llvm.func internal @new_b8_panel_stage1
  // PANEL: llvm.mlir.constant(3 : i32)
  // PANEL: llvm.lshr
  // PANEL-COUNT-2: llvm.call_intrinsic "llvm.hcu.ds.read.m32x32.b8"
  // PANEL: llvm.return
  tt.func private @new_b8_panel_stage1() -> tensor<64x128xi8, #dot> {
    %one = arith.constant 1 : i32
    %buffers = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xi8, #new, #smem, mutable>
    %tile = ttg.memdesc_index %buffers[%one] : !ttg.memdesc<2x64x128xi8, #new, #smem, mutable> -> !ttg.memdesc<64x128xi8, #new, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<64x128xi8, #new, #smem, mutable> -> tensor<64x128xi8, #dot>
    tt.return %b : tensor<64x128xi8, #dot>
  }
}

//--- target-parity.mlir
// The same proved flat-layout/fragment contracts must lower on both bank
// geometries. This tests instruction support, not the matmul pass's choice of
// an MFMA instruction or bank optimality of a legacy flat layout. In particular,
// the B32 contract uses K8/kWidth2; a selected K4/kWidth1 MFMA must not inherit it.
#mma8 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#mma16 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#mma32 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 8], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#mma32_k4 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}>
#dot8 = #ttg.dot_op<{opIdx = 1, parent = #mma8, kWidth = 8}>
#dot16 = #ttg.dot_op<{opIdx = 1, parent = #mma16, kWidth = 4}>
#dot32 = #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 2}>
#dot32_k4 = #ttg.dot_op<{opIdx = 1, parent = #mma32_k4, kWidth = 1}>
#flat8 = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#flat16 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 2, order = [1, 0]}>
#flat32 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // PARITY-LABEL: llvm.func internal @b8_flat
  // PARITY-COUNT-1: llvm.call_intrinsic "llvm.hcu.ds.read.m32x32.b8"
  // PARITY: llvm.return
  tt.func private @b8_flat() -> tensor<32x32xi8, #dot8> {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x32xi8, #flat8, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<32x32xi8, #flat8, #smem, mutable> -> tensor<32x32xi8, #dot8>
    tt.return %b : tensor<32x32xi8, #dot8>
  }
  // PARITY-LABEL: llvm.func internal @b16_flat
  // PARITY-COUNT-1: llvm.call_intrinsic "llvm.hcu.ds.read.m32x16.b16"
  // PARITY: llvm.return
  tt.func private @b16_flat() -> tensor<16x32xf16, #dot16> {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #flat16, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<16x32xf16, #flat16, #smem, mutable> -> tensor<16x32xf16, #dot16>
    tt.return %b : tensor<16x32xf16, #dot16>
  }
  // PARITY-LABEL: llvm.func internal @b32_flat
  // PARITY-COUNT-1: llvm.call_intrinsic "llvm.hcu.ds.read.m32x8.b32"
  // PARITY: llvm.return
  tt.func private @b32_flat() -> tensor<8x32xf32, #dot32> {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #flat32, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<8x32xf32, #flat32, #smem, mutable> -> tensor<8x32xf32, #dot32>
    tt.return %b : tensor<8x32xf32, #dot32>
  }
  // PARITY-LABEL: llvm.func internal @b32_k4_fallback
  // PARITY-NOT: llvm.hcu.ds.read.m
  // PARITY: llvm.return
  tt.func private @b32_k4_fallback() -> tensor<8x32xf32, #dot32_k4> {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #flat32, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<8x32xf32, #flat32, #smem, mutable> -> tensor<8x32xf32, #dot32_k4>
    tt.return %b : tensor<8x32xf32, #dot32_k4>
  }
}

//--- transposed-ring.mlir
// A transposed read aliases the slot overwritten by the next copy.
// CONVERT-LABEL: @transposed_ring
// CONVERT: scf.for
// CONVERT: ttg.memdesc_trans
// CONVERT: ttg.local_load
// CONVERT: ttg.local_barrier
// CONVERT: amdg.buffer_load_to_local
// CONVERT: scf.yield
// SYNC-LABEL: @transposed_ring
// SYNC: scf.for
// SYNC: ttg.memdesc_trans
// SYNC: ttg.async_wait {num = 1 : i32}
// SYNC-NEXT: ttg.local_barrier
// SYNC-NEXT: {{.*}}ttg.local_load
// SYNC: amdg.buffer_load_to_local
// SYNC-NEXT: {{.*}}ttg.async_commit_group
// SYNC-NOT: ttg.async_wait
// SYNC: scf.yield
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
#lds = #ttg.hcu_buffer_lds_shared<{linearComponent = #linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 4}>
#trans_linear = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 4], [0, 16], [0, 32], [0, 64], [0, 2], [0, 8]]}, alignment = 16>
#trans = #ttg.hcu_buffer_lds_shared<{linearComponent = #trans_linear, elementBytes = 1, contiguousDim = 0, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 4}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @transposed_ring(%p: !tt.ptr<f8E4M3FN>, %offsets: tensor<128x128xi32, #blocked>, %out: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable>
    %initial = ttg.memdesc_index %storage[%c0] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    %last = scf.for %iv = %c0 to %n step %c1 iter_args(%slot = %initial) -> (!ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>) : i32 {
      %transposed = ttg.memdesc_trans %slot {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #trans, #smem, mutable>
      %future = amdg.buffer_load %p[%offsets] : tensor<128x128xf8E4M3FN, #blocked>
      %current = ttg.local_load %transposed : !ttg.memdesc<128x128xf8E4M3FN, #trans, #smem, mutable> -> tensor<128x128xf8E4M3FN, #blocked>
      tt.store %out, %current : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked>
      %next = ttg.memdesc_index %storage[%c1] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
      ttg.local_store %future, %next : tensor<128x128xf8E4M3FN, #blocked> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
      scf.yield %next : !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    }
    tt.return
  }
}

//--- transposed-applied.mlir
// The standalone pass must leave BPP-owned prologue copies to the scheduler.
// APPLIED-LABEL: @transposed_applied_slot
// APPLIED: amdg.buffer_load_to_local
// APPLIED-NOT: ttg.async_commit_group
// APPLIED-NOT: ttg.async_wait
// APPLIED: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [4, 0], [16, 0], [32, 0], [64, 0], [2, 0], [8, 0]]}, alignment = 16>
#lds = #ttg.hcu_buffer_lds_shared<{linearComponent = #linear, elementBytes = 1, contiguousDim = 1, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 4}>
#trans_linear = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 4], [0, 16], [0, 32], [0, 64], [0, 2], [0, 8]]}, alignment = 16>
#trans = #ttg.hcu_buffer_lds_shared<{linearComponent = #trans_linear, elementBytes = 1, contiguousDim = 0, rows = 128, rowBytes = 128, waves = 4, copyBytesPerLane = 16, rowsPerChunk = 2, wrapCount = 4, wrapStepBytes = 32, wavesPerRowGroup = 4, rowBitSwap = 4}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @transposed_applied_slot(%p: !tt.ptr<f8E4M3FN>, %offsets: tensor<128x128xi32, #blocked>, %out: tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable>
    %slot = ttg.memdesc_index %storage[%c0] : !ttg.memdesc<2x128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable>
    %copy = amdg.buffer_load_to_local %p[%offsets] into %slot : <f8E4M3FN>[tensor<128x128xi32, #blocked>] -> <128x128xf8E4M3FN, #lds, #smem, mutable>
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %transposed = ttg.memdesc_trans %slot {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #lds, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #trans, #smem, mutable>
      %current = ttg.local_load %transposed : !ttg.memdesc<128x128xf8E4M3FN, #trans, #smem, mutable> -> tensor<128x128xf8E4M3FN, #blocked>
      tt.store %out, %current : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked>
    } {hcu.block_pingpong.applied}
    tt.return
  }
}
