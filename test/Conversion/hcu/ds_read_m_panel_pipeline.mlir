// Verify that a panel-major allocation remains eligible for ds_read_m after
// software pipelining adds a leading buffer dimension and memdesc_index selects
// one rank-2 [BK, BN] tile.
// RUN: triton-opt %s --convert-scf-to-cf --convert-index-to-llvm --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx938' | FileCheck %s

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#mma_n64 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#mma_b8 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2, tilesPerWarp = [1, 2]}>
#panel = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 0], [8, 0], [16, 0], [0, 32], [0, 64]]}, alignment = 16>
#panel_b8 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 16], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [1, 48], [2, 64]]}, alignment = 16>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @panel_pipeline_view
  // CHECK: llvm.mlir.constant(4096 : i32)
  // CHECK-COUNT-2: llvm.call_intrinsic "llvm.hcu.ds.read.m32x16.b16"
  tt.func public @panel_pipeline_view() {
    %index = arith.constant 1 : i32
    %buffers = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #panel, #smem, mutable>
    %tile = ttg.memdesc_index %buffers[%index] : !ttg.memdesc<2x32x128xf16, #panel, #smem, mutable> -> !ttg.memdesc<32x128xf16, #panel, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<32x128xf16, #panel, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return
  }

  // A K16-aligned subslice is a pure base translation inside every physical
  // N32 panel, so both halves remain eligible for one matrix read each.
  // CHECK-LABEL: llvm.func @panel_k_subslice
  // CHECK-COUNT-2: llvm.call_intrinsic "llvm.hcu.ds.read.m32x16.b16"
  tt.func public @panel_k_subslice() {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #panel, #smem, mutable>
    %k0 = ttg.memdesc_subslice %tile[0, 0] : !ttg.memdesc<32x128xf16, #panel, #smem, mutable> -> !ttg.memdesc<16x128xf16, #panel, #smem, mutable, 32x128>
    %k1 = ttg.memdesc_subslice %tile[16, 0] : !ttg.memdesc<32x128xf16, #panel, #smem, mutable> -> !ttg.memdesc<16x128xf16, #panel, #smem, mutable, 32x128>
    %b0 = ttg.local_load %k0 : !ttg.memdesc<16x128xf16, #panel, #smem, mutable, 32x128> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %b1 = ttg.local_load %k1 : !ttg.memdesc<16x128xf16, #panel, #smem, mutable, 32x128> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return
  }

  // Bit8 rotates adjacent panel bases through physical K rows 0/1/2/3.  The
  // read address XORs the same low panel-id bits back into k_addr and retains
  // native ds_read_m lowering.
  // CHECK-LABEL: llvm.func @panel_b8_k_skew
  // CHECK: llvm.and {{.*}}, %{{.*}} : i32
  // CHECK: llvm.xor {{.*}}, %{{.*}} : i32
  // CHECK-COUNT-2: llvm.call_intrinsic "llvm.hcu.ds.read.m32x32.b8"
  tt.func public @panel_b8_k_skew() {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<64x128xi8, #panel_b8, #smem, mutable>
    %b = ttg.local_load %tile : !ttg.memdesc<64x128xi8, #panel_b8, #smem, mutable> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma_b8, kWidth = 8}>>
    tt.return
  }

  // Cutting the logical N dimension no longer preserves the complete panel
  // address space and must use generic local_load lowering.
  // CHECK-LABEL: llvm.func @panel_n_subslice
  // CHECK-NOT: llvm.call_intrinsic "llvm.hcu.ds.read.m32x16.b16"
  tt.func public @panel_n_subslice() {
    %tile = ttg.local_alloc : () -> !ttg.memdesc<32x128xf16, #panel, #smem, mutable>
    %n0 = ttg.memdesc_subslice %tile[0, 0] : !ttg.memdesc<32x128xf16, #panel, #smem, mutable> -> !ttg.memdesc<32x64xf16, #panel, #smem, mutable, 32x128>
    %b = ttg.local_load %n0 : !ttg.memdesc<32x64xf16, #panel, #smem, mutable, 32x128> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_n64, kWidth = 4}>>
    tt.return
  }
}
