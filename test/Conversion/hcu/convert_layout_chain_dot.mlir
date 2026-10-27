// RUN: triton-opt %s --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx936" | FileCheck %s

// HCU LEGACY acc -> dot-A chain conversion (the GDN/FA chain-dot shape).
// Both element widths must lower to the fixed warp-shuffle closed forms in
// third_party/hcu/lib/TritonHCUToLLVM/ConvertLayoutOpToLLVM.cpp (Quad for
// 32-bit elements, HalfQuad for 16-bit ones) and never stage through shared
// memory.  Pinning the instruction sequence guards
// against silent regressions to the generic select/shuffle/select lowering,
// which was measured +2.2~4.5% on ALU-bound chains and mis-decodes the
// 16-bit halfword transposition (see
// third_party/hcu/perf/chain-dot-shuffle-analysis.md).

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: f32_chain
  // CHECK: rocdl.ds_swizzle
  // CHECK: rocdl.ds_bpermute
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  tt.func @f32_chain(%arg0: tensor<16x128xf32, #mma>) {
    %0 = ttg.convert_layout %arg0 : tensor<16x128xf32, #mma> -> tensor<16x128xf32, #dot_a>
    tt.return
  }

  // CHECK-LABEL: bf16_chain
  // CHECK: rocdl.ds_bpermute
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.perm"
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  tt.func @bf16_chain(%arg0: tensor<16x128xbf16, #mma>) {
    %0 = ttg.convert_layout %arg0 : tensor<16x128xbf16, #mma> -> tensor<16x128xbf16, #dot_a>
    tt.return
  }

  // A single 16x16 tile has too few registers to pack, so 16-bit elements
  // take the unpacked Quad form.
  // CHECK-LABEL: bf16_single_tile
  // CHECK: rocdl.ds_bpermute
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  tt.func @bf16_single_tile(%arg0: tensor<16x16xbf16, #mma>) {
    %0 = ttg.convert_layout %arg0 : tensor<16x16xbf16, #mma> -> tensor<16x16xbf16, #dot_a>
    tt.return
  }
}
