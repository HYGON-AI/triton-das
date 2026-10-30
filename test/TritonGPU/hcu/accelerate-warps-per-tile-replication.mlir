// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx946" | FileCheck %s
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx938" | FileCheck %s
// RUN: triton-opt %s --tritonhcu-accelerate-matmul="arch-generation-name=gfx936" | FileCheck %s

// With more warps than 16x16 output tiles, some warps must compute replicated
// tiles. warpsPerTile used to swap its split whenever dim1 was over-assigned,
// which moves the warps onto the narrow N axis and quadruples the redundant
// MMACs (GDN chunk_h: BT=64 x BV=16/32, num_warps=8). Keep the split that
// replicates less.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // 64x16: [8,1] replicates 2x without N aliasing; [2,4] replicates 4x.
  // CHECK: #mma = #ttg.amd_mfma<{{.*}}warpsPerCTA = [8, 1]
  // CHECK-LABEL: tt.func public @narrow_n_8_warps
  // CHECK: tt.dot {{.*}} -> tensor<64x16xf32, #mma>
  tt.func public @narrow_n_8_warps(%a: tensor<64x64xbf16, #blocked>, %b: tensor<64x16xbf16, #blocked>) -> tensor<64x16xf32, #blocked> {
    %c = arith.constant dense<0.0> : tensor<64x16xf32, #blocked>
    %av = ttg.convert_layout %a : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>>
    %bv = ttg.convert_layout %b : tensor<64x16xbf16, #blocked> -> tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>>
    %d = tt.dot %av, %bv, %c : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>> * tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>> -> tensor<64x16xf32, #blocked>
    tt.return %d : tensor<64x16xf32, #blocked>
  }
}
