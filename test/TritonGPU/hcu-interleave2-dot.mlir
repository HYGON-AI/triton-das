// RUN: triton-opt %s -split-input-file --verify-diagnostics

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [2, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#c = #ttg.hcu_mmac_interleave2<parent = #mma>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @valid(%a: tensor<64x32xf8E4M3FN, #a>, %b: tensor<32x32xf8E4M3FN, #b>, %c: tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c> {
  %d = "tt.dot"(%a, %b, %c) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<64x32xf8E4M3FN, #a>, tensor<32x32xf8E4M3FN, #b>, tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c>
  tt.return %d : tensor<64x32xf32, #c>
 }
}

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [2, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#c = #ttg.hcu_mmac_interleave2<parent = #mma>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @missing_b(%a: tensor<64x32xf8E4M3FN, #a>, %b: tensor<32x32xf8E4M3FN, #b>, %c: tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c> {
  // expected-error@+1 {{interleave2 result requires interleave2 B}}
  %d = "tt.dot"(%a, %b, %c) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<64x32xf8E4M3FN, #a>, tensor<32x32xf8E4M3FN, #b>, tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c>
  tt.return %d : tensor<64x32xf32, #c>
 }
}

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [2, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#c = #mma
module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @missing_result(%a: tensor<64x32xf8E4M3FN, #a>, %b: tensor<32x32xf8E4M3FN, #b>, %c: tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c> {
  // expected-error@+1 {{interleave2 B requires an interleave2 result}}
  %d = "tt.dot"(%a, %b, %c) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<64x32xf8E4M3FN, #a>, tensor<32x32xf8E4M3FN, #b>, tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c>
  tt.return %d : tensor<64x32xf32, #c>
 }
}

// -----
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [2, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
#c = #ttg.hcu_mmac_interleave2<parent = #mma>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @kwidth(%a: tensor<64x32xf8E4M3FN, #a>, %b: tensor<32x32xf8E4M3FN, #b>, %c: tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c> {
  // expected-error@+1 {{mismatching kWidth between A and B operands}}
  %d = "tt.dot"(%a, %b, %c) <{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}> : (tensor<64x32xf8E4M3FN, #a>, tensor<32x32xf8E4M3FN, #b>, tensor<64x32xf32, #c>) -> tensor<64x32xf32, #c>
  tt.return %d : tensor<64x32xf32, #c>
 }
}
