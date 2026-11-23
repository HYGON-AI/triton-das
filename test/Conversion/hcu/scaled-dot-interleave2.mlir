// RUN: triton-opt %s --split-input-file --tritonhcu-allocate-scale-buffer --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse --verify-diagnostics | FileCheck %s --check-prefixes=LOWER,ORDER,FP4
// Native B and result encodings must lower directly, without a layout conversion.
// LOWER-LABEL: llvm.func @native_interleave2
// LOWER-DAG: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// LOWER-DAG: %[[FIFTEEN:.*]] = llvm.mlir.constant(15 : i32)
// LOWER-DAG: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// LOWER-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
// LOWER-DAG: %[[MASK:.*]] = llvm.mlir.constant(96 : i32)
// LOWER-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// LOWER: rocdl.ds.scale.copy.ds2buf
// B source mapping within N32 is [even columns, odd columns].
// LOWER: %[[LO:.*]] = llvm.and %[[MN:.*]], %[[FIFTEEN]]
// LOWER-NEXT: %[[EVEN:.*]] = llvm.shl %[[LO]], %[[ONE]]
// LOWER-NEXT: %[[HI:.*]] = llvm.and %[[MN]], %[[MASK]]
// LOWER-NEXT: %[[OUTER:.*]] = llvm.lshr %[[HI]], %[[ZERO]]
// LOWER-NEXT: %[[BIT:.*]] = llvm.and %[[MN]], %[[SIXTEEN]]
// LOWER-NEXT: %[[IS_EVEN:.*]] = llvm.icmp "eq" %[[BIT]], %[[ZERO]]
// LOWER-NEXT: %[[ODD:.*]] = llvm.select %[[IS_EVEN]], %[[ZERO]], %[[ONE]]
// LOWER-NEXT: %[[INNER:.*]] = llvm.or disjoint %[[EVEN]], %[[OUTER]]
// LOWER-NEXT: %[[PERMUTED:.*]] = llvm.or disjoint %[[INNER]], %[[ODD]]
// LOWER-NEXT: %[[WITH_ZERO:.*]] = llvm.or disjoint %[[PERMUTED]], %[[ZERO]]
// LOWER-NEXT: %[[MAPPED:.*]] = llvm.xor %[[ZERO]], %[[WITH_ZERO]]
// LOWER: %[[PITCH:.*]] = llvm.mul %[[MAPPED]], %[[FOUR]]
// LOWER-NEXT: %[[OFFSET:.*]] = llvm.add %[[PITCH]], {{.*}}
// LOWER-NEXT: %[[ADDR:.*]] = llvm.getelementptr {{.*}}[%[[OFFSET]]]
// LOWER: rocdl.ds.scale.copy.ds2buf %[[ADDR]],
// LOWER: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"
// LOWER-SAME: "amdg.m0"
// LOWER-NOT: amdg.dot_scaled
// LOWER-NOT: ttg.convert_layout
// LOWER: llvm.return
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @native_interleave2(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @unpaired_b(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{interleave2 result requires interleave2 B}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @unpaired_result(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{interleave2 B requires an interleave2 result}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

// Textual block order differs from dominance order. Lowering must derive
// scale geometry from original SSA independently of textual block order.
// ORDER-LABEL: llvm.func @native_interleave2_reordered_blocks
// ORDER-DAG: rocdl.ds.scale.copy.ds2buf
// ORDER-DAG: rocdl.ds.scale.copy.ds2buf
// ORDER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"
// ORDER-NOT: amdg.dot_scaled
// ORDER-NOT: builtin.unrealized_conversion_cast
// ORDER: }
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @native_interleave2_reordered_blocks(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    cf.br ^produce
  ^consume:
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  ^produce:
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    cf.br ^consume
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @wrong_operand_index(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{Wrong opIdx}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#other = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [2, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #other, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @wrong_operand_parent(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{Incompatible parent encoding}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @missing_result_layout(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{requires an MFMA result layout}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @missing_a_layout(%a: tensor<64x128xf8E4M3FN>, %b: tensor<128x128xf8E4M3FN, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{requires an A DotOperand layout}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @missing_b_layout(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x128xf8E4M3FN>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    // expected-error @+1 {{requires a B operand layout}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<128x128xf8E4M3FN>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

// Packed K dimensions must not change the N permutation derived from B layouts.
// FP4-LABEL: llvm.func @interleave2_packed_fp4
// FP4-DAG: %[[F4_FIFTEEN:.*]] = llvm.mlir.constant(15 : i32)
// FP4-DAG: %[[F4_SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// FP4-DAG: %[[F4_ONE:.*]] = llvm.mlir.constant(1 : i32)
// FP4-DAG: %[[F4_ZERO:.*]] = llvm.mlir.constant(0 : i32)
// FP4: rocdl.ds.scale.copy.ds2buf
// B scales must still enumerate even N columns, then odd N columns.
// FP4: %[[F4_LO:.*]] = llvm.and %[[F4_MN:.*]], %[[F4_FIFTEEN]]
// FP4: %[[F4_EVEN:.*]] = llvm.shl %[[F4_LO]], %[[F4_ONE]]
// FP4: %[[F4_BIT:.*]] = llvm.and %[[F4_MN]], %[[F4_SIXTEEN]]
// FP4: %[[F4_IS_EVEN:.*]] = llvm.icmp "eq" %[[F4_BIT]], %[[F4_ZERO]]
// FP4: llvm.select %[[F4_IS_EVEN]], %[[F4_ZERO]], %[[F4_ONE]]
// FP4: rocdl.ds.scale.copy.ds2buf
// FP4-COUNT-16: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x64.fp4"
// FP4-NOT: llvm.call_intrinsic "llvm.hcu.mmac.scale
// FP4: llvm.return
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @interleave2_packed_fp4(%a: tensor<64x64xi8, #a>, %b: tensor<64x128xi8, #b>, %c: tensor<64x128xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x128x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> * tensor<64x128xi8, #b>, !ttg.memdesc<128x4xi8, #sb, #amdg.scale_memory> -> tensor<64x128xf32, #result>
    tt.store %out, %d : tensor<64x128x!tt.ptr<f32>, #result>
    tt.return
  }
}
