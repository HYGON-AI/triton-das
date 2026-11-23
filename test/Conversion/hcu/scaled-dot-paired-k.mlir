// RUN: triton-opt %s --split-input-file --verify-diagnostics > /dev/null
// RUN: triton-opt %s --split-input-file --tritonhcu-allocate-scale-buffer --verify-diagnostics | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s --split-input-file --tritonhcu-allocate-scale-buffer --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse --verify-diagnostics | FileCheck %s --check-prefix=LOWER
// Views are created early, but A0 stays live until the last consuming dot.
// ALLOC: hcu.scale_buffer = 6 : i32
// ALLOC-LABEL: tt.func public @paired_k_lifetime
// ALLOC: %[[A:.*]] = amdg.scale_write {{.*}}row_base = 0 : i32
// ALLOC: %[[B:.*]] = amdg.scale_write {{.*}}row_base = 32 : i32
// ALLOC: ttg.memdesc_subslice %[[A]][0, 0]
// ALLOC: ttg.memdesc_subslice %[[B]][0, 0]
// ALLOC: ttg.memdesc_subslice %[[A]][0, 2]
// ALLOC: ttg.memdesc_subslice %[[B]][0, 2]
// ALLOC: amdg.dot_scaled
// ALLOC: amdg.scale_write {{.*}}row_base = 64 : i32
// ALLOC: amdg.dot_scaled
// ALLOC: amdg.dot_scaled
// LOWER-LABEL: llvm.func @paired_k_lifetime
// LOWER-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// LOWER-DAG: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// LOWER-DAG: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// LOWER-DAG: %[[SIX:.*]] = llvm.mlir.constant(6 : i32)
// LOWER-COUNT-2: rocdl.ds.scale.copy.ds2buf
// K views must retain the publication dependency on their root writes.
// LOWER: rocdl.s.waitcnt
// LOWER-NEXT: rocdl.s.barrier
// First dot: K32 column 0, with B on the opposite bank (row 4).
// LOWER: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[ZERO]], %[[FOUR]])
// LOWER-COUNT-7: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"
// LOWER: rocdl.ds.scale.copy.ds2buf
// LOWER: rocdl.s.waitcnt
// LOWER-NEXT: rocdl.s.barrier
// LOWER-COUNT-8: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"
// Last dot reuses A0/B at K32 column 2; it must not copy scales again.
// LOWER-NOT: rocdl.ds.scale.copy.ds2buf
// LOWER: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[TWO]], %[[SIX]])
// LOWER-COUNT-7: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"
// LOWER-NOT: rocdl.ds.scale.copy.ds2buf
// LOWER-NOT: llvm.call_intrinsic "llvm.hcu.mmac.scale
// LOWER: llvm.return
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @paired_k_lifetime(%a: tensor<64x64xf8E4M3FN, #a>, %b: tensor<64x64xf8E4M3FN, #b>, %c: tensor<64x64xf32, #result>, %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>, %out: tensor<64x64x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sb, #amdg.scale_memory>
    %identity = ttg.memdesc_subslice %sa[0, 0] : !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %a0 = ttg.memdesc_subslice %identity[0, 0] : !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x4>
    %b0 = ttg.memdesc_subslice %sb[0, 0] : !ttg.memdesc<64x4xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x4>
    %a1 = ttg.memdesc_subslice %sa[0, 2] : !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x4>
    %b1 = ttg.memdesc_subslice %sb[0, 2] : !ttg.memdesc<64x4xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x4>
    %d0 = amdg.dot_scaled %a scale %a0, %b scale %b0, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #a>, !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x4> * tensor<64x64xf8E4M3FN, #b>, !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x4> -> tensor<64x64xf32, #result>
    %other = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory>
    %other0 = ttg.memdesc_subslice %other[0, 0] : !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x4>
    %b0_again = ttg.memdesc_subslice %sb[0, 0] : !ttg.memdesc<64x4xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x4>
    %d1 = amdg.dot_scaled %a scale %other0, %b scale %b0_again, %d0 lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #a>, !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x4> * tensor<64x64xf8E4M3FN, #b>, !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x4> -> tensor<64x64xf32, #result>
    %d = amdg.dot_scaled %a scale %a1, %b scale %b1, %d1 lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #a>, !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x4> * tensor<64x64xf8E4M3FN, #b>, !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x4> -> tensor<64x64xf32, #result>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#scale = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#other = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module {
  tt.func @negative(%src: !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory>) {
    // expected-error @+1 {{Scale Buffer K window is out of bounds}}
    %view = ttg.memdesc_subslice %src[0, -1] : !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #scale, #amdg.scale_memory, 64x4>
    tt.return
  }
}

// -----

#scale = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#other = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module {
  tt.func @past_end(%src: !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory>) {
    // expected-error @+1 {{Scale Buffer K window is out of bounds}}
    %view = ttg.memdesc_subslice %src[0, 4] : !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #scale, #amdg.scale_memory, 64x4>
    tt.return
  }
}

// -----

#scale = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#other = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module {
  tt.func @changed_layout(%src: !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory>) {
    // expected-error @+1 {{Scale Buffer subslice must preserve storage and layout}}
    %view = ttg.memdesc_subslice %src[0, 0] : !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #other, #amdg.scale_memory, 64x4>
    tt.return
  }
}

// -----

#scale = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#other = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module {
  tt.func @changed_allocation(%src: !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory>) {
    // expected-error @+1 {{Scale Buffer subslice must preserve storage and layout}}
    %view = ttg.memdesc_subslice %src[0, 0] : !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #scale, #amdg.scale_memory, 64x8>
    tt.return
  }
}

// -----

// A nested K view retains the full allocation's pitch across R groups.
// ALLOC-LABEL: tt.func public @nested_allocation_pitch
// ALLOC: amdg.scale_write {{.*}}row_base = 0 : i32
// ALLOC: amdg.scale_write {{.*}}row_base = 64 : i32
// LOWER-LABEL: llvm.func @nested_allocation_pitch
// LOWER-DAG: %[[EIGHTEEN:.*]] = llvm.mlir.constant(18 : i32)
// LOWER-DAG: %[[SIX:.*]] = llvm.mlir.constant(6 : i32)
// LOWER-DAG: %[[NINETEEN:.*]] = llvm.mlir.constant(19 : i32)
// LOWER-DAG: %[[SEVEN:.*]] = llvm.mlir.constant(7 : i32)
// LOWER-DAG: %[[FOURTEEN:.*]] = llvm.mlir.constant(14 : i32)
// LOWER-DAG: %[[FIFTEEN:.*]] = llvm.mlir.constant(15 : i32)
// LOWER-COUNT-2: rocdl.ds.scale.copy.ds2buf
// Address pairs are independent of the m0-window emission order.
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[EIGHTEEN]], %[[SIX]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[NINETEEN]], %[[SEVEN]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[EIGHTEEN]], %[[FOURTEEN]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[NINETEEN]], %[[FIFTEEN]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[EIGHTEEN]], %[[SIX]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[NINETEEN]], %[[SEVEN]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[EIGHTEEN]], %[[FOURTEEN]])
// LOWER-DAG: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[NINETEEN]], %[[FIFTEEN]])
// LOWER-NOT: rocdl.ds.scale.copy.ds2buf
// LOWER: llvm.return
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @nested_allocation_pitch(%a: tensor<64x64xf8E4M3FN, #a>, %b: tensor<64x64xf8E4M3FN, #b>, %c: tensor<64x64xf32, #result>, %srcA: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %out: tensor<64x64x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory>
    %half = ttg.memdesc_subslice %sa[0, 8] : !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16>
    %aw = ttg.memdesc_subslice %half[0, 2] : !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16> -> !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x16>
    %bw = ttg.memdesc_subslice %sb[0, 2] : !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x16>
    %d = amdg.dot_scaled %a scale %aw, %b scale %bw, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #a>, !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x16> * tensor<64x64xf8E4M3FN, #b>, !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x16> -> tensor<64x64xf32, #result>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @bank_boundary(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x64xf8E4M3FN, #b>, %c: tensor<64x64xf32, #result>, %srcA: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %out: tensor<64x64x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory>
    %half = ttg.memdesc_subslice %sa[0, 0] : !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16>
    %aw = ttg.memdesc_subslice %half[0, 0] : !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory, 64x16>
    %bw = ttg.memdesc_subslice %sb[0, 2] : !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x4xi8, #sb, #amdg.scale_memory, 64x16>
    // expected-error @+1 {{requires opposite A/B Scale Buffer banks throughout the K windows}}
    %d = amdg.dot_scaled %a scale %aw, %b scale %bw, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory, 64x16> * tensor<128x64xf8E4M3FN, #b>, !ttg.memdesc<64x4xi8, #sb, #amdg.scale_memory, 64x16> -> tensor<64x64xf32, #result>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @unaligned_instruction_window(%a: tensor<64x64xf8E4M3FN, #a>, %b: tensor<64x64xf8E4M3FN, #b>, %c: tensor<64x64xf32, #result>, %srcA: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %out: tensor<64x64x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory>
    %half = ttg.memdesc_subslice %sa[0, 8] : !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16>
    %aw = ttg.memdesc_subslice %half[0, 3] : !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16> -> !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x16>
    %bw = ttg.memdesc_subslice %sb[0, 2] : !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x16>
    // expected-error @+1 {{requires an instruction-aligned scale K window}}
    %d = amdg.dot_scaled %a scale %aw, %b scale %bw, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #a>, !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x16> * tensor<64x64xf8E4M3FN, #b>, !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x16> -> tensor<64x64xf32, #result>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

#scale = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
module {
  tt.func @changed_non_k_dimension(%src: !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory>) {
    // expected-error @+1 {{Scale Buffer subslice only supports K-dimension windows}}
    %view = ttg.memdesc_subslice %src[16, 0] : !ttg.memdesc<64x4xi8, #scale, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #scale, #amdg.scale_memory, 64x4>
    tt.return
  }
}

// -----

// Invalid offset rank must produce a diagnostic even with reordered blocks.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @malformed_view_before_definition(%a: tensor<64x64xf8E4M3FN, #a>, %b: tensor<64x64xf8E4M3FN, #b>, %c: tensor<64x64xf32, #result>, %srcA: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %out: tensor<64x64x!tt.ptr<f32>, #result>) {
    cf.br ^produce
  ^consume:
    // expected-error @+1 {{requires scales rooted in amdg.scale_write through valid static K views}}
    %d = amdg.dot_scaled %a scale %aw, %b scale %bw, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x64xf8E4M3FN, #a>, !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x16> * tensor<64x64xf8E4M3FN, #b>, !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x16> -> tensor<64x64xf32, #result>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #result>
    tt.return
  ^produce:
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory>
    %half = ttg.memdesc_subslice %sa[0] : !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16>
    %aw = ttg.memdesc_subslice %half[0, 2] : !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16> -> !ttg.memdesc<64x2xi8, #sa, #amdg.scale_memory, 64x16>
    %bw = ttg.memdesc_subslice %sb[0, 2] : !ttg.memdesc<64x16xi8, #sb, #amdg.scale_memory> -> !ttg.memdesc<64x2xi8, #sb, #amdg.scale_memory, 64x16>
    cf.br ^consume
  }
}

// -----

// Matching bank_xor is valid when different K origins select opposite banks.
// ALLOC-LABEL: tt.func public @opposite_banks_from_k_origins
// ALLOC: amdg.scale_write {{.*}}row_base = 0 : i32
// ALLOC: amdg.scale_write {{.*}}row_base = 64 : i32
// LOWER-LABEL: llvm.func @opposite_banks_from_k_origins
// LOWER-DAG: %[[BANK_ZERO:.*]] = llvm.mlir.constant(0 : i32)
// LOWER-DAG: %[[BANK_FOUR:.*]] = llvm.mlir.constant(4 : i32)
// LOWER-COUNT-2: rocdl.ds.scale.copy.ds2buf
// LOWER: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"({{.*}}, %[[BANK_ZERO]], %[[BANK_FOUR]])
// LOWER-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"
// LOWER-NOT: llvm.call_intrinsic "llvm.hcu.mmac.scale
// LOWER: llvm.return

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @opposite_banks_from_k_origins(%a: tensor<64x128xf8E4M3FN, #a>, %b: tensor<128x64xf8E4M3FN, #b>, %c: tensor<64x64xf32, #result>, %srcA: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, %out: tensor<64x64x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory>
    %half = ttg.memdesc_subslice %sa[0, 0] : !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16>
    %aw = ttg.memdesc_subslice %half[0, 0] : !ttg.memdesc<64x8xi8, #sa, #amdg.scale_memory, 64x16> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory, 64x16>
    %bw = ttg.memdesc_subslice %sb[0, 4] : !ttg.memdesc<64x16xi8, #sa, #amdg.scale_memory> -> !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory, 64x16>
    %d = amdg.dot_scaled %a scale %aw, %b scale %bw, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #a>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory, 64x16> * tensor<128x64xf8E4M3FN, #b>, !ttg.memdesc<64x4xi8, #sa, #amdg.scale_memory, 64x16> -> tensor<64x64xf32, #result>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_dot(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %sa: !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    // expected-error @+1 {{requires scales rooted in amdg.scale_write}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}


// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_dot(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    // expected-error @+1 {{requires HCU scaled dot consumers through static K views}}
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    ttg.local_dealloc %sa : !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    tt.return %c : tensor<64x64xf32, #mma>
  }
}


// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @same_bank(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    // expected-error @+1 {{requires opposite A/B Scale Buffer banks throughout the K windows}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @mixed_scale_operand_sides(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    // expected-error @+1 {{scale consumers must agree on operand side and interleave mapping}}
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    %e = amdg.dot_scaled %a scale %sb, %b scale %sa, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}
