// RUN: triton-opt %s --split-input-file --tritonhcu-allocate-scale-buffer | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s --split-input-file --tritonhcu-allocate-scale-buffer --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse | FileCheck %s --check-prefix=WINDOW --implicit-check-not='"amdg.m0"'
// Two compute waves per operand each own complete R groups.
// ALLOC: amdg.scale_write {{.*}}row_base = 0 : i32
// ALLOC: amdg.scale_write {{.*}}row_base = 64 : i32
// ALLOC: amdg.dot_scaled
// With per-warp scale partitions, all 128 MMACs form two contiguous m0 groups of 64.
// An unexpected m0 group between checks is rejected by implicit-check-not.
// WINDOW-LABEL: llvm.func @interleave2_m0_windows
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[W0:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-63: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[W0]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[W1:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-63: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[W1]] : i32)]
// WINDOW: llvm.return
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
#result = #ttg.hcu_mmac_interleave2<parent = #mma>
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @interleave2_m0_windows(%a: tensor<128x128xf8E4M3FN, #a>, %b: tensor<128x256xf8E4M3FN, #b>, %c: tensor<128x256xf32, #result>, %srcA: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<256x4xi8, #shared, #ttg.shared_memory>, %out: tensor<128x256x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<256x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<256x4xi8, #sb, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #a>, !ttg.memdesc<128x4xi8, #sa, #amdg.scale_memory> * tensor<128x256xf8E4M3FN, #b>, !ttg.memdesc<256x4xi8, #sb, #amdg.scale_memory> -> tensor<128x256xf32, #result>
    tt.store %out, %d : tensor<128x256x!tt.ptr<f32>, #result>
    tt.return
  }
}

// -----

// Ordinary scaled dot also revisits scale windows in the original M/N/K order.
// Grouping by both scale windows gives eight contiguous groups of 16 MMACs.
// WINDOW-LABEL: llvm.func @plain_m0_windows
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P0:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P0]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P1:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P1]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P2:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P2]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P3:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P3]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P4:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P4]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P5:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P5]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P6:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P6]] : i32)]
// WINDOW: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P7:[a-zA-Z0-9_]+]] : i32)]
// WINDOW-COUNT-15: llvm.call_intrinsic "llvm.hcu.mmac.scale.f32.16x16x32.f8f6f4"{{.*}}["amdg.m0"(%[[P7]] : i32)]
// WINDOW: llvm.return
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#result = #mma
#sa = #amdg.scale<{blockShape = [32, 8], rowStride = 16}>
#sb = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "hip:gfx946", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @plain_m0_windows(%a: tensor<128x128xf8E4M3FN, #a>, %b: tensor<128x256xf8E4M3FN, #b>, %c: tensor<128x256xf32, #result>, %srcA: !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory>, %srcB: !ttg.memdesc<256x4xi8, #shared, #ttg.shared_memory>, %out: tensor<128x256x!tt.ptr<f32>, #result>) {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<128x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<128x4xi8, #sa, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<256x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<256x4xi8, #sb, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #a>, !ttg.memdesc<128x4xi8, #sa, #amdg.scale_memory> * tensor<128x256xf8E4M3FN, #b>, !ttg.memdesc<256x4xi8, #sb, #amdg.scale_memory> -> tensor<128x256xf32, #result>
    tt.store %out, %d : tensor<128x256x!tt.ptr<f32>, #result>
    tt.return
  }
}
