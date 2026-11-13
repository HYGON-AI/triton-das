// RUN: triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse | FileCheck %s

// One consumer owns M128/N128/BK128. Two native 64x16 global copies
// form a K64/N32 interleave2 panel. TN2 with WM2/WN2 keeps B reads at
// eight per wave/BK, instead of sixteen for the K128/N64 TN4 panel.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func internal @mls_small_interleave2
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE:%[a-zA-Z0-9_]+]][0, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][32, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][2048, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][2080, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][8192, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][8224, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][10240, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][10272, 1, 2, 2, 1]
  // CHECK-NOT: rocdl.ds.read.matrix.trans.format
  // CHECK: llvm.return
  tt.func private @mls_small_interleave2(%B: !tt.ptr<f8E4M3FN>) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %b = tt.matrix_load %B, shape = [%c128, %c128], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 32], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 1, version = 3, order = [0, 1], warpsPerCTA = [2, 2]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    tt.return %b : tensor<128x128xf8E4M3FN, #b>
  }
}

// Native K128/N32 interleave2 panel: two K64/N32 global loads, four DS
// reads per panel. Loader waves cover [1,4]; MFMA keeps WM2/WN2 and TN2.
#native_mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#native_b = #ttg.hcu_mmac_interleave2<parent = #ttg.dot_op<{opIdx = 1, parent = #native_mma, kWidth = 8}>>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func internal @mls_native_interleave2
  // CHECK-COUNT-2: rocdl.matrix.load.64x32.b8
  // CHECK-NOT: rocdl.matrix.load.64x16.b8
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE:%[a-zA-Z0-9_]+]][0, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][32, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][2048, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][2080, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][8192, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][8224, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][10240, 1, 2, 2, 1]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][10272, 1, 2, 2, 1]
  // CHECK-NOT: rocdl.ds.read.matrix.trans.format
  // CHECK: llvm.return
  tt.func private @mls_native_interleave2(%B: !tt.ptr<f8E4M3FN>) -> tensor<128x128xf8E4M3FN, #native_b> {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %b = tt.matrix_load %B, shape = [%c128, %c128], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [128, 32], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 1, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #native_b>
    tt.return %b : tensor<128x128xf8E4M3FN, #native_b>
  }
}
