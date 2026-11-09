// Modified by Hygon Information Technology Co., Ltd., 2026.
// RUN: triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' --canonicalize --cse | FileCheck %s

// B is one consumer's [K,N]=[128,128] tile after splitting N.
// Before: each read used gep(base, waveOffset + tileOffset), immediate=0.
//   %next = llvm.getelementptr %base[%c1024] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
//   %read = rocdl.ds.read.matrix.trans.format %next[0, 1, 3, 1, 0]
// After:
//   %read = rocdl.ds.read.matrix.trans.format %base[1024, 1, 3, 1, 0]
// All eight reads share gep(base, waveOffset), with static byte immediates
// 0/1024/4096/5120/8192/9216/12288/13312. No per-read scalar add is needed.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func internal @mls_read_static_offsets
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE:%[a-zA-Z0-9_]+]][0, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][1024, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][4096, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][5120, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][8192, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][9216, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][12288, 1, 3, 1, 0]
  // CHECK: rocdl.ds.read.matrix.trans.format [[BASE]][13312, 1, 3, 1, 0]
  // CHECK-NOT: rocdl.ds.read.matrix.trans.format
  // CHECK: llvm.return
  tt.func private @mls_read_static_offsets(%B: !tt.ptr<f8E4M3FN>) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %b = tt.matrix_load %B, shape = [%c128, %c128], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [2, 2]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    tt.return %b : tensor<128x128xf8E4M3FN, #b>
  }
}
