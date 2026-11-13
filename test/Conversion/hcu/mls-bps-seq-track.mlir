// RUN: env -u TRITON_HCU_DISABLE_MLS_BPS triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s
// RUN: env TRITON_HCU_DISABLE_MLS_BPS=0 triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s
// RUN: env TRITON_HCU_DISABLE_MLS_BPS=false triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s
// RUN: env TRITON_HCU_DISABLE_MLS_BPS=1 triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s --check-prefix=OFF
// RUN: env TRITON_HCU_DISABLE_MLS_BPS=true triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s --check-prefix=OFF
// RUN: env TRITON_HCU_DISABLE_MLS_BPS=on triton-opt %s --tritonhcu-mls-lowering --allocate-shared-memory --convert-triton-hcu-to-llvm='arch=gfx946' | FileCheck %s --check-prefix=OFF

// TRITON_HCU_DISABLE_MLS_BPS defaults false; 0/false still require seq
// tracking, while 1/true/on disable BPS even inside a seq track.
// BPS follows active seq tracking, not the kernel's WASP/WDRA mode.
// CFG merges require tracking on every reachable path, including backedges.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func internal @mls_bps_no_seq(
  // OFF-LABEL: llvm.func internal @mls_bps_no_seq(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_no_seq(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_tracked(
  // OFF-LABEL: llvm.func internal @mls_bps_tracked(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, true]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_tracked(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    rocdl.hcu.s.abarrier.seq %c1
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_closed(
  // OFF-LABEL: llvm.func internal @mls_bps_closed(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_closed(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    rocdl.hcu.s.abarrier.seq %c1
    %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_future_seq(
  // OFF-LABEL: llvm.func internal @mls_bps_future_seq(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_future_seq(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    rocdl.hcu.s.abarrier.seq %c1
    %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_one_path(
  // OFF-LABEL: llvm.func internal @mls_bps_one_path(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_one_path(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    cf.cond_br %cond, ^left, ^right
  ^left:
    rocdl.hcu.s.abarrier.seq %c1
    cf.br ^join
  ^right:
    cf.br ^join
  ^join:
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_both_paths(
  // OFF-LABEL: llvm.func internal @mls_bps_both_paths(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, true]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_both_paths(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    cf.cond_br %cond, ^left, ^right
  ^left:
    rocdl.hcu.s.abarrier.seq %c1
    cf.br ^join
  ^right:
    rocdl.hcu.s.abarrier.seq %c1
    cf.br ^join
  ^join:
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_loop_track(
  // OFF-LABEL: llvm.func internal @mls_bps_loop_track(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, true]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_loop_track(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    rocdl.hcu.s.abarrier.seq %c1
    cf.br ^loop
  ^loop:
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    cf.cond_br %cond, ^loop, ^exit
  ^exit:
    %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

  // CHECK-LABEL: llvm.func internal @mls_bps_loop_closed_backedge(
  // OFF-LABEL: llvm.func internal @mls_bps_loop_closed_backedge(
  // OFF-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // OFF-NOT: rocdl.matrix.load
  // CHECK-COUNT-4: rocdl.matrix.load.64x16.b8 {{.*}}, false]
  // CHECK-NOT: rocdl.matrix.load
  // CHECK: llvm.return
  tt.func private @mls_bps_loop_closed_backedge(%p: !tt.ptr<f8E4M3FN>, %cond: i1) -> tensor<128x128xf8E4M3FN, #b> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %size = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    rocdl.hcu.s.abarrier.seq %c1
    cf.br ^loop
  ^loop:
    %tile = tt.matrix_load %p, shape = [%size, %size], strides = [%s1, %s128],
        tensorShape = [128, 128], indices = [%c0, %c0]
        {MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [64, 16], elemBitWidth = 8, elemBitTyKind = 0, alt2Kind = 0, version = 3, order = [0, 1], warpsPerCTA = [1, 4]>}
        : <f8E4M3FN> -> tensor<128x128xf8E4M3FN, #b>
    %ready = rocdl.hcu.s.abarrier.arrive i32, %c1, 1
    cf.cond_br %cond, ^loop, ^exit
  ^exit:
    tt.return %tile : tensor<128x128xf8E4M3FN, #b>
  }

}
