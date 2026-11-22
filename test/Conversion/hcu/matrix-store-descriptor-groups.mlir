// RUN: triton-opt %s --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" --canonicalize --cse | FileCheck %s
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" --canonicalize --cse | FileCheck %s --check-prefix=REBASE

// Explicit interleave2 register tiles must retain their row-major store order.
// Choose the descriptor-sharing direction as matrix-load does: minimize the
// number of descriptors, prefer the matrix's major direction on ties, and
// keep padding constant within each group. Physical moffsets are 10 bits.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}>
#layout = #ttg.hcu_mmac_interleave2<parent = #mma>
#store = #amdg<MlsStoreEncoding storeTile = [16, 32], elemBitWidth = 16, interleaveKind = 1, version = 3, order = [1, 0], warpsPerCTA = [2, 2]>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func {{.*}}@current_4x4(
  // Production case: 16 stores, four row descriptors. Equal costs prefer N
  // offsets (T=R=1). Previously this path built 16 descriptors.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D0:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D0]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D0]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D0]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D1:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D1]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D1]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D1]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D2:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D2]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D2]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D2]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D3:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D3]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D3]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[CURRENT_4X4_D3]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @current_4x4(%v: tensor<128x256xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 128 : i32
    %n = arith.constant 256 : i32
    %sm = arith.constant 256 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 256], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<128x256xbf16, #layout>, <bf16>
    tt.return
  }
  // CHECK-LABEL: llvm.func {{.*}}@tall_4x2(
  // Eight stores: sharing along M needs two descriptors; sharing along N
  // would need four. R=0 advances rows without changing register order.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D0:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D1:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D0]][32, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D1]][32, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D0]][64, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D1]][64, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D0]][96, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[TALL_4X2_D1]][96, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @tall_4x2(%v: tensor<128x128xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 128 : i32
    %n = arith.constant 128 : i32
    %sm = arith.constant 128 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 128], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<128x128xbf16, #layout>, <bf16>
    tt.return
  }
  // CHECK-LABEL: llvm.func {{.*}}@wide_4x8(
  // 32 stores: sharing along N needs four descriptors; sharing along M
  // would need eight.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][256, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][320, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][384, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D0]][448, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][256, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][320, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][384, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D1]][448, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][256, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][320, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][384, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D2]][448, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][256, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][320, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][384, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[WIDE_4X8_D3]][448, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @wide_4x8(%v: tensor<128x512xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 128 : i32
    %n = arith.constant 512 : i32
    %sm = arith.constant 512 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 512], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<128x512xbf16, #layout>, <bf16>
    tt.return
  }
  // CHECK-LABEL: llvm.func {{.*}}@padding_m(
  // M padding varies by row, so share along N and keep each row filter in
  // its descriptor.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D0:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D0]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D0]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D0]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D1:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D1]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D1]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D1]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D2:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D2]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D2]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D2]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D3:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D3]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D3]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_M_D3]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @padding_m(%v: tensor<128x256xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 123 : i32
    %n = arith.constant 256 : i32
    %sm = arith.constant 256 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 256], indices = [%zero, %zero] {MlsStoreEncoding = #store, boundaryCheck = array<i32: 0>} : tensor<128x256xbf16, #layout>, <bf16>
    tt.return
  }
  // CHECK-LABEL: llvm.func {{.*}}@padding_n(
  // N padding varies by column, so share along M even though the tie
  // normally prefers N.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D0:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D1:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D2:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D3:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D0]][32, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D1]][32, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D2]][32, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D3]][32, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D0]][64, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D1]][64, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D2]][64, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D3]][64, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D0]][96, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D1]][96, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D2]][96, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_N_D3]][96, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @padding_n(%v: tensor<128x256xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 128 : i32
    %n = arith.constant 249 : i32
    %sm = arith.constant 256 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 256], indices = [%zero, %zero] {MlsStoreEncoding = #store, boundaryCheck = array<i32: 1>} : tensor<128x256xbf16, #layout>, <bf16>
    tt.return
  }
  // CHECK-LABEL: llvm.func {{.*}}@padding_both(
  // Both filters vary: retain an individual descriptor and zero immediate
  // per instruction.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D0:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D1:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D2:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D3:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D4:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D5:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D6:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D7:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D8:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D9:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D10:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D11:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D12:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D13:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D14:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[PADDING_BOTH_D15:%[a-zA-Z0-9_]+]][0, true, false, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @padding_both(%v: tensor<128x256xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 123 : i32
    %n = arith.constant 249 : i32
    %sm = arith.constant 256 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 256], indices = [%zero, %zero] {MlsStoreEncoding = #store, boundaryCheck = array<i32: 0, 1>} : tensor<128x256xbf16, #layout>, <bf16>
    tt.return
  }
  // CHECK-LABEL: llvm.func {{.*}}@column_windows(
  // REBASE-LABEL: llvm.func {{.*}}@column_windows(
  // REBASE-DAG: %[[C1024:.*]] = llvm.mlir.constant(1024 : i64)
  // REBASE-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // Verify the new descriptor comes from base + 1024 columns, not merely
  // a new capture name. Reusing the previous window must fail this check.
  // REBASE: %[[DELTA:.*]] = llvm.mul %[[C1024]], %[[ONE]] : i64
  // REBASE: %[[COL:.*]] = llvm.add {{.*}}, %[[DELTA]] : i64
  // REBASE: %[[OFF:.*]] = llvm.add {{.*}}, %[[COL]] : i64
  // REBASE: %[[PTR:.*]] = llvm.getelementptr %arg1[%[[OFF]]]
  // REBASE: %[[LO:.*]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr<1> to i32
  // REBASE: %[[ADDR:.*]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr<1> to i64
  // REBASE: %[[SHIFT:.*]] = llvm.lshr %[[ADDR]], {{.*}} : i64
  // REBASE: %[[HIGH:.*]] = llvm.trunc %[[SHIFT]] : i64 to i32
  // REBASE: %[[HI:.*]] = llvm.and %[[HIGH]], {{.*}} : i32
  // REBASE: %[[DLO:.*]] = llvm.insertelement %[[LO]], {{.*}} : vector<4xi32>
  // REBASE: %[[DHI:.*]] = llvm.insertelement %[[HI]], %[[DLO]]{{.*}} : vector<4xi32>
  // REBASE: %[[STRIDE:.*]] = llvm.insertelement {{.*}}, %[[DHI]]{{.*}} : vector<4xi32>
  // REBASE: %[[DESC:.*]] = llvm.insertelement {{.*}}, %[[STRIDE]]{{.*}} : vector<4xi32>
  // REBASE: rocdl.matrix.store.32x16.b16 {{.*}}, %[[DESC]][0, true, true, false, false]
  // Physical columns span two 1024-element windows. Rebase the descriptor
  // at 1024 and restart the immediate, rather than overflow its 10 bits.
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][256, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][320, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][384, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][448, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][512, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][576, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][640, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][704, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][768, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][832, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][896, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D0]][960, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1:%[a-zA-Z0-9_]+]][0, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][64, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][128, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][192, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][256, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][320, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][384, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][448, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][512, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][576, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][640, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][704, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][768, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][832, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][896, true, true, false, false]
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: rocdl.matrix.store.32x16.b16 %{{.*}}, [[COLUMN_WINDOWS_D1]][960, true, true, false, false]
  // CHECK-COUNT-96: rocdl.matrix.store.32x16.b16
  // CHECK-NOT: rocdl.matrix.store
  // CHECK: llvm.return
  tt.func private @column_windows(%v: tensor<128x2048xbf16, #layout>, %ptr: !tt.ptr<bf16>) {
    %m = arith.constant 128 : i32
    %n = arith.constant 2048 : i32
    %sm = arith.constant 2048 : i64
    %sn = arith.constant 1 : i64
    %zero = arith.constant 0 : i32
    amdg.matrix_store_from_reg %v, %ptr, shape = [%m, %n], strides = [%sm, %sn], tensorShape = [128, 2048], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<128x2048xbf16, #layout>, <bf16>
    tt.return
  }
}
