// RUN: triton-opt %s --allocate-shared-memory --convert-triton-hcu-to-llvm="arch=gfx946" --canonicalize --cse | FileCheck %s

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [2, 2]}>
#layout = #ttg.hcu_mmac_interleave2<parent = #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#store = #amdg<MlsStoreEncoding storeTile = [32, 32], elemBitWidth = 16, interleaveKind = 1, version = 3, order = [1, 0], warpsPerCTA = [1, 1]>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func {{.*}}@direct_f16
  // CHECK: %[[direct_f16A:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[direct_f16B:.*]] = llvm.extractvalue %arg0[4]
  // CHECK: rocdl.hcu.cvt.pk.f16.f32{{.*}}%[[direct_f16A]]{{.*}}%[[direct_f16B]]
  tt.func private @direct_f16(%arg0: tensor<32x32xf32, #layout>, %ptr: !tt.ptr<f16>) -> tensor<32x32xf16, #layout> {
    %v = arith.truncf %arg0 : tensor<32x32xf32, #layout> to tensor<32x32xf16, #layout>
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %s32 = arith.constant 32 : i64
    %s1 = arith.constant 1 : i64
    amdg.matrix_store_from_reg %v, %ptr, shape = [%c32, %c32], strides = [%s32, %s1], tensorShape = [32, 32], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<32x32xf16, #layout>, <f16>
    tt.return %v : tensor<32x32xf16, #layout>
  }
  // CHECK-LABEL: llvm.func {{.*}}@deep_bf16
  // CHECK: %[[deep_bf16A:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[deep_bf16B:.*]] = llvm.extractvalue %arg0[4]
  // CHECK: rocdl.hcu.cvt.pk.bf16.f32{{.*}}%[[deep_bf16A]]{{.*}}%[[deep_bf16B]]
  tt.func private @deep_bf16(%arg0: tensor<32x32xf32, #layout>, %ptr: !tt.ptr<bf16>) -> tensor<32x32xbf16, #layout> {
    %v = arith.truncf %arg0 : tensor<32x32xf32, #layout> to tensor<32x32xbf16, #layout>
    %c0 = ttg.convert_layout %v : tensor<32x32xbf16, #layout> -> tensor<32x32xbf16, #layout>
    %c1 = ttg.convert_layout %c0 : tensor<32x32xbf16, #layout> -> tensor<32x32xbf16, #layout>
    %c2 = ttg.convert_layout %c1 : tensor<32x32xbf16, #layout> -> tensor<32x32xbf16, #layout>
    %c3 = ttg.convert_layout %c2 : tensor<32x32xbf16, #layout> -> tensor<32x32xbf16, #layout>
    %c4 = ttg.convert_layout %c3 : tensor<32x32xbf16, #layout> -> tensor<32x32xbf16, #layout>
    %c5 = ttg.convert_layout %c4 : tensor<32x32xbf16, #layout> -> tensor<32x32xbf16, #layout>
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %s32 = arith.constant 32 : i64
    %s1 = arith.constant 1 : i64
    amdg.matrix_store_from_reg %c5, %ptr, shape = [%c32, %c32], strides = [%s32, %s1], tensorShape = [32, 32], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<32x32xbf16, #layout>, <bf16>
    tt.return %v : tensor<32x32xbf16, #layout>
  }
  // CHECK-LABEL: llvm.func {{.*}}@fallback_f16
  // CHECK: %[[fallback_f16A:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[fallback_f16B:.*]] = llvm.extractvalue %arg0[1]
  // CHECK: rocdl.hcu.cvt.pk.f16.f32{{.*}}%[[fallback_f16A]]{{.*}}%[[fallback_f16B]]
  tt.func private @fallback_f16(%arg0: tensor<32x32xf32, #layout>, %ptr: !tt.ptr<f16>) -> tensor<32x32xf16, #layout> {
    %v = arith.truncf %arg0 : tensor<32x32xf32, #layout> to tensor<32x32xf16, #layout>
    tt.return %v : tensor<32x32xf16, #layout>
  }
  // CHECK-LABEL: llvm.func {{.*}}@changed_layout_f16
  // CHECK: %[[changed_layout_f16A:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[changed_layout_f16B:.*]] = llvm.extractvalue %arg0[1]
  // CHECK: rocdl.hcu.cvt.pk.f16.f32{{.*}}%[[changed_layout_f16A]]{{.*}}%[[changed_layout_f16B]]
  tt.func private @changed_layout_f16(%arg0: tensor<32x32xf32, #layout>, %ptr: !tt.ptr<f16>) -> tensor<32x32xf16, #layout> {
    %v = arith.truncf %arg0 : tensor<32x32xf32, #layout> to tensor<32x32xf16, #layout>
    %different = ttg.convert_layout %v : tensor<32x32xf16, #layout> -> tensor<32x32xf16, #blocked>
    %back = ttg.convert_layout %different : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #layout>
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %s32 = arith.constant 32 : i64
    %s1 = arith.constant 1 : i64
    amdg.matrix_store_from_reg %back, %ptr, shape = [%c32, %c32], strides = [%s32, %s1], tensorShape = [32, 32], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<32x32xf16, #layout>, <f16>
    tt.return %v : tensor<32x32xf16, #layout>
  }
  // CHECK-LABEL: llvm.func {{.*}}@direct_fp32_store
  // CHECK: %[[direct_fp32_storeA:.*]] = llvm.extractvalue %arg0[0]
  // CHECK: %[[direct_fp32_storeB:.*]] = llvm.extractvalue %arg0[4]
  // CHECK: rocdl.hcu.cvt.pk.f16.f32{{.*}}%[[direct_fp32_storeA]]{{.*}}%[[direct_fp32_storeB]]
  tt.func private @direct_fp32_store(%arg0: tensor<32x32xf32, #layout>, %ptr: !tt.ptr<f16>) {
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %s32 = arith.constant 32 : i64
    %s1 = arith.constant 1 : i64
    amdg.matrix_store_from_reg %arg0, %ptr, shape = [%c32, %c32], strides = [%s32, %s1], tensorShape = [32, 32], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<32x32xf32, #layout>, <f16>
    tt.return
  }
  // Explicit RTZ must not be replaced by the store's RTNE packed conversion.
  // CHECK-LABEL: llvm.func {{.*}}@rtz_f16
  // CHECK-NOT: rocdl.hcu.cvt.pk.f16.f32
  // CHECK: llvm.return
  tt.func private @rtz_f16(%arg0: tensor<32x32xf32, #layout>, %ptr: !tt.ptr<f16>) -> tensor<32x32xf16, #layout> {
    %v = tt.fp_to_fp %arg0, rounding = rtz : tensor<32x32xf32, #layout> -> tensor<32x32xf16, #layout>
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %s32 = arith.constant 32 : i64
    %s1 = arith.constant 1 : i64
    amdg.matrix_store_from_reg %v, %ptr, shape = [%c32, %c32], strides = [%s32, %s1], tensorShape = [32, 32], indices = [%zero, %zero] {MlsStoreEncoding = #store} : tensor<32x32xf16, #layout>, <f16>
    tt.return %v : tensor<32x32xf16, #layout>
  }
}
