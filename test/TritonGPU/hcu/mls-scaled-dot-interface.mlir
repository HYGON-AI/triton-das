// RUN: triton-opt %s --tritonhcu-mls-encoding-insertion | FileCheck %s --check-prefixes=COMMON,NUMERIC
// RUN: triton-opt %s --tritonhcu-scale-pipeline --tritonhcu-mls-encoding-insertion | FileCheck %s --check-prefixes=COMMON,AMD
// MLS selection must work with either scaled-dot form, including when this
// pass is the only pass responsible for loading the AMD dialect.

// COMMON-LABEL: tt.func public @scaled_mls
// COMMON: tt.matrix_load
// COMMON-SAME: MlsEncoding = #amdg<MlsEncoding opIdx = 0, mlsTile = [32, 128]
// COMMON-SAME: elemBitTyKind = 1
// COMMON: tt.matrix_load
// COMMON-SAME: MlsEncoding = #amdg<MlsEncoding opIdx = 1, mlsTile = [128, 32]
// COMMON-SAME: elemBitTyKind = 1
// NUMERIC: tt.dot_scaled
// AMD: amdg.scale_write
// AMD: amdg.scale_write
// AMD: amdg.dot_scaled

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2, tilesPerWarp = [2, 2]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8, mlsScaledExt = 1}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8, mlsScaledExt = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx946"} {
  tt.func public @scaled_mls(%A: !tt.ptr<i8>, %B: !tt.ptr<i8>,
                            %sa: tensor<64x8xi8, #blocked>,
                            %sb: tensor<64x8xi8, #blocked>)
      -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %s1 = arith.constant 1 : i64
    %s128 = arith.constant 128 : i64
    %acc = arith.constant dense<0.0> : tensor<64x64xf32, #mma>
    %av = tt.matrix_load %A, shape = [%c64, %c128],
        strides = [%s128, %s1], tensorShape = [64, 128], indices = [%c0, %c0]
        : <i8> -> tensor<64x128xi8, #blocked>
    %bv = tt.matrix_load %B, shape = [%c128, %c64],
        strides = [%s1, %s128], tensorShape = [128, 64], indices = [%c0, %c0]
        : <i8> -> tensor<128x64xi8, #blocked>
    %a = ttg.convert_layout %av
        : tensor<64x128xi8, #blocked> -> tensor<64x128xi8, #a>
    %b = ttg.convert_layout %bv
        : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #b>
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %acc
        lhs = e2m1 rhs = e2m1 {fastMath = false}
        : tensor<64x128xi8, #a>, tensor<64x8xi8, #blocked> *
          tensor<128x64xi8, #b>, tensor<64x8xi8, #blocked>
          -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}
