// RUN: triton-opt %s \
// RUN:   -convert-triton-to-tritongpu="target=hip:gfx938 num-warps=8 threads-per-warp=64 num-ctas=1" \
// RUN:   -tritonhcu-utc-warmup | FileCheck %s

module {
  // The outer loop models a GEMM nested in a persistent/grouped loop. Each
  // outer iteration selects a different A/B panel; UTC warmup must therefore
  // remain inside the outer loop and surround only the inner K reduction.
  //
  // CHECK-LABEL: tt.func public @nested_matrix_load_gemm
  // CHECK-NOT:   amdg.utc_warmup
  // CHECK:       scf.for %[[OUTER:.+]] =
  // CHECK:         amdg.utc_warmup
  // CHECK:         amdg.utc_warmup
  // CHECK:         %[[INNER_RESULT:.+]] = scf.for %[[K:.+]] =
  // CHECK-NOT:       hcu.utc_warmup.id
  // CHECK:           tt.matrix_load
  // CHECK-NOT:       hcu.utc_warmup.id
  // CHECK:           tt.matrix_load
  // CHECK-NOT:       hcu.utc_warmup.id
  // CHECK:           tt.dot
  // CHECK:         amdg.utc_warmup_consume
  // CHECK-NEXT:    amdg.utc_warmup_consume
  // CHECK-NOT:     amdg.utc_warmup
  // CHECK:         scf.yield %[[INNER_RESULT]]
  // CHECK-NEXT:  }
  // CHECK-NOT:   amdg.utc_warmup
  // CHECK:       tt.return
  tt.func public @nested_matrix_load_gemm(
      %a_ptr: !tt.ptr<f16>,
      %b_ptr: !tt.ptr<f16>,
      %m: i32,
      %n: i32,
      %k: i32,
      %stride_am: i64,
      %stride_bn: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<256x256xf32>

    %result = scf.for %outer = %c0_i32 to %c2_i32 step %c1_i32
        iter_args(%outer_acc = %zero) -> (tensor<256x256xf32>) : i32 {
      %m_index = arith.muli %outer, %c256_i32 : i32
      %n_index = arith.muli %outer, %c256_i32 : i32
      %inner_result = scf.for %iv = %c0_i32 to %k step %c32_i32
          iter_args(%acc = %outer_acc) -> (tensor<256x256xf32>) : i32 {
        %a = tt.matrix_load %a_ptr,
            shape = [%m, %k],
            strides = [%stride_am, %c1_i64],
            tensorShape = [256, 32],
            indices = [%m_index, %iv]
            : <f16> -> tensor<256x32xf16>
        %b = tt.matrix_load %b_ptr,
            shape = [%k, %n],
            strides = [%c1_i64, %stride_bn],
            tensorShape = [32, 256],
            indices = [%iv, %n_index]
            : <f16> -> tensor<32x256xf16>
        %next = tt.dot %a, %b, %acc, inputPrecision = tf32
            : tensor<256x32xf16>
            * tensor<32x256xf16>
            -> tensor<256x256xf32>
        scf.yield %next : tensor<256x256xf32>
      }
      scf.yield %inner_result : tensor<256x256xf32>
    }
    tt.return
  }
}
