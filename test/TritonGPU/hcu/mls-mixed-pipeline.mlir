// Modified by Hygon Information Technology Co., Ltd., 2026.
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=2 use_async_copy=0" -tritonamdgpu-schedule-loops=num_stages=2 -tritonamdgpu-pipeline | FileCheck %s
// RUN: triton-opt %s -tritonhcu-mls-stream-pipeline="num_stages=3 use_async_copy=0" -tritonamdgpu-schedule-loops=num_stages=3 -tritonamdgpu-pipeline="use_async_copy=0" --tritonhcu-update-async-wait-count=arch-generation-name=gfx938 | FileCheck %s --check-prefix=MULTI

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx938", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func public @matmul_kernel
  // CHECK-COUNT-1: scf.for
  // CHECK:   ttg.async_wait
  // CHECK-NEXT:   ttg.local_load
  // CHECK-NEXT:   ttg.local_load
  // CHECK:   amdg.matrix_load_to_local
  // CHECK:   ttg.async_commit_group
  // CHECK:   tt.load
  // CHECK:   tt.dot
  // CHECK:   ttg.local_store
  // Only the MLS operand contributes a commit group, not the ordinary tt.load.
  // MULTI-LABEL: tt.func public @matmul_kernel
  // MULTI: scf.for
  // MULTI: ttg.async_wait %{{[^ ,]+}} {num = 1 : i32}
  // MULTI: amdg.matrix_load_to_local
  // MULTI: ttg.async_commit_group
  // MULTI: tt.load
  // MULTI: scf.yield
  // MULTI: ttg.async_wait {{.*}} {num = 0 : i32}
  tt.func public @matmul_kernel(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: i32 {tt.divisibility = 16 : i32},
      %arg4: i32 {tt.divisibility = 16 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32},
      %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32},
      %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x256xi32, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = arith.cmpi sgt, %arg6, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %1 = arith.cmpi sgt, %arg7, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    %3 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    %4 = arith.cmpi sgt, %arg4, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    %5 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg4, %c255_i32 : i32
    %8 = arith.divsi %7, %c256_i32 : i32
    %9 = arith.divsi %6, %8 : i32
    %10 = arith.remsi %6, %8 : i32
    %11 = arith.muli %9, %c256_i32 : i32
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %13 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %15 = tt.splat %11 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %16 = arith.addi %15, %12 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %17 = arith.muli %10, %c256_i32 : i32
    %18 = tt.splat %17 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.splat %17 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %20 = arith.addi %18, %13 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %21 = arith.addi %19, %14 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %22 = tt.expand_dims %16 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xi32, #mma>
    %23 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %25 = tt.expand_dims %20 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %26 = tt.expand_dims %21 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi32, #mma>
    %27 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %28 = arith.muli %25, %27 : tensor<1x256xi32, #blocked>
    %29 = tt.broadcast %24 : tensor<32x1xi32, #blocked> -> tensor<32x256xi32, #blocked>
    %30 = tt.broadcast %28 : tensor<1x256xi32, #blocked> -> tensor<32x256xi32, #blocked>
    %31 = arith.addi %29, %30 : tensor<32x256xi32, #blocked>
    %32 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x256x!tt.ptr<f16>, #blocked>
    %33 = tt.addptr %32, %31 : tensor<32x256x!tt.ptr<f16>, #blocked>, tensor<32x256xi32, #blocked>
    %34 = arith.extsi %arg6 : i32 to i64
    %35:2 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32
        iter_args(%arg10 = %cst_0, %arg11 = %33)
        -> (tensor<256x256xf32, #mma>, tensor<32x256x!tt.ptr<f16>, #blocked>) : i32 {
      %44 = tt.matrix_load %arg0, shape = [%arg3, %arg5],
          strides = [%34, %c1_i64], tensorShape = [256, 32],
          indices = [%11, %arg9]
          {MlsEncoding = #amdg<MlsEncoding opIdx = 0, mlsTile = [16, 32], elemBitWidth = 16, elemBitTyKind = 0, alt2Kind = 0, version = 1, order = [1, 0], warpsPerCTA = [8, 1]>}
          : <f16> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %45 = tt.load %arg11 : tensor<32x256x!tt.ptr<f16>, #blocked>
      %46 = ttg.convert_layout %45 : tensor<32x256xf16, #blocked> -> tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %47 = tt.dot %44, %46, %arg10, inputPrecision = tf32
          : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
          * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
          -> tensor<256x256xf32, #mma>
      %48 = tt.addptr %arg11, %cst : tensor<32x256x!tt.ptr<f16>, #blocked>, tensor<32x256xi32, #blocked>
      scf.yield %47, %48 : tensor<256x256xf32, #mma>, tensor<32x256x!tt.ptr<f16>, #blocked>
    }
    %36 = arith.truncf %35#0 : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %37 = tt.splat %arg8 : i32 -> tensor<256x1xi32, #mma>
    %38 = arith.muli %37, %22 : tensor<256x1xi32, #mma>
    %39 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #mma>
    %40 = tt.addptr %39, %38 : tensor<256x1x!tt.ptr<f16>, #mma>, tensor<256x1xi32, #mma>
    %41 = tt.broadcast %40 : tensor<256x1x!tt.ptr<f16>, #mma> -> tensor<256x256x!tt.ptr<f16>, #mma>
    %42 = tt.broadcast %26 : tensor<1x256xi32, #mma> -> tensor<256x256xi32, #mma>
    %43 = tt.addptr %41, %42 : tensor<256x256x!tt.ptr<f16>, #mma>, tensor<256x256xi32, #mma>
    tt.store %43, %36 : tensor<256x256x!tt.ptr<f16>, #mma>
    tt.return
  }
}
