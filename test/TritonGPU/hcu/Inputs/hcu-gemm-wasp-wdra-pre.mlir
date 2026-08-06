#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 2}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x128xi32, #blocked1>
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = arith.cmpi sgt, %arg6, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %1 = arith.cmpi sgt, %arg7, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    %3 = tt.get_program_id x : i32
    %4 = arith.addi %arg3, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg4, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.divsi %3, %7 : i32
    %9 = arith.subi %5, %8 : i32
    %10 = arith.minsi %9, %c1_i32 : i32
    %11 = arith.remsi %3, %7 : i32
    %12 = arith.remsi %11, %10 : i32
    %13 = arith.addi %8, %12 : i32
    %14 = arith.divsi %11, %10 : i32
    %15 = arith.cmpi sgt, %13, %c0_i32 : i32
    llvm.intr.assume %15 : i1
    %16 = arith.cmpi sgt, %14, %c0_i32 : i32
    llvm.intr.assume %16 : i1
    %17 = arith.muli %13, %c128_i32 : i32
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %21 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %22 = arith.addi %21, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %23 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %24 = arith.remsi %22, %23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %25 = arith.muli %14, %c128_i32 : i32
    %26 = tt.splat %25 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %27 = tt.splat %25 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.addi %26, %19 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = arith.addi %27, %20 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %30 = tt.splat %arg4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %31 = arith.remsi %28, %30 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %32 = tt.expand_dims %24 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %33 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %34 = arith.muli %32, %33 : tensor<128x1xi32, #blocked1>
    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %36 = tt.expand_dims %35 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %37 = tt.broadcast %34 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %38 = tt.broadcast %36 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %39 = arith.addi %37, %38 : tensor<128x128xi32, #blocked1>
    %40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %41 = tt.expand_dims %40 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %42 = tt.expand_dims %31 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %43 = tt.splat %arg7 : i32 -> tensor<1x128xi32, #blocked>
    %44 = arith.muli %42, %43 : tensor<1x128xi32, #blocked>
    %45 = tt.broadcast %41 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %46 = tt.broadcast %44 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %47 = arith.addi %45, %46 : tensor<128x128xi32, #blocked>
    %48 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %49 = tt.addptr %48, %39 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %50 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %51 = tt.addptr %50, %47 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %52 = arith.addi %arg5, %c127_i32 : i32
    %53 = arith.divsi %52, %c128_i32 : i32
    %54:3 = scf.for %arg9 = %c0_i32 to %53 step %c1_i32 iter_args(%arg10 = %cst_1, %arg11 = %49, %arg12 = %51) -> (tensor<128x128xf32, #mma>, tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128x!tt.ptr<f16>, #blocked>)  : i32 {
      %73 = tt.load %arg11 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %74 = tt.load %arg12 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %75 = ttg.local_alloc %73 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %76 = ttg.local_load %75 : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %77 = ttg.local_alloc %74 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %78 = ttg.local_load %77 : !ttg.memdesc<128x128xf16, #shared1, #smem> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %79 = tt.dot %76, %78, %arg10, inputPrecision = tf32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
      %80 = tt.addptr %arg11, %cst_0 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
      %81 = tt.addptr %arg12, %cst : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
      scf.yield %79, %80, %81 : tensor<128x128xf32, #mma>, tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128x!tt.ptr<f16>, #blocked>
    } {tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
    %55 = arith.truncf %54#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %56 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %57 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %58 = arith.muli %57, %56 : tensor<128x1xi32, #blocked1>
    %59 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %60 = tt.addptr %59, %58 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %61 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %62 = tt.broadcast %60 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %63 = tt.broadcast %61 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %64 = tt.addptr %62, %63 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %65 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %66 = arith.cmpi slt, %56, %65 : tensor<128x1xi32, #blocked1>
    %67 = tt.splat %arg4 : i32 -> tensor<1x128xi32, #blocked1>
    %68 = arith.cmpi slt, %61, %67 : tensor<1x128xi32, #blocked1>
    %69 = tt.broadcast %66 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %70 = tt.broadcast %68 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %71 = arith.andi %69, %70 : tensor<128x128xi1, #blocked1>
    %72 = ttg.convert_layout %55 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.store %64, %72, %71 : tensor<128x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

