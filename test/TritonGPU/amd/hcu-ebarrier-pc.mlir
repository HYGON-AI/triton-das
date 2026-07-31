// Verify runtime-correct ebarrier-only producer/consumer TTGIR lowers to true LLIR.
// The TTGIR input is inlined below; triton-opt runs the HCU lowering pipeline,
// Triton's mlir-translate emits LLVM IR, and FileCheck validates the LLIR.
//
// RUN: triton-opt %s \
// RUN:   --allocate-shared-memory \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-triton-hcu-to-llvm="arch=gfx946" \
// RUN:   --triton-hcu-convert-warp-specialize-to-llvm="arch=gfx946 wasp_num_load_warps=4 wasp_num_mma_warps=4 wdra-enabled=false" \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 8 : i32} {
  tt.func public @ebarrier_pc_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%0, %arg0, %arg1, %arg2) attributes {requestedRegisters = array<i32: 88, 88>, warpGroupStartIds = array<i32: 0, 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg3: !ttg.memdesc<1024xf32, #shared, #smem, mutable>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32) num_warps(4) {
      %1 = llvm.mlir.constant(9 : i32) : i32
      %2 = llvm.mlir.constant(10 : i32) : i32
      %3 = llvm.mlir.constant(11 : i32) : i32
      %4 = llvm.mlir.constant(12 : i32) : i32
      %c0_i32_0 = arith.constant 0 : i32
      %c1_i32_1 = arith.constant 1 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
      %5 = scf.for %arg7 = %c0_i32_0 to %arg6 step %c1_i32_1 iter_args(%arg8 = %cst) -> (tensor<1024xf32, #blocked>) : i32 {
        rocdl.hcu.s.ebarrier.sync %1, 8
        %6 = ttg.local_load %arg3 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
        %7 = arith.addf %arg8, %6 : tensor<1024xf32, #blocked>
        rocdl.hcu.s.ebarrier.sync %2, 8
        scf.yield %7 : tensor<1024xf32, #blocked>
      }
      %8 = rocdl.hcu.s.ebarrier.and i32, %3, 4
      %9 = arith.cmpi eq, %8, %c1_i32_1 : i32
      %10 = rocdl.hcu.s.ebarrier.or i32, %3, 4
      %11 = arith.select %9, %10, %c0_i32_0 : i32
      %12 = rocdl.hcu.s.ebarrier.popc i32, %3, 4
      %13 = arith.addi %11, %12 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = tt.splat %14 : f32 -> tensor<1024xf32, #blocked>
      %16 = arith.addf %5, %15 : tensor<1024xf32, #blocked>
      %17 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
      %18 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
      %19 = tt.addptr %18, %17 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %19, %16 : tensor<1024x!tt.ptr<f32>, #blocked>
      rocdl.hcu.s.ebarrier.arrive %4, 4
      ttg.warp_return
    }
    partition1(%arg3: !ttg.memdesc<1024xf32, #shared, #smem, mutable>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32) num_warps(4) {
      %1 = llvm.mlir.constant(9 : i32) : i32
      %2 = llvm.mlir.constant(10 : i32) : i32
      %3 = llvm.mlir.constant(13 : i32) : i32
      %c0_i32_0 = arith.constant 0 : i32
      %c1_i32_1 = arith.constant 1 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
      %5 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
      scf.for %arg7 = %c0_i32_0 to %arg6 step %c1_i32_1 : i32 {
        %6 = arith.muli %arg7, %c1024_i32 : i32
        %7 = tt.splat %6 : i32 -> tensor<1024xi32, #blocked>
        %8 = arith.addi %7, %4 : tensor<1024xi32, #blocked>
        %9 = tt.addptr %5, %8 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
        ttg.local_store %10, %arg3 : tensor<1024xf32, #blocked> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
        rocdl.hcu.s.ebarrier.sync %1, 8
        rocdl.hcu.s.ebarrier.sync %2, 8
      }
      rocdl.hcu.s.ebarrier.arrive %3, 4
      ttg.warp_return
    } : (!ttg.memdesc<1024xf32, #shared, #smem, mutable>, !tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
    tt.return
  }
}

// CHECK: @global_smem = external addrspace(3) global
// CHECK-LABEL: define ptx_kernel void @ebarrier_pc_kernel
// CHECK: call {{.*}} @llvm.hcu.get.wave.id
// CHECK: switch i32

// Cross-partition rendezvous uses wvCnt = 8 (barrierId, wvCnt).
// CHECK-DAG: call void @llvm.hcu.s.ebarrier.sync(i32 {{[0-9]+}}, i32 8)

// Per-partition non-blocking arrive uses wvCnt = 4.
// CHECK-DAG: call void @llvm.hcu.s.ebarrier.arrive(i32 {{[0-9]+}}, i32 4)

// Reduction primitives; results stay live through icmp/select/add.
// CHECK-DAG: call i32 @llvm.hcu.s.ebarrier.and(i32 {{[0-9]+}}, i32 4)
// CHECK-DAG: call i32 @llvm.hcu.s.ebarrier.or(i32 {{[0-9]+}}, i32 4)
// CHECK-DAG: call i32 @llvm.hcu.s.ebarrier.popc(i32 {{[0-9]+}}, i32 4)
// CHECK-DAG: icmp eq i32
// CHECK-DAG: select i1
// CHECK-DAG: add i32

// This fixture is ebarrier-only.
// CHECK-NOT: @llvm.hcu.s.abarrier