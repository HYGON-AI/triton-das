// Verify runtime-correct abarrier-only producer/consumer TTGIR lowers to true LLIR.
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
  tt.func public @abarrier_pc_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %ready = llvm.mlir.constant(8 : i32) : i32
    %done  = llvm.mlir.constant(9 : i32) : i32
    rocdl.hcu.s.abarrier.init %ready, 4
    rocdl.hcu.s.abarrier.init %done,  4
    %seed = rocdl.hcu.s.abarrier.arrive i32, %done, 1
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%0, %arg0, %arg1, %arg2) attributes {requestedRegisters = array<i32: 88, 88>, warpGroupStartIds = array<i32: 0, 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg3: !ttg.memdesc<1024xf32, #shared, #smem, mutable>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32) num_warps(4) {
      %1 = llvm.mlir.constant(8 : i32) : i32
      %2 = llvm.mlir.constant(9 : i32) : i32
      %c0_i32_0 = arith.constant 0 : i32
      %c1_i32_1 = arith.constant 1 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
      %3:2 = scf.for %arg7 = %c0_i32_0 to %arg6 step %c1_i32_1 iter_args(%arg8 = %cst, %arg9 = %c0_i32_0) -> (tensor<1024xf32, #blocked>, i32) : i32 {
        %4 = rocdl.hcu.s.abarrier.try.wait i32, %1, %arg9
        %5 = ttg.local_load %arg3 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
        %6 = arith.addf %arg8, %5 : tensor<1024xf32, #blocked>
        %7 = rocdl.hcu.s.abarrier.arrive i32, %2, 1
        %8 = arith.xori %arg9, %c1_i32_1 : i32
        scf.yield %6, %8 : tensor<1024xf32, #blocked>, i32
      }
      %9  = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
      %10 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
      %11 = tt.addptr %10, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %11, %3#0 : tensor<1024x!tt.ptr<f32>, #blocked>
      rocdl.hcu.s.abarrier.inv %1
      rocdl.hcu.s.abarrier.inv %2
      ttg.warp_return
    }
    partition1(%arg3: !ttg.memdesc<1024xf32, #shared, #smem, mutable>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32) num_warps(4) {
      %1 = llvm.mlir.constant(8 : i32) : i32
      %2 = llvm.mlir.constant(9 : i32) : i32
      %c0_i32_0 = arith.constant 0 : i32
      %c1_i32_1 = arith.constant 1 : i32
      %c1024_i32 = arith.constant 1024 : i32
      %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
      %4 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
      scf.for %arg7 = %c0_i32_0 to %arg6 step %c1_i32_1 iter_args(%arg8 = %c0_i32_0) -> (i32) : i32 {
        %5 = rocdl.hcu.s.abarrier.try.wait i32, %2, %arg8
        %6 = arith.muli %arg7, %c1024_i32 : i32
        %7 = tt.splat %6 : i32 -> tensor<1024xi32, #blocked>
        %8 = arith.addi %7, %3 : tensor<1024xi32, #blocked>
        %9 = tt.addptr %4, %8 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>, #blocked>
        ttg.local_store %10, %arg3 : tensor<1024xf32, #blocked> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
        %11 = rocdl.hcu.s.abarrier.arrive i32, %1, 1
        %12 = arith.xori %arg8, %c1_i32_1 : i32
        scf.yield %12 : i32
      }
      ttg.warp_return
    } : (!ttg.memdesc<1024xf32, #shared, #smem, mutable>, !tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
    tt.return
  }
}

// CHECK: @global_smem = external addrspace(3) global
// CHECK-LABEL: define ptx_kernel void @abarrier_pc_kernel
// CHECK: call {{.*}} @llvm.hcu.get.wave.id
// CHECK: switch i32

// Init on ids 8/9 (avoid warp_specialize reserves 0..7). Each barrier expects
// 4 arrivals from the signaling partition.
// CHECK-DAG: call void @llvm.hcu.s.abarrier.init(i32 8, i32 4)
// CHECK-DAG: call void @llvm.hcu.s.abarrier.init(i32 9, i32 4)

// Seed on `done` (id 9): 4 default waves x cnt=1 == expected=4, so `done`
// phase 0 completes at entry and producer can enter iter 0 without waiting.
// CHECK-DAG: call i32 {{.*}}@llvm.hcu.s.abarrier.arrive(i32 9, i32 1)

// Runtime handshake: consumer waits ready(8) and arrives done(9); producer
// waits done(9) and arrives ready(8). All in-loop arrives use cnt=1
// (4 waves x 1 == expected 4 per iteration).
// CHECK-DAG: call i32 {{.*}}@llvm.hcu.s.abarrier.try.wait(i32 8, i32 {{.*}})
// CHECK-DAG: call i32 {{.*}}@llvm.hcu.s.abarrier.try.wait(i32 9, i32 {{.*}})
// CHECK-DAG: call i32 {{.*}}@llvm.hcu.s.abarrier.arrive(i32 8, i32 1)

// Cleanup: only consumer invalidates each barrier (single inv per id).
// CHECK-DAG: call void @llvm.hcu.s.abarrier.inv(i32 8)
// CHECK-DAG: call void @llvm.hcu.s.abarrier.inv(i32 9)