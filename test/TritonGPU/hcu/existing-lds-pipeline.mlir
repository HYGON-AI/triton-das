// RUN: triton-opt %s --tritonamdgpu-schedule-loops=num_stages=2 --tritonamdgpu-pipeline=use_async_copy=true --canonicalize --cse | FileCheck %s
// Existing LDS producer/consumer chains use the ordinary scheduler without
// a scale marker or a dot. REG values are carried by SSA; direct copies rotate
// LDS storage. Both use standard initialized local_alloc/local_load SSA.
#blocked = #ttg.blocked<{sizePerThread=[1,2], threadsPerWarp=[16,4], warpsPerCTA=[4,1], order=[1,0]}>
#loader = #ttg.blocked<{sizePerThread=[1,8], threadsPerWarp=[64,1], warpsPerCTA=[4,1], order=[1,0]}>
#shared = #ttg.swizzled_shared<{vec=4, perPhase=1, maxPhase=1, order=[1,0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps"=4:i32, "ttg.threads-per-warp"=64:i32, "ttg.num-ctas"=1:i32, ttg.target="hip:gfx946"} {
  // CHECK-LABEL: @existing_reg
  // CHECK: tt.load
  // CHECK-NOT: tt.load
  // CHECK: ttg.local_store
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: ttg.local_store
  // CHECK: ttg.local_load
  tt.func public @existing_reg(%ptr: tensor<64x8x!tt.ptr<i8>, #blocked>, %n:i32) -> tensor<64x8xi8, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<64x8xi8, #blocked>
    %result = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<64x8xi8, #blocked> : i32 {
      %offset = tt.splat %iv : i32 -> tensor<64x8xi32, #blocked>
      %p = tt.addptr %ptr, %offset : tensor<64x8x!tt.ptr<i8>, #blocked>, tensor<64x8xi32, #blocked>
      %value = tt.load %p : tensor<64x8x!tt.ptr<i8>, #blocked>
      %lds = ttg.local_alloc %value : (tensor<64x8xi8, #blocked>) -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable>
      %loaded = ttg.local_load %lds : !ttg.memdesc<64x8xi8, #shared, #smem, mutable> -> tensor<64x8xi8, #blocked>
      %sum = arith.addi %acc, %loaded : tensor<64x8xi8, #blocked>
      scf.yield %sum : tensor<64x8xi8, #blocked>
    } {tt.num_stages=2:i32}
    tt.return %result : tensor<64x8xi8, #blocked>
  }
  // CHECK-LABEL: @existing_async
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x8xi8
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: scf.for
  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  tt.func public @existing_async(%ptr: tensor<64x8x!tt.ptr<i8>, #loader> {tt.contiguity=dense<[1,8]> : tensor<2xi32>, tt.divisibility=dense<[16,16]> : tensor<2xi32>}, %n:i32) -> tensor<64x8xi8, #loader> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %tile = arith.constant 512 : i32
    %zero = arith.constant dense<0> : tensor<64x8xi8, #loader>
    %result = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %zero) -> tensor<64x8xi8, #loader> : i32 {
      %tileOffset = arith.muli %iv, %tile : i32
      %offset = tt.splat %tileOffset : i32 -> tensor<64x8xi32, #loader>
      %p = tt.addptr %ptr, %offset : tensor<64x8x!tt.ptr<i8>, #loader>, tensor<64x8xi32, #loader>
      %value = tt.load %p : tensor<64x8x!tt.ptr<i8>, #loader>
      %lds = ttg.local_alloc %value : (tensor<64x8xi8, #loader>) -> !ttg.memdesc<64x8xi8, #shared, #smem, mutable>
      %loaded = ttg.local_load %lds : !ttg.memdesc<64x8xi8, #shared, #smem, mutable> -> tensor<64x8xi8, #loader>
      %sum = arith.addi %acc, %loaded : tensor<64x8xi8, #loader>
      scf.yield %sum : tensor<64x8xi8, #loader>
    } {tt.num_stages=2:i32}
    tt.return %result : tensor<64x8xi8, #loader>
  }
}
