// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls | FileCheck %s --check-prefix=DISABLED --implicit-check-not=tt.matrix_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_ --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: env TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS=0 triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx946" | FileCheck %s --check-prefix=DISABLED --implicit-check-not=tt.matrix_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx942" | FileCheck %s --check-prefix=DISABLED --implicit-check-not=tt.matrix_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx946" | FileCheck %s --check-prefix=SELECT
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx946" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx948" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: env TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS=0 triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx946" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_ --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx950" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_ --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx942" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_ --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx92a" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_ --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: triton-opt %s --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx938" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_ --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_
// RUN: triton-opt %s --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse | FileCheck %s --check-prefix=GENERIC --implicit-check-not=tt.matrix_
// RUN: env TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS=0 triton-opt %S/../../Triton/rewrite-tensor-descriptor-to-pointer.mlir --tritonhcu-rewrite-tensor-descriptor-to-mls="arch=gfx946" --triton-rewrite-tensor-descriptor-to-pointer --canonicalize --cse --split-input-file | FileCheck %S/../../Triton/rewrite-tensor-descriptor-to-pointer.mlir --implicit-check-not=tt.tensordesc

module {
  // A descriptor may legally produce a rank-reduced load result. MLS requires
  // a two-dimensional result as well as a two-dimensional descriptor.
  // SELECT-LABEL: @rank_reduced_load
  // SELECT-NOT: tt.matrix_load
  // SELECT: tt.descriptor_load
  // CHECK-LABEL: @rank_reduced_load
  // CHECK-NOT: tt.matrix_load
  // CHECK: tt.return
  tt.func public @rank_reduced_load(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %d = tt.make_tensor_descriptor %p, [%c1, %c64], [%s64, %s1] : <f16>, <tensor<1x64xf16>>
    %v = tt.descriptor_load %d[%c0, %c0] : !tt.tensordesc<tensor<1x64xf16>> -> tensor<64xf16>
    tt.return
  }

  // SELECT-LABEL: @aligned_dot
  // SELECT: tt.matrix_load
  // CHECK-LABEL: @aligned_dot
  // CHECK: tt.matrix_load
  // CHECK-NOT: boundaryCheck
  // CHECK: tt.dot
  // GENERIC-LABEL: @aligned_dot
  // GENERIC: tt.load
  tt.func public @aligned_dot(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %acc = arith.constant dense<0.0> : tensor<64x64xf32>
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
    %v = tt.descriptor_load %d[%c0, %c0] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    %result = tt.dot %v, %b, %acc : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    tt.return %result : tensor<64x64xf32>
  }

  // Standalone loads stay generic even with an MLS-compatible address and tile.
  // DISABLED-LABEL: @aligned_load
  // DISABLED: tt.descriptor_load
  // SELECT-LABEL: @aligned_load
  // SELECT-NOT: tt.matrix_load
  // SELECT: tt.descriptor_load
  // SELECT: tt.return
  // CHECK-LABEL: @aligned_load
  // CHECK-NOT: tt.matrix_load
  // CHECK: tt.load
  // GENERIC-LABEL: @aligned_load
  // GENERIC: tt.load
  tt.func public @aligned_load(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x64xf16> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
    %v = tt.descriptor_load %d[%c0, %c0] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    tt.return %v : tensor<64x64xf16>
  }

  // While yields map to before-region arguments, not operation results. This
  // loop has two yielded values but only one result. Keep its load generic.
  // SELECT-LABEL: @while_result_arity
  // SELECT-NOT: tt.matrix_load
  // SELECT: tt.descriptor_load
  // SELECT-NOT: tt.matrix_load
  // SELECT: tt.return
  // CHECK-LABEL: @while_result_arity
  // CHECK-NOT: tt.matrix_load
  // CHECK: tt.load
  // GENERIC-LABEL: @while_result_arity
  // GENERIC: tt.load
  tt.func public @while_result_arity(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %zero = arith.constant dense<0.0> : tensor<64x64xf16>
    %r = scf.while (%i = %c0, %v = %zero) : (i32, tensor<64x64xf16>) -> tensor<64x64xf16> {
      %cond = arith.cmpi slt, %i, %c1 : i32
      scf.condition(%cond) %v : tensor<64x64xf16>
    } do {
    ^bb0(%v: tensor<64x64xf16>):
      %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
      %loaded = tt.descriptor_load %d[%c0, %c0] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
      scf.yield %c1, %loaded : i32, tensor<64x64xf16>
    }
    %acc = arith.constant dense<0.0> : tensor<64x64xf32>
    %result = tt.dot %r, %b, %acc : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    tt.return %result : tensor<64x64xf32>
  }

  // CHECK-LABEL: @bounded_runtime_index
  // CHECK: tt.matrix_load
  // GENERIC-LABEL: @bounded_runtime_index
  // GENERIC: tt.load
  tt.func public @bounded_runtime_index(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row: i32, %b: tensor<32x64xf16>) -> tensor<16x64xf32> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c1024 = arith.constant 1024 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %lo = arith.cmpi sge, %row, %c0 : i32
    %hi = arith.cmpi sle, %row, %c1024 : i32
    llvm.intr.assume %lo : i1
    llvm.intr.assume %hi : i1
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<16x32xf16>>
    %v = tt.descriptor_load %d[%row, %c0] : !tt.tensordesc<tensor<16x32xf16>> -> tensor<16x32xf16>
    %acc = arith.constant dense<0.0> : tensor<16x64xf32>
    %result = tt.dot %v, %b, %acc : tensor<16x32xf16> * tensor<32x64xf16> -> tensor<16x64xf32>
    tt.return %result : tensor<16x64xf32>
  }

  // SELECT-LABEL: @unknown_runtime_index
  // SELECT: tt.descriptor_load
  // CHECK-LABEL: @unknown_runtime_index
  // CHECK-NOT: tt.matrix_load
  // CHECK: tt.load
  // GENERIC-LABEL: @unknown_runtime_index
  // GENERIC: tt.load
  tt.func public @unknown_runtime_index(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %row: i32, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
    %v = tt.descriptor_load %d[%row, %c0] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32>
    %result = tt.dot %v, %b, %acc : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    tt.return %result : tensor<64x64xf32>
  }

  // SELECT-LABEL: @store_fallback
  // SELECT: tt.descriptor_store
  // CHECK-LABEL: @store_fallback
  // CHECK: tt.store
  // CHECK-NOT: tt.matrix_store
  // GENERIC-LABEL: @store_fallback
  // GENERIC: tt.store
  tt.func public @store_fallback(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x64xf16>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
    tt.descriptor_store %d[%c0, %c0], %v : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16>
    tt.return
  }

  // The HCU pass selects the default-policy load; the generic pass lowers
  // the other access of the same descriptor and preserves its cache policy.
  // CHECK-LABEL: @mixed_loads
  // CHECK: tt.matrix_load
  // CHECK: tt.load {{.*}} cacheModifier = cg
  // GENERIC-LABEL: @mixed_loads
  // GENERIC: tt.load
  // GENERIC: tt.load {{.*}} cacheModifier = cg
  tt.func public @mixed_loads(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
    %a = tt.descriptor_load %d[%c0, %c0] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    %b = tt.descriptor_load %d[%c0, %c0] cacheModifier = cg : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32>
    %result = tt.dot %a, %b, %acc : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    tt.return %result : tensor<64x64xf32>
  }

  // CHECK-LABEL: @cache_policy
  // CHECK-NOT: tt.matrix_load
  // CHECK: tt.load {{.*}} cacheModifier = cg evictionPolicy = evict_last
  // GENERIC-LABEL: @cache_policy
  // GENERIC: tt.load {{.*}} cacheModifier = cg evictionPolicy = evict_last
  tt.func public @cache_policy(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %s1 = arith.constant 1 : i64
    %s64 = arith.constant 64 : i64
    %d = tt.make_tensor_descriptor %p, [%c64, %c64], [%s64, %s1] : <f16>, <tensor<64x64xf16>>
    %v = tt.descriptor_load %d[%c0, %c0] cacheModifier = cg evictionPolicy = evict_last : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32>
    %result = tt.dot %v, %b, %acc : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    tt.return %result : tensor<64x64xf32>
  }
}
