// RUN: triton-opt %s --split-input-file --verify-diagnostics --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=2" | FileCheck %s --check-prefix=EARLY
// RUN: triton-opt %s --split-input-file --verify-diagnostics --tritonhcu-accelerate-matmul="arch-generation-name=gfx946 mmac-layout-force=2" --tritonhcu-scale-pipeline | FileCheck %s --check-prefix=LATE
#blocked = #ttg.blocked<{sizePerThread=[1,2],threadsPerWarp=[16,4],warpsPerCTA=[4,1],order=[1,0]}>
module attributes {"ttg.num-warps"=4:i32,"ttg.threads-per-warp"=64:i32,"ttg.num-ctas"=1:i32,ttg.target="hip:gfx946"} {
  // EARLY-LABEL: tt.func @prepare
  // EARLY-NOT: ttg.local_alloc
  // EARLY-NOT: ttg.local_load
  // EARLY-NOT: amdg.scale_write
  // EARLY-NOT: amdg.dot_scaled
  // EARLY: tt.dot_scaled
  // EARLY: tt.return
  // LATE-LABEL: tt.func @prepare
  // LATE-COUNT-2: amdg.scale_write
  // LATE-NOT: tt.dot_scaled
  // LATE: amdg.dot_scaled
  // LATE: tt.return
  tt.func @prepare(%a:tensor<64x128xi8,#blocked>,%b:tensor<128x64xi8,#blocked>,
                   %c:tensor<64x64xf32,#blocked>,
                   %pa:tensor<64x8x!tt.ptr<i8>,#blocked>,%pb:tensor<64x8x!tt.ptr<i8>,#blocked>) -> tensor<64x64xf32,#blocked> {
    %sa = tt.load %pa : tensor<64x8x!tt.ptr<i8>,#blocked>
    %sb = tt.load %pb : tensor<64x8x!tt.ptr<i8>,#blocked>
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs=e2m1 rhs=e2m1 {fastMath=false}
      : tensor<64x128xi8,#blocked>,tensor<64x8xi8,#blocked> * tensor<128x64xi8,#blocked>,tensor<64x8xi8,#blocked> -> tensor<64x64xf32,#blocked>
    tt.return %d : tensor<64x64xf32,#blocked>
  }
}
// -----
#mma = #ttg.amd_mfma<{version=4,warpsPerCTA=[2,2],instrShape=[16,16,64],isTransposed=false,mmacLayout=2}>
#a = #ttg.dot_op<{opIdx=0,parent=#mma,kWidth=8}>
#b = #ttg.dot_op<{opIdx=1,parent=#mma,kWidth=8}>
#blocked = #ttg.blocked<{sizePerThread=[1,2],threadsPerWarp=[16,4],warpsPerCTA=[4,1],order=[1,0]}>
#shared = #ttg.swizzled_shared<{vec=4,perPhase=1,maxPhase=1,order=[1,0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps"=4:i32,"ttg.threads-per-warp"=64:i32,"ttg.num-ctas"=1:i32,ttg.target="hip:gfx946"} {
  // EARLY-LABEL: tt.func @both_branches
  // EARLY-NOT: amdg.scale_write
  // EARLY: tt.dot_scaled
  // LATE-LABEL: tt.func @both_branches
  // LATE-SAME: %[[COND:[^:]+]]: i1
  // LATE-DAG: amdg.scale_write %{{.*}} pred %[[COND]]
  // LATE-DAG: amdg.scale_write %{{.*}} pred %[[COND]]
  // LATE-DAG: %[[NOT0:.*]] = arith.xori %[[COND]], %{{.*}}
  // LATE-DAG: amdg.scale_write %{{.*}} pred %[[NOT0]]
  // LATE-DAG: %[[NOT1:.*]] = arith.xori %[[COND]], %{{.*}}
  // LATE-DAG: amdg.scale_write %{{.*}} pred %[[NOT1]]
  // LATE: scf.if
  // LATE: amdg.dot_scaled
  // LATE: } else {
  // LATE: amdg.dot_scaled
  tt.func @both_branches(%cond:i1,%a:tensor<64x128xi8,#a>,%b:tensor<128x64xi8,#b>,
                         %c:tensor<64x64xf32,#mma>,%lds:!ttg.memdesc<64x8xi8,#shared,#smem,mutable>) -> tensor<64x64xf32,#mma> {
    %scale = ttg.local_load %lds : !ttg.memdesc<64x8xi8,#shared,#smem,mutable> -> tensor<64x8xi8,#blocked>
    %d = scf.if %cond -> tensor<64x64xf32,#mma> {
      %x = tt.dot_scaled %a scale %scale, %b scale %scale, %c lhs=e2m1 rhs=e2m1 {fastMath=false}
        : tensor<64x128xi8,#a>,tensor<64x8xi8,#blocked> * tensor<128x64xi8,#b>,tensor<64x8xi8,#blocked> -> tensor<64x64xf32,#mma>
      scf.yield %x : tensor<64x64xf32,#mma>
    } else {
      %x = tt.dot_scaled %a scale %scale, %b scale %scale, %c lhs=e2m1 rhs=e2m1 {fastMath=false}
        : tensor<64x128xi8,#a>,tensor<64x8xi8,#blocked> * tensor<128x64xi8,#b>,tensor<64x8xi8,#blocked> -> tensor<64x64xf32,#mma>
      scf.yield %x : tensor<64x64xf32,#mma>
    }
    tt.return %d : tensor<64x64xf32,#mma>
  }
}

// -----
// No scaled dot: this pass must not recompute unrelated wait counts.
module {
  // EARLY-LABEL: tt.func @ordinary_wait
  // EARLY: ttg.async_wait {{.*}}num = 7
  // LATE-LABEL: tt.func @ordinary_wait
  // LATE: ttg.async_wait {{.*}}num = 7
  tt.func @ordinary_wait(%token: !ttg.async.token) {
    %ready = ttg.async_wait %token {num = 7 : i32}
    tt.return
  }
}

// -----

// The public packing semantics remain valid. Reject unsupported target lowering
// before constructing local_load_packed_transposed with no gfx946 conversion.
#blocked = #ttg.blocked<{sizePerThread=[1,2],threadsPerWarp=[16,4],warpsPerCTA=[4,1],order=[1,0]}>
module attributes {"ttg.num-warps"=4:i32,"ttg.threads-per-warp"=64:i32,"ttg.num-ctas"=1:i32,ttg.target="hip:gfx946"} {
  tt.func @plain_m_pack(%a:tensor<32x128xi8,#blocked>,%b:tensor<64x64xi8,#blocked>,
                         %c:tensor<64x64xf32,#blocked>,%s:tensor<64x4xi8,#blocked>) -> tensor<64x64xf32,#blocked> {
    // expected-error @+1 {{M/N-packed FP4 requires a supported MLS operand; plain-load packed-transpose lowering is unavailable}}
    %d = tt.dot_scaled %a scale %s, %b scale %s, %c lhs=e2m1 rhs=e2m1 {fastMath=false,lhs_k_pack=false}
      : tensor<32x128xi8,#blocked>,tensor<64x4xi8,#blocked> * tensor<64x64xi8,#blocked>,tensor<64x4xi8,#blocked> -> tensor<64x64xf32,#blocked>
    tt.return %d : tensor<64x64xf32,#blocked>
  }
  tt.func @plain_n_pack(%a:tensor<64x64xi8,#blocked>,%b:tensor<128x32xi8,#blocked>,
                         %c:tensor<64x64xf32,#blocked>,%s:tensor<64x4xi8,#blocked>) -> tensor<64x64xf32,#blocked> {
    // expected-error @+1 {{M/N-packed FP4 requires a supported MLS operand; plain-load packed-transpose lowering is unavailable}}
    %d = tt.dot_scaled %a scale %s, %b scale %s, %c lhs=e2m1 rhs=e2m1 {fastMath=false,rhs_k_pack=false}
      : tensor<64x64xi8,#blocked>,tensor<64x4xi8,#blocked> * tensor<128x32xi8,#blocked>,tensor<64x4xi8,#blocked> -> tensor<64x64xf32,#blocked>
    tt.return %d : tensor<64x64xf32,#blocked>
  }
}
