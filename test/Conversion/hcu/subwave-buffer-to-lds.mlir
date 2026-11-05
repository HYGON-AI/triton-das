// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx946 | FileCheck %s

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_1_f32_w4
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_1_f32_w4(%base: !tt.ptr<f32>, %offset: tensor<1xi32, #blocked>, %dst: !ttg.memdesc<1xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<1xi32, #blocked>] -> <1xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_2_f32_w4
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_2_f32_w4(%base: !tt.ptr<f32>, %offset: tensor<2xi32, #blocked>, %dst: !ttg.memdesc<2xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<2xi32, #blocked>] -> <2xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_4_f32_w8
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_4_f32_w8(%base: !tt.ptr<f32>, %offset: tensor<4xi32, #blocked>, %dst: !ttg.memdesc<4xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<4xi32, #blocked>] -> <4xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_16_i32_w4
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_16_i32_w4(%base: !tt.ptr<i32>, %offset: tensor<16xi32, #blocked>, %dst: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <i32>[tensor<16xi32, #blocked>] -> <16xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_32_f32_w8
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_32_f32_w8(%base: !tt.ptr<f32>, %offset: tensor<32xi32, #blocked>, %dst: !ttg.memdesc<32xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<32xi32, #blocked>] -> <32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_64_f32_w4
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_64_f32_w4(%base: !tt.ptr<f32>, %offset: tensor<64xi32, #blocked>, %dst: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @copy_128_f32_w4
  // CHECK: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @copy_128_f32_w4(%base: !tt.ptr<f32>, %offset: tensor<128xi32, #blocked>, %dst: !ttg.memdesc<128xf32, #shared, #smem, mutable>) {
    %token = amdg.buffer_load_to_local %base[%offset] into %dst : <f32>[tensor<128xi32, #blocked>] -> <128xf32, #shared, #smem, mutable>
    tt.return
  }
}
