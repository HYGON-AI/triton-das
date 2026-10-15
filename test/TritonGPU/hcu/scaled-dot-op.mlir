// RUN: triton-opt %s --split-input-file --triton-licm --verify-diagnostics | FileCheck %s

// CHECK-LABEL: tt.func @scaled_dot
// CHECK: amdg.scale_write
// CHECK: amdg.dot_scaled
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_dot(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_dot(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %sa: !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    // expected-error @+1 {{requires scales produced directly by amdg.scale_write}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_dot(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    // expected-error @+1 {{requires one direct HCU scaled dot consumer}}
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    ttg.local_dealloc %sa : !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    tt.return %c : tensor<64x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_dot(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x4xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x4xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    // expected-error @+1 {{requires scale shapes}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x4xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x4xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

// expected-error @+1 {{requires a positive rank-two E8M0 Scale Buffer descriptor}}
!bad_space = !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #ttg.shared_memory>
module {}

// -----

// expected-error @+1 {{requires a supported Scale Buffer block and aligned rowStride}}
#bad_stride = #amdg.scale<{blockShape = [32, 8], rowStride = 8}>
module {}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module {
  tt.func @reject_mode(%src: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) {
    // expected-error @+1 {{copy mode must match the Scale Buffer block size}}
    %s = amdg.scale_write %src {mode = 1 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @same_bank(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    // expected-error @+1 {{requires opposite A/B Scale Buffer bank polarities}}
    %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}


// -----

// expected-error @+1 {{requires bankXor = 0 or 1}}
#bad_bank = #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 2}>

// -----

// CHECK-LABEL: tt.func @scale_buffer
// CHECK: scf.for
// CHECK: amdg.dot_scaled
// CHECK: scf.yield
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scale_buffer(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %lb: index, %ub: index, %srcA: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>,
                    %srcB: !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory>) -> tensor<64x64xf32, #mma> {
    %sa = amdg.scale_write %srcA {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory>
    %sb = amdg.scale_write %srcB {mode = 2 : i32} : !ttg.memdesc<64x8xi8, #shared, #ttg.shared_memory> -> !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory>
    %step = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %c) -> (tensor<64x64xf32, #mma>) {
      %d = amdg.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16}>, #amdg.scale_memory> * tensor<128x64xi8, #b>, !ttg.memdesc<64x8xi8, #amdg.scale<{blockShape = [32, 8], rowStride = 16, bankXor = 1}>, #amdg.scale_memory> -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    }
    tt.return %r : tensor<64x64xf32, #mma>
  }
}

// -----

// CHECK-LABEL: tt.func @numeric
// CHECK: tt.dot_scaled
// CHECK: scf.for
// CHECK-NOT: tt.dot_scaled
// CHECK: scf.yield
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 64], isTransposed = false, mmacLayout = 2}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @numeric(%a: tensor<64x128xi8, #a>, %b: tensor<128x64xi8, #b>,
                    %c: tensor<64x64xf32, #mma>,
                    %lb: index, %ub: index, %sa: tensor<64x8xi8, #blocked>,
                    %sb: tensor<64x8xi8, #blocked>) -> tensor<64x64xf32, #mma> {
    %step = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %c) -> (tensor<64x64xf32, #mma>) {
      %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x128xi8, #a>, tensor<64x8xi8, #blocked> * tensor<128x64xi8, #b>, tensor<64x8xi8, #blocked> -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    }
    tt.return %r : tensor<64x64xf32, #mma>
  }
}
