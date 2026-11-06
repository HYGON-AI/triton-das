// RUN: triton-opt %s -triton-hcu-convert-warp-specialize-to-llvm='arch=gfx946 wdra-enabled=false' | FileCheck %s

// A loaded value used only by partition 0 must stay in registers. The empty
// consumer arguments must not force allocation.offset or LDS communication.
// CHECK-NOT: llvm.store
// CHECK-NOT: !llvm.ptr<3>
// CHECK: %[[VALUE:.*]] = llvm.load
// CHECK: llvm.store %[[VALUE]],
// CHECK-NOT: llvm.load
// CHECK-NOT: !llvm.ptr<3>
// CHECK-NOT: ttg.warp_specialize
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32,
                   "ttg.threads-per-warp" = 64 : i32,
                   "ttg.total-num-warps" = 12 : i32, ttg.target = "hip:gfx946"} {
  llvm.func @producer_capture(%input: !llvm.ptr<1>, %output: !llvm.ptr<1>) {
    %value = llvm.load %input : !llvm.ptr<1> -> i32
    ttg.warp_specialize(%value, %output) attributes {warpGroupStartIds = array<i32: 0, 4, 8>}
    default { ttg.warp_yield }
    partition0(%v: i32, %out: !llvm.ptr<1>) num_warps(4) {
      llvm.store %v, %out : i32, !llvm.ptr<1>
      ttg.warp_return
    }
    partition1(%unused: i32, %unused_ptr: !llvm.ptr<1>) num_warps(4) { ttg.warp_return }
    partition2(%unused: i32, %unused_ptr: !llvm.ptr<1>) num_warps(4) { ttg.warp_return }
    : (i32, !llvm.ptr<1>) -> ()
    llvm.return
  }
}
