// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx942 analyze-small-tensor-ofst=false" | FileCheck %s
// RUN: triton-opt %s --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx950 analyze-small-tensor-ofst=false" | FileCheck %s

// WASP/WDRA captures func pointer args into ttg.warp_specialize partitions.
// ConvertToBufferOps must look through those captures to honor tt.pointer_range.
#blocked_ws = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: ws_capture_pointer_range
  tt.func @ws_capture_pointer_range(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    ttg.warp_specialize(%arg0, %arg1)
    default {
      ttg.warp_yield
    }
    partition0(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) num_warps(4) {
      %offs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked_ws>
      %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked_ws>
      %ptrs = tt.addptr %base, %offs : tensor<256x!tt.ptr<f32>, #blocked_ws>, tensor<256xi32, #blocked_ws>
      // CHECK: buffer_load
      %v = tt.load %ptrs : tensor<256x!tt.ptr<f32>, #blocked_ws>
      %obase = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked_ws>
      %optrs = tt.addptr %obase, %offs : tensor<256x!tt.ptr<f32>, #blocked_ws>, tensor<256xi32, #blocked_ws>
      // CHECK: buffer_store
      tt.store %optrs, %v : tensor<256x!tt.ptr<f32>, #blocked_ws>
      ttg.warp_return
    } : (!tt.ptr<f32>, !tt.ptr<f32>) -> ()
    tt.return
  }
}
