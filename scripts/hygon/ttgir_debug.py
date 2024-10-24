# flake8: noqa: F821,F841
import triton.language.semantic
import itertools
import os
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.common.build import is_hip
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']
if is_hip():
    GPU_DIALECT = "triton_gpu"
    THREADS_PER_WARP = 64
else:
    GPU_DIALECT = "triton_gpu"
    THREADS_PER_WARP = 32
class BlockedLayout:
    def __init__(self, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTAsPerCGA, CTASplitNum, CTAOrder):
        self.sz_per_thread = str(sizePerThread)
        self.threadsPerWarp = str(threadsPerWarp)
        self.warpsPerCTA = str(warpsPerCTA)
        self.order = str(order)
        self.CTAsPerCGA = str(CTAsPerCGA)
        self.CTASplitNum = str(CTASplitNum)
        self.CTAOrder = str(CTAOrder)

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threadsPerWarp}, warpsPerCTA={self.warpsPerCTA}, order={self.order}, CTAsPerCGA={self.CTAsPerCGA}, CTASplitNum={self.CTASplitNum}, CTAOrder={self.CTAOrder}}}>"


class SharedLayout:
    def __init__(self, vec, perPhase, maxPhase, order, CTAsPerCGA, CTASplitNum, CTAOrder):
        self.vec = str(vec)
        self.perPhase = str(perPhase)
        self.maxPhase = str(maxPhase)
        self.order = str(order)
        self.CTAsPerCGA = str(CTAsPerCGA)
        self.CTASplitNum = str(CTASplitNum)
        self.CTAOrder = str(CTAOrder)

    def __str__(self):
        return f"#{GPU_DIALECT}.shared<{{vec={self.vec}, perPhase={self.perPhase}, maxPhase={self.maxPhase}, order={self.order}, CTAsPerCGA={self.CTAsPerCGA}, CTASplitNum={self.CTASplitNum}, CTAOrder={self.CTAOrder}}}>"
def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')

def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")

def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)

blocked = BlockedLayout(sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1])
blocked1 = BlockedLayout(sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0])
blocked2 = BlockedLayout(sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1])
shared = SharedLayout(vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1])
layouts = f"""
#blocked = {blocked} 
#blocked1 = {blocked1}
#blocked2 = {blocked2}
#shared = {shared}
"""
ir = layouts + """
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<16> : tensor<64x16xi32, #blocked> 
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1> 
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked2>
    %c8_i32 = arith.constant 8 : i32 
    %c64_i32 = arith.constant 64 : i32 
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.addi %arg3, %c63_i32 : i32 
    %2 = arith.divsi %1, %c64_i32 : i32 
    %3 = arith.addi %arg4, %c63_i32 : i32 
    %4 = arith.divsi %3, %c64_i32 : i32 
    %5 = arith.muli %4, %c8_i32 : i32 
    %6 = arith.divsi %0, %5 : i32 
    %7 = arith.muli %6, %c8_i32 : i32 
    %8 = arith.subi %2, %7 : i32 
    %9 = arith.minsi %8, %c8_i32 : i32 
    %10 = arith.remsi %0, %9 : i32 
    %11 = arith.addi %7, %10 : i32 
    %12 = arith.remsi %0, %5 : i32 
    %13 = arith.divsi %12, %9 : i32 
    %14 = arith.muli %11, %c64_i32 : i32 
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %21 = tt.splat %14 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %22 = tt.splat %14 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %23 = tt.splat %14 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %24 = arith.addi %21, %15 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %25 = arith.addi %22, %16 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %26 = arith.addi %23, %17 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %27 = arith.muli %13, %c64_i32 : i32 
    %28 = tt.splat %27 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %29 = tt.splat %27 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %30 = tt.splat %27 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %31 = arith.addi %28, %18 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %32 = arith.addi %29, %19 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %33 = arith.addi %30, %20 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> 
    %34 = tt.expand_dims %24 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked> 
    %35 = tt.expand_dims %25 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2> 
    %36 = tt.expand_dims %26 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2> 
    %37 = tt.splat %arg6 : (i32) -> tensor<64x1xi32, #blocked> 
    %38 = arith.muli %34, %37 : tensor<64x1xi32, #blocked> 
    %39 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked> 
    %40 = tt.addptr %39, %38 : tensor<64x1x!tt.ptr<f32, 1>, #blocked>, tensor<64x1xi32, #blocked> 
    %41 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> 
    %42 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> 
    %43 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x16xi32, #blocked> 
    %44 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x16xi32, #blocked> 
    %45 = tt.broadcast %40 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<64x16x!tt.ptr<f32, 1>, #blocked> 
    %46 = tt.broadcast %43 : (tensor<1x16xi32, #blocked>) -> tensor<64x16xi32, #blocked> 
    %47 = tt.addptr %45, %46 : tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<64x16xi32, #blocked> 
    %48 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %49 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> 
    %50 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16x1xi32, #blocked2> 
    %51 = tt.expand_dims %49 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16x1xi32, #blocked2> 
    %52 = tt.splat %arg7 : (i32) -> tensor<16x1xi32, #blocked2> 
    %53 = arith.muli %50, %52 : tensor<16x1xi32, #blocked2> 
    %54 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<16x1x!tt.ptr<f32, 1>, #blocked2> 
    %55 = tt.addptr %54, %53 : tensor<16x1x!tt.ptr<f32, 1>, #blocked2>, tensor<16x1xi32, #blocked2> 
    %56 = tt.expand_dims %31 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2> 
    %57 = tt.expand_dims %32 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2> 
    %58 = tt.expand_dims %33 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2> 
    %59 = tt.broadcast %55 : (tensor<16x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<16x64x!tt.ptr<f32, 1>, #blocked2> 
    %60 = tt.broadcast %56 : (tensor<1x64xi32, #blocked2>) -> tensor<16x64xi32, #blocked2> 
    %61 = tt.addptr %59, %60 : tensor<16x64x!tt.ptr<f32, 1>, #blocked2>, tensor<16x64xi32, #blocked2> 
    %62 = arith.addi %arg5, %c15_i32 : i32 
    %63 = arith.divsi %62, %c16_i32 : i32 
    %64 = arith.muli %arg7, %c16_i32 : i32 
    %65 = tt.splat %64 : (i32) -> tensor<16x64xi32, #blocked2> 
    %66 = arith.muli %c0_i32, %c16_i32 : i32 
    %67 = arith.subi %arg5, %66 : i32 
    %68 = tt.splat %67 : (i32) -> tensor<1x16xi32, #blocked> 
    %69 = "triton_gpu.cmpi"(%44, %68) <{predicate = 2 : i64}> : (tensor<1x16xi32, #blocked>, tensor<1x16xi32, #blocked>) -> tensor<1x16xi1, #blocked> 
    %70 = tt.broadcast %69 : (tensor<1x16xi1, #blocked>) -> tensor<64x16xi1, #blocked> 
    %71 = tt.load %47, %70, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf32, #blocked> 
    %72 = triton_gpu.convert_layout %71 : (tensor<64x16xf32, #blocked>) -> tensor<64x16xf32, #shared> 
    %73 = tt.splat %67 : (i32) -> tensor<16x1xi32, #blocked2> 
    %74 = "triton_gpu.cmpi"(%51, %73) <{predicate = 2 : i64}> : (tensor<16x1xi32, #blocked2>, tensor<16x1xi32, #blocked2>) -> tensor<16x1xi1, #blocked2> 
    %75 = tt.broadcast %74 : (tensor<16x1xi1, #blocked2>) -> tensor<16x64xi1, #blocked2> 
    %76 = tt.load %61, %75, %cst_2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64xf32, #blocked2> 
    %77 = triton_gpu.convert_layout %76 : (tensor<16x64xf32, #blocked2>) -> tensor<16x64xf32, #shared> 
    %78 = tt.addptr %47, %cst : tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<64x16xi32, #blocked> 
    %79 = tt.addptr %61, %65 : tensor<16x64x!tt.ptr<f32, 1>, #blocked2>, tensor<16x64xi32, #blocked2> 
    %80 = arith.subi %63, %c1_i32 : i32 
    %81:7 = scf.for %arg9 = %c0_i32 to %80 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %47, %arg12 = %61, %arg13 = %72, %arg14 = %77, %arg15 = %78, %arg16 = %79) -> (tensor<64x64xf32, #blocked1>, tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<16x64x!tt.ptr<f32, 1>, #blocked2>, tensor<64x16xf32, #shared>, tensor<16x64xf32, #shared>, tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<16x64x!tt.ptr<f32, 1>, #blocked2>)  : i32 {
      %100 = arith.addi %arg9, %c1_i32 : i32 
      %101 = arith.cmpi slt, %100, %80 : i32 
      %102 = arith.muli %100, %c16_i32 : i32 
      %103 = arith.subi %arg5, %102 : i32 
      %104 = tt.splat %103 : (i32) -> tensor<1x16xi32, #blocked> 
      %105 = "triton_gpu.cmpi"(%44, %104) <{predicate = 2 : i64}> : (tensor<1x16xi32, #blocked>, tensor<1x16xi32, #blocked>) -> tensor<1x16xi1, #blocked> 
      %106 = tt.broadcast %105 : (tensor<1x16xi1, #blocked>) -> tensor<64x16xi1, #blocked> 
      %107 = tt.load %arg15, %106, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf32, #blocked> 
      %108 = tt.splat %103 : (i32) -> tensor<16x1xi32, #blocked2> 
      %109 = "triton_gpu.cmpi"(%51, %108) <{predicate = 2 : i64}> : (tensor<16x1xi32, #blocked2>, tensor<16x1xi32, #blocked2>) -> tensor<16x1xi1, #blocked2> 
      %110 = tt.broadcast %109 : (tensor<16x1xi1, #blocked2>) -> tensor<16x64xi1, #blocked2> 
      %111 = tt.load %arg16, %110, %cst_2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64xf32, #blocked2> 
      %112 = triton_gpu.convert_layout %arg13 : (tensor<64x16xf32, #shared>) -> tensor<64x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> 
      %113 = triton_gpu.convert_layout %arg14 : (tensor<16x64xf32, #shared>) -> tensor<16x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> 
      %114 = tt.dot %112, %113, %arg10 {allowTF32 = true} : tensor<64x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x64xf32, #blocked1> 
      %115 = tt.addptr %arg15, %cst : tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<64x16xi32, #blocked> 
      %116 = tt.addptr %arg16, %65 : tensor<16x64x!tt.ptr<f32, 1>, #blocked2>, tensor<16x64xi32, #blocked2> 
      %117 = triton_gpu.convert_layout %107 : (tensor<64x16xf32, #blocked>) -> tensor<64x16xf32, #shared> 
      %118 = triton_gpu.convert_layout %111 : (tensor<16x64xf32, #blocked2>) -> tensor<16x64xf32, #shared> 
      scf.yield %114, %arg11, %arg12, %117, %118, %115, %116 : tensor<64x64xf32, #blocked1>, tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<16x64x!tt.ptr<f32, 1>, #blocked2>, tensor<64x16xf32, #shared>, tensor<16x64xf32, #shared>, tensor<64x16x!tt.ptr<f32, 1>, #blocked>, tensor<16x64x!tt.ptr<f32, 1>, #blocked2> 
    } 
    %82 = triton_gpu.convert_layout %81#3 : (tensor<64x16xf32, #shared>) -> tensor<64x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> 
    %83 = triton_gpu.convert_layout %81#4 : (tensor<16x64xf32, #shared>) -> tensor<16x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> 
    %84 = tt.dot %82, %83, %81#0 {allowTF32 = true} : tensor<64x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x64xf32, #blocked1> 
    %85 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked2> 
    %86 = arith.muli %85, %35 : tensor<64x1xi32, #blocked2> 
    %87 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked2> 
    %88 = tt.addptr %87, %86 : tensor<64x1x!tt.ptr<f32, 1>, #blocked2>, tensor<64x1xi32, #blocked2> 
    %89 = tt.broadcast %88 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked2> 
    %90 = tt.broadcast %57 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2> 
    %91 = tt.addptr %89, %90 : tensor<64x64x!tt.ptr<f32, 1>, #blocked2>, tensor<64x64xi32, #blocked2> 
    %92 = tt.splat %arg3 : (i32) -> tensor<64x1xi32, #blocked2> 
    %93 = "triton_gpu.cmpi"(%36, %92) <{predicate = 2 : i64}> : (tensor<64x1xi32, #blocked2>, tensor<64x1xi32, #blocked2>) -> tensor<64x1xi1, #blocked2> 
    %94 = tt.splat %arg4 : (i32) -> tensor<1x64xi32, #blocked2> 
    %95 = "triton_gpu.cmpi"(%58, %94) <{predicate = 2 : i64}> : (tensor<1x64xi32, #blocked2>, tensor<1x64xi32, #blocked2>) -> tensor<1x64xi1, #blocked2> 
    %96 = tt.broadcast %93 : (tensor<64x1xi1, #blocked2>) -> tensor<64x64xi1, #blocked2> 
    %97 = tt.broadcast %95 : (tensor<1x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked2> 
    %98 = arith.andi %96, %97 : tensor<64x64xi1, #blocked2> 
    %99 = triton_gpu.convert_layout %84 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked2> 
    tt.store %91, %99, %98 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked2> 
    tt.return 
  } 
}
"""
shapeA = (64, 128)
shapeB = (128, 64)
dtype = 'float32'
x_np = numpy_random(shapeA, dtype_str=dtype)
y_np = numpy_random(shapeB, dtype_str=dtype)
x_tri = to_triton(x_np)
y_tri = to_triton(y_np)
z_np = torch.matmul(x_tri, y_tri)
z_tri = torch.zeros((64,64), device='cuda') #torch.empty_like(to_triton(numpy_random((64,64), dtype_str=dtype)))

import tempfile
directory = './'  # 指定目录
filename = 'ir_code.ttgir'  # 指定文件名
file_path = os.path.join(directory, filename)
#with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:

with open(file_path, mode='w') as f:
    f.write(ir)
    f.flush()
    arch_triple = "amdgcn-amd-amdhsa"
    arch_name = "gfx926"
    features = ""
    warp_size = 64
    capabilities = [arch_triple, arch_name, features, warp_size]
    print("print f.name",f.name)
    kernel = triton.compile(f.name, device_type="hip", cc=capabilities)
    print(kernel.asm["amdgcn"])

import triton.language.semantic as sem
if torch.version.hip is not None: #and sem.gpu_matrix_core_version() > 0:
# a,b,c,stride_am,stride_bk,stride_cm
    kernel[(1, 1, 1)](x_tri, y_tri, z_tri, 64, 64, 128, 128, 64, 64)
    print("z_tri",z_tri)
    print("z_np",z_np)


