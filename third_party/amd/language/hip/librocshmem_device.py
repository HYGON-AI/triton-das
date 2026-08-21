################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from triton.language import core
import triton.language as tl
# from triton_dist.language.core import extern_call
import sys
import builtins

from triton.language.core import builtin, tensor
from triton.language import semantic as _semantic

from typing import List

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
pi_i64_t = tl.core.pointer_type(tl.core.dtype("int64"))
pi_i32_t = tl.core.pointer_type(tl.core.dtype("int32"))

# adapted from python/triton/language/core.py
def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _semantic=None):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :param _semantic: the builder
        :return: the return value of the function
    '''
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_types = arg_type_symbol_dict[arg_types][1]
        if not isinstance(ret_types, (List, tuple)):
            ret_types = [ret_types]

        if symbol == "":
            raise ValueError("Symbol can not be empty")
        call = func(lib_name, lib_path, symbol, arg_list, [ret_type.to_ir(_semantic.builder) for ret_type in ret_types],
                    is_pure)

        if len(ret_types) == 0:
            return tensor(call, tl.void)
        if len(ret_types) == 1:
            return tensor(call.get_result(0), ret_types[0])
        return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(ret_types))

@builtin
def extern_call(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _semantic=None):
    '''
        Dispatch an function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :param _semantic: the builder
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if not all_scalar:
        raise ValueError("extern call only support inputs with scalr type")

    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    # func = _semantic.builder.create_extern_call
    func = _semantic.builder.create_extern_call
    # return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, is_pure, _semantic)
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, is_pure, _semantic)


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    elem_dtype_list = tl.core.dtype.SINT_TYPES + tl.core.dtype.UINT_TYPES + tl.core.dtype.FP_TYPES + tl.core.dtype.OTHER_TYPES
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


# apply monkey patch in runtime
patch_hash_method_for_pointer_type()

# ROCSHMEM_CMPS (enum)
ROCSHMEM_CMP_EQ = 0
ROCSHMEM_CMP_NE = 1
ROCSHMEM_CMP_GT = 2
ROCSHMEM_CMP_GE = 3
ROCSHMEM_CMP_LT = 4
ROCSHMEM_CMP_LE = 5

# ROCSHMEM_SIGNAL_OPS (enum)
ROCSHMEM_SIGNAL_SET = 0
ROCSHMEM_SIGNAL_ADD = 1

@core.extern
def set_rocshmem_ctx(ctx, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(ctx, tl.pointer_type(tl.void), _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void),): ("rocshmem_set_rocshmem_ctx", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


void_ptr = core.pointer_type(core.void)


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {(): ("rocshmem_my_pe_wrapper", (tl.int32))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {(): ("rocshmem_n_pes_wrapper", (tl.int32))},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(value, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.int32, tl.int32): ("rocshmem_int_p_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


# @core.extern
# def remote_ptr(local_ptr, pe, _semantic=None):
#     return tl.cast(
#         extern_call(
#             "librocshmem_device",
#             "",
#             [
#                 tl.cast(local_ptr, tl.pointer_type(tl.void), _semantic=_semantic),
#                 tl.cast(pe, tl.int32, _semantic=_semantic)
#             ],
#             {
#                 (tl.pointer_type(tl.void), tl.int32): ("rocshmem_ptr_wrapper", tl.pointer_type(tl.void)),
#             },
#             is_pure=False,
#             _semantic=_semantic,
#         ),
#         local_ptr.dtype,
#         _semantic=_semantic,
#     )

@core.extern
def _remote_ptr_wrapper(local_ptr, pe, _semantic=None):
    return extern_call(
        "",
        "",
        [local_ptr, pe],
        {
            (core.pointer_type(core.void), core.dtype("int32")): (
                "rocshmem_ptr_wrapper",
                core.pointer_type(core.void),  # of the same dtype
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    tl.static_assert(
        local_ptr.dtype.is_ptr(),
        "remote_ptr(local_ptr, pe) local_ptr should be a pointer",
        _semantic=_semantic,
    )
    tl.static_assert(
        pe.dtype.is_int(),
        "remote_ptr(local_ptr, pe) pe should be an integer",
        _semantic=_semantic,
    )
    return tl.cast(
        _remote_ptr_wrapper(
            tl.cast(local_ptr, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            _semantic=_semantic,
        ),
        local_ptr.dtype,
        _semantic=_semantic,
    )

@core.extern
def _getmem_impl(dest,
                 source,
                 nbytes,
                 pe,
                 SCOPE_SUFFIX: core.constexpr,
                 NBI: core.constexpr = core.constexpr(""),
                 _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32):
            (
                f"rocshmem_getmem{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def getmem(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr(""),
                        core.constexpr(""),
                        _semantic=_semantic)


@core.extern
def getmem_wave(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wave"),
                        core.constexpr(""),
                        _semantic=_semantic)



@core.extern
def getmem_wg(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wg"),
                        core.constexpr(""),
                        _semantic=_semantic)


@core.extern
def getmem_nbi(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr(""),
                        core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def getmem_nbi_wave(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wave"),
                        core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def getmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wg"),
                        core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def _putmem_impl(dest,
                 source,
                 nbytes,
                 pe,
                 SCOPE_SUFFIX: core.constexpr,
                 NBI: core.constexpr = core.constexpr(""),
                 _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32):
            (
                f"rocshmem_putmem{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr(""),
                        core.constexpr(""),
                        _semantic=_semantic)


@core.extern
def putmem_wave(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wave"),
                        core.constexpr(""),
                        _semantic=_semantic)


@core.extern
def putmem_wg(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wg"),
                        core.constexpr(""),
                        _semantic=_semantic)


@core.extern
def putmem_nbi(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr(""),
                        core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def putmem_nbi_wave(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wave"),
                        core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def putmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest,
                        source,
                        nbytes,
                        pe,
                        core.constexpr("_wg"),
                        core.constexpr("_nbi"),
                        _semantic=_semantic)

@core.extern
def _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""), _semantic=None):
    tl.static_assert(sig_addr.dtype == tl.pointer_type(tl.int64),
                     "sig_addr should be a pointer of uint64_t",
                     _semantic=_semantic)
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(sig_addr, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.pointer_type(tl.int64), tl.uint64, tl.int32, tl.int32):
            (
                f"rocshmem_putmem_signal{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def putmem_signal(dest,
                  source,
                  nbytes,
                  sig_addr,
                  signal,
                  sig_op,
                  pe,
                  _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr(""),
        core.constexpr(""),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_wg(dest,
                  source,
                  nbytes,
                  sig_addr,
                  signal,
                  sig_op,
                  pe,
                  _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wg"),
        core.constexpr(""),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_wave(dest,
                  source,
                  nbytes,
                  sig_addr,
                  signal,
                  sig_op,
                  pe,
                  _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wave"),
        core.constexpr(""),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi(dest,
                  source,
                  nbytes,
                  sig_addr,
                  signal,
                  sig_op,
                  pe,
                  _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr(""),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )

@core.extern
def putmem_signal_nbi_wg(dest,
                  source,
                  nbytes,
                  sig_addr,
                  signal,
                  sig_op,
                  pe,
                  _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wg"),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi_wave(dest,
                  source,
                  nbytes,
                  sig_addr,
                  signal,
                  sig_op,
                  pe,
                  _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wave"),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )

@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t,
                     "sig_addr should be a pointer of uint64_t",
                     _semantic=_semantic)
    int_ptr = tl.cast(sig_addr, pi_i32_t, _semantic=_semantic)
    return extern_call(
        "librocshmem_device",
        "",
        [
            int_ptr,  # dest (dummy)
            int_ptr,  # source (dummy)
            tl.cast(1, tl.uint64, _semantic=_semantic),
            tl.cast(sig_addr, pi_u64_t, _semantic=_semantic
                    ),  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_i32_t, pi_i32_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "rocshmem_int_put_signal_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t or sig_addr.dtype == pi_i64_t,
                     "sig_addr should be a pointer of uint64_t/int64_t",
                     _semantic=_semantic)
    int_ptr = tl.cast(sig_addr, pi_i32_t, _semantic=_semantic)
    return extern_call(
        "librocshmem_device",
        "",
        [
            int_ptr,
            tl.cast(cmp_, tl.int32, _semantic=_semantic),
            tl.cast(cmp_val, tl.int32, _semantic=_semantic),
        ],  # no cast
        {
            (pi_i32_t, tl.int32, tl.int32): (
                "rocshmem_int_wait_until_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )

# @core.extern
# def wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
#     tl.static_assert(sig_addr.dtype == pi_u64_t or sig_addr.dtype == pi_i64_t,
#                      "sig_addr should be a pointer of uint64_t/int64_t",
#                      _semantic=_semantic)
#     return extern_call(
#         "librocshmem_device",
#         "",
#         [
#             tl.cast(sig_addr, pi_u64_t, _semantic=_semantic),
#             tl.cast(cmp_, tl.int32, _semantic=_semantic),
#             tl.cast(cmp_val, tl.uint64, _semantic=_semantic),
#         ],  # no cast
#         {
#             (pi_u64_t, tl.int32, tl.uint64): (
#                 "rocshmem_wait_until_wrapper",
#                 (),
#             ),
#         },
#         is_pure=False,
#         _semantic=_semantic,
#     )


@core.extern
def _barrier_all_impl(SCOPE_SUFFIX: core.constexpr, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {
            ():
            (f"rocshmem_barrier_all{SCOPE_SUFFIX.value}_wrapper",
             ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def barrier_all(_semantic=None):
    return _barrier_all_impl(core.constexpr(""), _semantic=_semantic)


@core.extern
def barrier_all_wave(_semantic=None):
    return _barrier_all_impl(core.constexpr("_wave"), _semantic=_semantic)


@core.extern
def barrier_all_wg(_semantic=None):
    return _barrier_all_impl(core.constexpr("_wg"), _semantic=_semantic)


@core.extern
def fence(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {
            (): ("rocshmem_fence_wave_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )