"""Experimental HCU matrix transfer and explicit pipeline primitives."""

from dataclasses import dataclass, fields
from triton.language.core import constexpr_type
from .._layouts import MmacLayout
from triton.experimental.gluon.language._layouts import (
    SharedLayout, DistributedLayout, SharedLinearLayout, DistributedLinearLayout,
)
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.experimental.gluon import language as gl


@dataclass(frozen=True)
class MlsSharedLayout(SharedLayout):
    """HCU MLS shared encoding. Native instruction legality is checked in lowering."""

    op_idx: int
    tile: tuple
    bits: int = 8
    kind: int = 0
    alt: int = 0
    version: int = 3
    order: tuple = (1, 0)

    def __post_init__(self):
        for name in ("op_idx", "bits", "kind", "alt", "version"):
            object.__setattr__(self, name, _unwrap_if_constexpr(getattr(self, name)))
        object.__setattr__(self, "tile", tuple(_unwrap_if_constexpr(self.tile)))
        object.__setattr__(self, "order", tuple(_unwrap_if_constexpr(self.order)))

        if self.op_idx not in (0, 1) or len(self.tile) != 2 or any(x <= 0 for x in self.tile):
            raise ValueError("MLS requires operand 0/1 and a positive rank-two tile")
        if sorted(self.order) != [0, 1]:
            raise ValueError("MLS order must be a permutation of (0, 1)")

    @property
    def cga_layout(self):
        return []

    def _to_ir(self, builder):
        return builder.get_hcu_mls_shared_layout(
            self.op_idx,
            list(self.tile),
            self.bits,
            self.kind,
            self.alt,
            self.version,
            list(self.order),
        )

    def mangle(self):
        return "HCUMLS_" + str(
            (self.op_idx, self.tile, self.bits, self.kind, self.alt, self.version, self.order)
        ).replace(" ", "")


@builtin
def set_priority(priority, _semantic=None):
    """Set the current wave's scheduling priority (0..3); no synchronization."""
    priority = _unwrap_if_constexpr(priority)
    if type(priority) is not int or not 0 <= priority <= 3:
        raise ValueError("priority must be a compile-time integer in [0, 3]")
    _semantic.builder.create_hcu_set_priority(priority)


@builtin
def wait_lgkmcnt(count=0, _semantic=None):
    """Wait until this wave's LGKM counter is at most count (0..15).

    VMEM/export counters are left unconstrained. This is not a CTA rendezvous.
    The existing target memory-counter lowering encodes the native wait.
    """
    count = _unwrap_if_constexpr(count)
    if type(count) is not int or not 0 <= count <= 15:
        raise ValueError("lgkmcnt must be a compile-time integer in [0, 15]")
    _semantic.builder.create_hcu_wait_lgkmcnt(count)


@builtin
def get_wave_id(_semantic=None):
    """Return the HCU CTA-wide hardware wave index as a scalar int32.

    The value is uniform within a wave. In warp-specialized kernels it includes
    the partition's starting wave offset; it is not a partition-local index.
    """
    return gl.tensor(_semantic.builder.create_hcu_wave_id(), gl.int32)


@builtin
def matrix_load(smem, base, shape, strides, indices, warps=(1, 4), pred=True, boundary_check=(), _semantic=None):
    """Issue a native MLS transfer; caller owns readiness tracking.

    The global base must be 16-byte aligned. boundary_check selects dimensions
    checked against shape; unchecked dimensions must cover the shared tile.
    """
    if not isinstance(smem.layout, MlsSharedLayout):
        raise ValueError("matrix_load requires MlsSharedLayout")

    def values(xs, dtype):
        return [
            _semantic.cast(_semantic.to_tensor(x), dtype).handle for x in _unwrap_if_constexpr(xs)
        ]

    boundary_check = _semantic._canonicalize_boundary_check(_unwrap_if_constexpr(boundary_check), smem.shape)
    _semantic.builder.create_hcu_mls_load_to_shared(
        smem.handle,
        base.handle,
        values(shape, gl.int32),
        values(strides, gl.int64),
        values(indices, gl.int32),
        list(_unwrap_if_constexpr(warps)),
        _semantic.cast(_semantic.to_tensor(pred), gl.int1).handle,
        boundary_check,
    )


@dataclass(frozen=True)
class MmacInterleave2Layout(DistributedLayout):
    """MLS interleave2 B or its matching LTS result register layout.

    Wrap DotOperandLayout(1, F, k_width) for B, or F for the accumulator/result.
    A keeps its ordinary DotOperandLayout. Dtypes and instruction geometry
    are checked by the target instruction selector, not by this layout.
    This is a data mapping; it does not change the MMAC instruction modifiers.
    """

    parent: DistributedLayout

    def __post_init__(self):
        parent = _unwrap_if_constexpr(self.parent)
        object.__setattr__(self, "parent", parent)
        f = parent
        if isinstance(parent, gl.DotOperandLayout):
            if parent.operand_index != 1:
                raise ValueError("interleave2 supports only B")
            f = parent.parent
        if ((not isinstance(f, gl.amd.AMDMFMALayout) or f.mmac_layout != MmacLayout.LTS)
                or f.rank != 2 or (f.instr_shape[1] * f.tiles_per_warp[1]) % 32):
            raise ValueError("interleave2 requires rank-2 LTS with complete 32-column N groups")

    @property
    def rank(self):
        return self.parent.rank

    @property
    def cga_layout(self):
        return self.parent.cga_layout

    def _to_ir(self, builder):
        return builder.get_hcu_mmac_interleave2_layout(self.parent._to_ir(builder))

    def mangle(self):
        return "HCUInterleave2_" + self.parent.mangle()


@builtin
def get_mmac_instruction(a_dtype, b_dtype, acc_dtype, max_k, tile=(16, 16), version=4, mmac_layout=3, _semantic=None):
    """Query the target MMAC database; return (instruction_shape, k_width).

    No input conversion is performed. Unsupported dtype/geometry combinations
    fail in the same native instruction selector used by MMAC lowering.
    """
    a_dtype, b_dtype, acc_dtype, max_k, tile = map(
        _unwrap_if_constexpr, (a_dtype, b_dtype, acc_dtype, max_k, tile))
    result = _semantic.builder.get_hcu_mmac_instruction(
        a_dtype.to_ir(_semantic.builder), b_dtype.to_ir(_semantic.builder),
        acc_dtype.to_ir(_semantic.builder), int(max_k), list(tile),
        int(_unwrap_if_constexpr(version)), int(_unwrap_if_constexpr(mmac_layout)),
        _semantic.builder.options.arch)
    return gl.constexpr((tuple(result[0]), result[1]))


def _check_ebarrier_arguments(id, count, allow_zero=False):
    id = _unwrap_if_constexpr(id)
    count = _unwrap_if_constexpr(count)
    if not isinstance(id, gl.tensor) and (type(id) is not int or not 0 <= id < 16):
        raise ValueError("ebarrier ID must be an integer in [0, 16) or a wave-uniform tensor")
    minimum = 0 if allow_zero else 1
    if type(count) is not int or not minimum <= count <= 0x7fffffff:
        raise ValueError(f"ebarrier count must be a compile-time integer in [{minimum}, 2147483647]")
    return id, count


@builtin
def ebarrier_arrive(id, count, _semantic=None):
    """Arrive without waiting; count is the rendezvous wave count, not an increment.

    The caller owns wave-uniform barrier IDs and participation. Count must be
    positive. Outstanding memory operations require separate waits.
    """
    id, count = _check_ebarrier_arguments(id, count)
    _semantic.builder.create_hcu_ebarrier_arrive(
        _semantic.cast(_semantic.to_tensor(id), gl.int32).handle, count
    )


@builtin
def ebarrier_sync(id, count, _semantic=None):
    """Arrive and wait for the rendezvous wave count (zero selects the whole CTA).

    The caller owns wave-uniform barrier IDs and participation. This is not a memory drain.
    """
    id, count = _check_ebarrier_arguments(id, count, allow_zero=True)
    _semantic.builder.create_hcu_ebarrier_sync(
        _semantic.cast(_semantic.to_tensor(id), gl.int32).handle, count
    )


@builtin
def abarrier_init(id, count, _semantic=None):
    """Initialize a barrier after its previous lifetime has retired."""
    id = _unwrap_if_constexpr(id)
    count = _unwrap_if_constexpr(count)
    if type(id) is not int or not 0 <= id < 16:
        raise ValueError("abarrier ID must be a compile-time integer in [0, 16)")
    if type(count) is not int or not 0 < count <= 0x7fffffff:
        raise ValueError("abarrier count must be a compile-time positive signed 32-bit integer")
    _semantic.builder.create_hcu_abarrier_init(id, count)


@builtin
def abarrier_invalidate(id, _semantic=None):
    """Retire a barrier after final use; this operation does not drain memory."""
    id = _unwrap_if_constexpr(id)
    if type(id) is not int or not 0 <= id < 16:
        raise ValueError("abarrier ID must be a compile-time integer in [0, 16)")
    _semantic.builder.create_hcu_abarrier_invalidate(id)


@builtin
def abarrier_try_wait(id, phase, _semantic=None):
    """Wait on a barrier phase using the native suspending try-wait instruction.

    Return the native int32 completion status. The wave may suspend while the
    requested phase is pending. Barrier ID and phase must be wave-uniform scalar
    values; use abarrier_test_wait for a nonblocking completion probe.
    """
    handle = _semantic.builder.create_hcu_abarrier_try_wait(
        _semantic.cast(_semantic.to_tensor(id), gl.int32).handle,
        _semantic.cast(_semantic.to_tensor(phase), gl.int32).handle,
    )
    return gl.tensor(handle, gl.int32)


@builtin
def abarrier_test_wait(id, phase, _semantic=None):
    """Test an initialized barrier phase without suspending the wave.

    Return the native int32 completion status: zero means pending, nonzero means
    complete. Barrier ID and phase must be wave-uniform scalar values.
    """
    handle = _semantic.builder.create_hcu_abarrier_test_wait(
        _semantic.cast(_semantic.to_tensor(id), gl.int32).handle,
        _semantic.cast(_semantic.to_tensor(phase), gl.int32).handle,
    )
    return gl.tensor(handle, gl.int32)


@builtin
def abarrier_seq(id, _semantic=None):
    """Associate subsequent native transfers with the initialized abarrier."""
    _semantic.builder.create_hcu_abarrier_seq(
        _semantic.cast(_semantic.to_tensor(id), gl.int32).handle
    )


@builtin
def abarrier_arrive(id, count=1, _semantic=None):
    """Contribute count wave arrivals to an initialized native abarrier."""
    count = _unwrap_if_constexpr(count)
    if type(count) is not int or not 0 < count <= 0x7fffffff:
        raise ValueError("abarrier count must be a compile-time positive signed 32-bit integer")
    _semantic.builder.create_hcu_abarrier_arrive(
        _semantic.cast(_semantic.to_tensor(id), gl.int32).handle, count
    )


@dataclass(frozen=True)
class BufferLdsSharedLayout(SharedLayout):
    """Physical buffer-to-LDS placement, including producer waves and M0 rotation.

    linear_component maps unwrapped offsets; wrap_count is a wave period.
    row_bit_swap is zero or a power-of-two mask: 4 swaps row bits 2 and 3.
    The linear component must include the same row permutation.
    waves_per_row_group=0 groups all waves; transfer_major uses transfer/wave/lane
    slot order. Consumers must preserve this full placement, not only its linear part.
    """

    linear_component: SharedLinearLayout
    element_bytes: int
    contiguous_dim: int
    rows: int
    row_bytes: int
    waves: int
    copy_bytes_per_lane: int
    rows_per_chunk: int
    wrap_count: int
    wrap_step_bytes: int
    waves_per_row_group: int = 0
    transfer_major: bool = False
    row_bit_swap: int = 0

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _unwrap_if_constexpr(getattr(self, field.name)))
        if not isinstance(self.linear_component, SharedLinearLayout):
            raise ValueError("linear_component must be a SharedLinearLayout")
        if any(len(basis) != 2 for basis in self.linear_component.offset_bases):
            raise ValueError("BufferLdsSharedLayout requires a rank-two linear component")
        if any(len(basis) != 2 for basis in self.linear_component.block_bases):
            raise ValueError("BufferLdsSharedLayout requires rank-two block bases")
        for field in fields(self):
            if field.name in ("linear_component", "transfer_major"):
                continue
            value = getattr(self, field.name)
            if type(value) is not int or not 0 <= value <= 0xffffffff:
                raise ValueError(f"{field.name} must be a compile-time unsigned 32-bit integer")
        if type(self.transfer_major) is not bool:
            raise ValueError("transfer_major must be a compile-time bool")

    def _to_ir(self, builder):
        # Validate before the generic SharedLinearLayout constructor can abort.
        self.producer_layout()
        return builder.get_hcu_buffer_lds_shared_layout(self.linear_component._to_ir(builder), self.element_bytes,
                                                     self.contiguous_dim, self.rows, self.row_bytes, self.waves,
                                                     self.copy_bytes_per_lane, self.rows_per_chunk, self.wrap_count,
                                                     self.wrap_step_bytes, self.waves_per_row_group,
                                                     self.transfer_major, self.row_bit_swap)

    def mangle(self):
        values = "_".join(str(getattr(self, field.name)) for field in fields(self) if field.name != "linear_component")
        return f"HCUBufferLds_{self.linear_component.mangle()}_{values}"

    def producer_layout(self):
        """Register/lane/wave ownership required by this explicit placement.

        This derives ownership from the declared transfer geometry. It does
        not select a copy width, rotation, or pipeline configuration.
        """
        # Query the same implementation used by buffer-copy validation and
        # LLVM lowering; do not duplicate producer/M0 geometry in Python.
        from triton._C.libtriton import hcu
        component = self.linear_component
        return DistributedLinearLayout(*hcu.get_buffer_lds_producer_layout(
            component.offset_bases, component.block_bases, component.alignment, self.element_bytes, self.contiguous_dim,
            self.rows, self.row_bytes, self.waves, self.copy_bytes_per_lane, self.rows_per_chunk, self.wrap_count,
            self.wrap_step_bytes, self.waves_per_row_group, self.transfer_major, self.row_bit_swap))


@builtin
def matrix_store(
    base, value, shape, strides, indices, tile=(32, 32),
    interleave=1, version=3, order=(1, 0), boundary_check=(), _semantic=None
):
    """Store a rank-two tensor through a native HCU matrix-store instruction.

    Values are converted to the destination's FP16/BF16/I16/U16 element type
    and to an explicit native LTS/interleave2 register layout as needed. The
    native selector defines legal tile/version/order/interleave combinations;
    unsupported combinations raise an error, with no scalar-store fallback.

    shape and indices are global element extents/origins. Strides are in
    elements; the output must have unit column stride. boundary_check selects
    dimensions checked against shape. Unchecked dimensions must be in bounds.
    The value tile must contain complete native instruction tiles. Multiple
    kernel CTAs may use independent indices; clustered CTA layouts are not
    supported. Redundant-wave handling remains pending in lowering.
    Conversion may introduce shared-memory traffic/synchronization.
    """
    from triton.experimental.gluon.language._layouts import warps_per_cta

    tile = tuple(_unwrap_if_constexpr(tile))
    order = tuple(_unwrap_if_constexpr(order))
    shape, strides, indices = map(_unwrap_if_constexpr, (shape, strides, indices))
    version = _unwrap_if_constexpr(version)
    interleave = _unwrap_if_constexpr(interleave)
    if len(tile) != 2 or any(type(x) is not int or x <= 0 for x in tile):
        raise ValueError("matrix_store requires a positive rank-two tile")
    if sorted(order) != [0, 1]:
        raise ValueError("matrix_store order must be a permutation of (0, 1)")
    if len(shape) != 2 or len(strides) != 2 or len(indices) != 2 or len(value.shape) != 2:
        raise ValueError("matrix_store requires rank-two value, shape, strides and indices")
    if not isinstance(base.dtype, gl.pointer_type) or base.dtype.address_space != 1:
        raise ValueError("matrix_store requires a global-memory base pointer")
    dtype = base.dtype.element_ty
    if not isinstance(strides[1], gl.tensor) and _unwrap_if_constexpr(strides[1]) != 1:
        raise ValueError("matrix_store requires unit column stride")
    # Keep instruction capability decisions in the backend selector rather
    # than duplicating a tile, version or element-type whitelist here.
    _semantic.builder.check_hcu_mls_store_configuration(
        list(tile), dtype.primitive_bitwidth, int(interleave), int(version), list(order))
    if any(int(value.shape[i]) % tile[i] for i in range(2)):
        raise ValueError("matrix_store value shape must contain complete instruction tiles")

    source_layout = value.type.layout
    layout = source_layout.parent if isinstance(source_layout, MmacInterleave2Layout) else source_layout
    if getattr(layout, "cga_layout", []):
        raise ValueError("matrix_store does not support clustered CTA layouts")
    waves = list(warps_per_cta(layout, value.shape))
    if len(waves) != 2:
        raise ValueError("matrix_store requires a rank-two warp distribution")
    # A whole native instruction must belong to one warp.
    native = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], mmac_layout=MmacLayout.LTS,
        warps_per_cta=waves, element_bitwidth=32,
        tiles_per_warp=[max(tile[0] // 16, int(value.shape[0]) // (16 * waves[0])),
                        max(2, int(value.shape[1]) // (16 * waves[1]))],
    )
    native_layout = MmacInterleave2Layout(native)
    value = _semantic.cast(value, dtype)
    value = _semantic.convert_layout(value, native_layout)

    def values(xs, dtype):
        return [_semantic.cast(_semantic.to_tensor(x), dtype).handle for x in xs]

    boundary_check = _semantic._canonicalize_boundary_check(_unwrap_if_constexpr(boundary_check), value.shape)
    _semantic.builder.create_hcu_mls_store_from_registers(
        base.handle,
        value.handle,
        values(shape, gl.int32),
        values(strides, gl.int64),
        values(indices, gl.int32),
        list(_unwrap_if_constexpr(tile)),
        int(_unwrap_if_constexpr(interleave)),
        int(_unwrap_if_constexpr(version)),
        list(order),
        boundary_check,
    )


@builtin
def sched_barrier(mask=0, _semantic=None):
    _semantic.builder.create_hcu_sched_barrier(int(_unwrap_if_constexpr(mask)))


@builtin
def sched_group_barrier(mask, count, sync=0, _semantic=None):
    _semantic.builder.create_hcu_sched_group_barrier(
        int(_unwrap_if_constexpr(mask)),
        int(_unwrap_if_constexpr(count)),
        int(_unwrap_if_constexpr(sync)),
    )


@builtin
def packed_scale_fma(dot, a_scale, b_scale, accumulator, _semantic=None):
    """Compute fma(dot * a_scale, b_scale, accumulator) in packed FP32 pairs.

    b_scale is scalar within each thread. The multiply rounds before the fused
    multiply-add. Each thread must own an even number (at most 64) of elements.
    """
    dot, a_scale, accumulator = _semantic.broadcast_tensors(dot, a_scale, accumulator)
    b_scale = _semantic.to_tensor(b_scale)
    if any(x.dtype != gl.float32 for x in (dot, a_scale, b_scale, accumulator)):
        raise ValueError("packed_scale_fma requires FP32 operands")
    if b_scale.type.is_block():
        raise ValueError("b_scale must be a scalar per thread")
    if dot.type != a_scale.type or dot.type != accumulator.type:
        raise ValueError("dot, a_scale and accumulator must have the same layout")
    handle = _semantic.builder.create_hcu_packed_fp32_scale_fma(
        dot.handle, a_scale.handle, b_scale.handle, accumulator.handle
    )
    return gl.tensor(handle, dot.type)


@dataclass(frozen=True)
class ScaleLayout:
    """E8M0 coordinates in Scale Buffer storage, not LDS/register layout.

    block_shape is one ds2buf tile; row_stride is in hardware rows.
    A/B require opposite bank_xor. The native verifier checks geometry.
    """

    block_shape: tuple
    row_stride: int
    bank_xor: int = 0

    def __post_init__(self):
        shape = _unwrap_if_constexpr(self.block_shape)
        shape = tuple(_unwrap_if_constexpr(dim) for dim in shape)
        object.__setattr__(self, "block_shape", shape)
        for name in ("row_stride", "bank_xor"):
            object.__setattr__(self, name, _unwrap_if_constexpr(getattr(self, name)))
        if len(shape) != 2 or any(type(dim) is not int for dim in shape):
            raise ValueError("ScaleLayout requires two compile-time integer block dimensions")
        if type(self.row_stride) is not int or type(self.bank_xor) is not int:
            raise ValueError("ScaleLayout row_stride and bank_xor must be compile-time integers")

    @property
    def type(self):
        return constexpr_type(self)

    def _to_ir(self, builder):
        return builder.get_hcu_scale_layout(self.block_shape, self.row_stride, self.bank_xor)

    def mangle(self):
        return f"HCUScale_{self.block_shape[0]}_{self.block_shape[1]}_{self.row_stride}_{self.bank_xor}"


class scale_buffer_descriptor_type(gl.base_type):
    """Read-only storage reference produced by scale_write, not numeric data."""

    def __init__(self, shape, layout):
        self.shape = tuple(shape)
        self.layout = layout
        if len(self.shape) != 2 or not isinstance(layout, ScaleLayout):
            raise ValueError("Scale Buffer requires a rank-two shape and ScaleLayout")

    def to_ir(self, builder):
        return builder.get_hcu_scale_mem_desc_ty(self.shape, self.layout._to_ir(builder))

    def _flatten_ir_types(self, builder, out):
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        return scale_buffer_descriptor(handles[cursor], self.shape, self.layout), cursor + 1

    def __eq__(self, other):
        return type(self) is type(other) and self.shape == other.shape and self.layout == other.layout

    def __str__(self):
        return f"scale_buffer_descriptor<{self.shape}, {self.layout}>"

    def mangle(self):
        return f"HCUSD_{self.shape[0]}_{self.shape[1]}_{self.layout.mangle()}"


class scale_buffer_descriptor(gl.base_value):
    """Scale storage for one direct dot_scaled use in the same region.

    No numeric loads/views or cross-loop/JIT passing; pass LDS and write at consumption.
    """

    def __init__(self, handle, shape, layout):
        self.handle = handle
        self.type = scale_buffer_descriptor_type(shape, layout)

    def _flatten_ir(self, handles):
        handles.append(self.handle)

    @property
    def dtype(self):
        return gl.uint8

    @property
    def shape(self):
        return self.type.shape

    @property
    def layout(self):
        return self.type.layout

    @property
    def rank(self):
        return len(self.type.shape)


@builtin
def scale_write(src, layout, pred=True, _semantic=None):
    """Copy ready LDS E8M0 to explicit Scale Buffer storage.

    The logical M/N dimension must contain at least 16 rows.

    The result is storage, not completion. Complete LDS copies before calling
    this operation and publish before dot_scaled. A false predicate leaves no valid scale values.
    """
    layout = _unwrap_if_constexpr(layout)
    if not isinstance(src, gl.shared_memory_descriptor) or src.rank != 2 or src.dtype != gl.uint8:
        raise ValueError("scale_write requires a rank-two uint8 E8M0 LDS descriptor")
    if src.shape[0] < 16:
        raise ValueError("scale_write requires at least 16 logical M/N rows")
    if not isinstance(layout, ScaleLayout):
        raise ValueError("scale_write requires a ScaleLayout")
    pred = _semantic.to_tensor(pred)
    if pred.type != gl.int1:
        raise ValueError("scale_write pred must be a wave-uniform scalar bool")
    result = scale_buffer_descriptor_type(src.type.shape, layout)
    result_ty = result.to_ir(_semantic.builder)
    area = layout.block_shape[0] * layout.block_shape[1]
    mode = {64: 0, 128: 1, 256: 2}[area]  # Geometry was checked by the native layout verifier.
    handle = _semantic.builder.create_hcu_scale_write(result_ty, src.handle, pred.handle, mode)
    return scale_buffer_descriptor(handle, result.shape, layout)


@builtin
def dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, lhs_k_pack=True, rhs_k_pack=True, _semantic=None):
    """Scaled MMAC with explicit operands, layouts and Scale Buffer descriptors.

    Each E8M0 scale covers 32 logical K elements, including packed FP4.
    Caller owns operand delivery, pipeline and publication.
    """
    if not isinstance(a_scale, scale_buffer_descriptor) or not isinstance(b_scale, scale_buffer_descriptor):
        raise ValueError("dot_scaled requires two Scale Buffer descriptors")
    if acc.dtype != gl.float32:
        raise ValueError("the current HCU scaled MMAC path requires an FP32 accumulator")
    if not isinstance(acc.type.layout, gl.amd.AMDMFMALayout) or acc.type.layout.version != 4:
        raise ValueError("dot_scaled requires a version-4 HCU MFMA accumulator layout")
    lhs_k_pack = _unwrap_if_constexpr(lhs_k_pack)
    rhs_k_pack = _unwrap_if_constexpr(rhs_k_pack)
    if type(lhs_k_pack) is not bool or type(rhs_k_pack) is not bool:
        raise ValueError("dot_scaled packing flags must be compile-time bools")
    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    allowed = {"e2m1": (gl.int8, gl.uint8), "e4m3": (gl.float8e4nv, ), "e5m2": (gl.float8e5, )}
    for value, fmt in ((a, a_format), (b, b_format)):
        if fmt not in allowed or value.dtype not in allowed[fmt]:
            raise ValueError("scaled MMAC requires packed i8/u8 E2M1, E4M3FN, or E5M2 matching its format")
    a_format = _semantic._str_to_fp_type(a_format)
    b_format = _semantic._str_to_fp_type(b_format)
    handle = _semantic.builder.create_hcu_dot_scaled(a.handle, b.handle, acc.handle, a_scale.handle, b_scale.handle,
                                                  a_format, b_format, lhs_k_pack, rhs_k_pack)
    return gl.tensor(handle, acc.type)
