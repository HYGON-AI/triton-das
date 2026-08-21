# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

from triton.language import core

# Allowed key prefixes. Frontend only checks these; no per-key registry.
_HINT_PREFIXES = ("val.", "func.", "mod.flag.")


def _unwrap(x):
    return core._unwrap_if_constexpr(x)


def _to_ir_attr(builder, val):
    """Map a Python hint value to an MLIR attribute (bool / int / str)."""
    val = _unwrap(val)
    if isinstance(val, bool):
        return builder.get_bool_attr(val)
    if isinstance(val, int):
        return builder.get_int32_attr(val)
    if isinstance(val, str):
        return builder.get_string_attr(val)
    raise TypeError(
        f"hint value must be bool, int, or str, got {type(val)}")


@core.builtin
def hint(key, arg=None, value=None, _semantic=None):
    """Compile-time hint marker. Keys must start with val. / func. / mod.flag.

    Forms::

        x = hint("val.no_negative", x)            # val.*: value defaults to True
        x = hint("val.no_negative", x, False)
        x = hint("val.unroll_factor", x, 4)
        hint("func.args_ptr.no_alias", True)      # func.*: value required
        hint("func.attr.optsize", True)            # → LLVM fn attr "optsize"
        hint("mod.flag.my_feature", True)          # module flag; bool/int/str OK
        hint("mod.flag.tile_factor", 4)
        hint("mod.flag.mode", "fast")
    """
    key = _unwrap(key)
    if not isinstance(key, str) or not key.startswith(_HINT_PREFIXES):
        raise ValueError(
            f"hint key must start with {_HINT_PREFIXES}, got {key!r}")

    builder = _semantic.builder
    ir_attr = "tt.hint." + key

    if key.startswith("val."):
        if arg is None:
            raise ValueError(f"hint '{key}' requires a value target")
        # val.* may omit the attr value; default True.
        attr_val = True if value is None else value
        handle = arg.handle if hasattr(arg, "handle") else arg
        handle.set_attr(ir_attr, _to_ir_attr(builder, attr_val))
        return arg

    # func.* / mod.flag.*: hint(key, value) — second positional is required.
    if arg is None:
        raise ValueError(f"hint '{key}' requires an explicit value")
    if value is not None:
        raise ValueError(
            f"hint '{key}' takes hint(key, value), not a third argument")
    attr = _to_ir_attr(builder, arg)

    if key.startswith("func."):
        builder.set_func_attr(ir_attr, attr)
    else:
        # mod.flag.* → enclosing module (mechanism only; no built-in consumer)
        builder.set_module_attr(ir_attr, attr)
    return None
