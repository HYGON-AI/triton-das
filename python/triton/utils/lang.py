from triton.language import core
import triton


@core.builtin
def annotate_hint(value, attr_name, attr_val=True, _semantic=None):
    attr_name = core._unwrap_if_constexpr(attr_name)
    attr_val = core._unwrap_if_constexpr(attr_val)

    if attr_name != "non-negative":
        raise ValueError(f"Unsupported attribute name: {attr_name}")

    value.handle.set_attr(attr_name, _semantic.builder.get_bool_attr(attr_val))
    return value
