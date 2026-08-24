"""Triton 3.4 Inductor compatibility for launch_* / kernel_load_* hooks.

3.4 assigned ``knobs.runtime.launch_enter_hook = fn`` (or ``None``).
3.5+ uses ``HookChain`` with ``.add`` / ``.remove``.
"""
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Callable, Generic, Optional, Type, TypeVar, Union

F = TypeVar("F", bound=Callable)


def install() -> None:
    from . import knobs

    HookChain = knobs.HookChain
    if getattr(HookChain, "_hygon_assign_compat", False):
        return

    if not hasattr(HookChain, "clear"):

        def clear(self) -> None:
            self.calls.clear()

        HookChain.clear = clear  # type: ignore[attr-defined]

    HookChain.__bool__ = lambda self: bool(self.calls)  # type: ignore[method-assign]
    HookChain.__len__ = lambda self: len(self.calls)  # type: ignore[method-assign]
    HookChain._hygon_assign_compat = True  # type: ignore[attr-defined]

    class HookChainField(Generic[F]):
        def __init__(self, reversed: bool = False):
            self.reversed = reversed
            self.private_name = ""

        def __set_name__(self, owner: Type[object], name: str) -> None:
            self.private_name = f"_hookchain_{name}"

        def __get__(self, obj: Optional[object], objclass: Optional[Type[object]] = None):
            if obj is None:
                return self
            chain = obj.__dict__.get(self.private_name)
            if chain is None:
                chain = HookChain(reversed=self.reversed)
                obj.__dict__[self.private_name] = chain
            return chain

        def __set__(self, obj: object, value: Union[None, F, object]) -> None:
            chain = self.__get__(obj, type(obj))
            if value is None:
                chain.clear()
            elif isinstance(value, HookChain):
                obj.__dict__[self.private_name] = value
            elif callable(value):
                chain.clear()
                chain.add(value)
            else:
                raise TypeError(f"hook must be callable, HookChain, or None; got {type(value)}")

        def __delete__(self, obj: object) -> None:
            self.__get__(obj, type(obj)).clear()

    fields = {
        "launch_enter_hook": False,
        "launch_exit_hook": True,
        "kernel_load_start_hook": False,
        "kernel_load_end_hook": True,
    }
    runtime_cls = knobs.runtime_knobs
    runtime = knobs.runtime
    for name, reversed_flag in fields.items():
        old = getattr(runtime_cls, name, None)
        field = HookChainField(reversed=reversed_flag)
        field.__set_name__(runtime_cls, name)
        setattr(runtime_cls, name, field)
        if isinstance(old, HookChain) and old.calls:
            chain = field.__get__(runtime, runtime_cls)
            for fn in old.calls:
                chain.add(fn)
        runtime.__dict__.pop(name, None)

    knobs.HookChainField = HookChainField  # type: ignore[attr-defined]
