"""HCU kernel-side language helpers; import helpers from their submodules.

Do not eagerly import JIT helpers here: this package is discovered while
triton.language (and potentially triton.jit) is still being initialized.
"""
