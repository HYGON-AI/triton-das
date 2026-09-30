# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""IR test runner shared by transport and scheduling tests.

Keep this module independent of either layer's tests so transport tests can
run with only the async-copy commit applied.
"""
from triton._C.libtriton import ir, hcu


def run_pass(tmp_path, source, add_pass):
    path = tmp_path / "input.mlir"
    path.write_text(source)
    context = ir.context()
    ir.load_dialects(context)
    hcu.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    manager = ir.pass_manager(context)
    manager.enable_debug()
    add_pass(manager)
    manager.run(module, "buffer_to_lds_test")
    return str(module)
