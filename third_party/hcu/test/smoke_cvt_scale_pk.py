#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Deprecated entrypoint: use test_cvt_scale_pk.py (pytest, chip-gated, asserted).

Kept so old `python smoke_cvt_scale_pk.py` invocations still run the suite.
"""

import pytest
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.exit(
        pytest.main([str(Path(__file__).with_name("test_cvt_scale_pk.py")), "-v"])
    )
