#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import pytest


def pytest_addoption(parser):
    parser.addoption("--Te", action='store_true', help='Use transformer engine for benchmark')


@pytest.fixture
def use_Te(request):
    return request.config.getoption('--Te')
