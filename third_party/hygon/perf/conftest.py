#!/usr/bin/env python
# coding=utf-8

import pytest


def pytest_addoption(parser):
    parser.addoption("--Te", action='store_true', help='Use transformer engine for benchmark')


@pytest.fixture
def use_Te(request):
    return request.config.getoption('--Te')
