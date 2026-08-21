#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -e

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"

function build_triton() {
  cd ${SRC_HOME}
  pip uninstall -y triton
  pip install -e .
  cd -
}

function regression_for_test() {
  # =====================
  # run pytest cases
  # =====================
  run_pytest

  # =====================
  # run googletest cases
  # =====================
  run_google_test

  # =====================
  # run lit cases
  # =====================
  run_lit 0
}

source ${CUR_PATH}/launch.sh

build_llvm
clean_cache
build_triton
regression_for_test
