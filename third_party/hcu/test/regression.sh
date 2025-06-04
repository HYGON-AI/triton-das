#!/bin/bash
set -e

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"
export SRC_HOME=${CUR_PATH}/../../../

function run_pytest() {
  test_files=(
    "matmul.py"
    "rmsnorm.py"
  )
  file_path=${SRC_HOME}/third_party/hcu/test
  for f in ${test_files[@]}; do
    pytest_file=${file_path}/${f}
    pytest ${pytest_file}
    ret=$?
    if [ $ret -ne 0 ]; then
      return $ret
    fi
  done
  echo -e "\n================="
  echo    "Run all passed!!!"
  echo -e "=================\n"
}

# pytest
run_pytest
# TODO: benchmark