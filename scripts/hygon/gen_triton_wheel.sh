#!/bin/bash

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"
cd $CUR_PATH/../../python
export DEBUG=OFF

source ${CUR_PATH}/launch.sh
build_llvm

function usage() {
  echo "--release - To Specify release the triton wheel package"
}

BRANCH_TAG=$(git rev-parse --abbrev-ref HEAD | sed 's/\//./g')
ROCM_TAG="rocm5.7x"
export TRITON_WHEEL_VERSION_SUFFIX=".${BRANCH_TAG}.${ROCM_TAG}"

while [ $# -gt 0 ]; do
  if [ "$1" == "--release" ]; then
    RELEASE="true"
    shift 1
  else
    echo -e "\nArgument Error!!!\n"
    usage
    cd - > /dev/null 2>&1
    return
  fi
done

if [[ ${RELEASE} == "true" ]]; then
  PYTHON_LIST=(
    "python3.8"
    "python3.10"
    "python3.12"
  )
else
  PYTHON_LIST=(
    "python3"
  )
fi

for python_app in ${PYTHON_LIST[@]}; do
  app_path=$(which ${python_app})
  if [[ -z ${app_path} ]]; then
    echo "Can't find python app for ${python_app}"
    exit -1
  else
    if [ -z ${LLVM_BUILD_DIR} ]; then
      ${python_app} setup.py bdist_wheel
    else
      LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
      LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
      LLVM_SYSPATH=$LLVM_BUILD_DIR \
      ${python_app} setup.py bdist_wheel
    fi
  fi
done

WHL_FILE=$(ls -t dist/*.whl | head -n 1)
echo -e "\nThe wheel path is:"
echo -e "$(realpath $(dirname "${WHL_FILE}"))\n"

cd - > /dev/null 2>&1