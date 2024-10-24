#!/bin/bash

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"
cd $CUR_PATH/../../python
export DEBUG=OFF

source ${CUR_PATH}/launch.sh
build_llvm

function usage() {
  echo "--version <version index> - To Specify the version for Triton"
  echo "    Default Value is 2.1.0"
  echo "--llvm_build_dir <path> - To Specify the llvm-project build path"
  echo "    If not specify, will use default llvm package"
  echo "--release - To Specify release the triton wheel package"
}

COMMIT_ID=$(git rev-parse --short HEAD)
BRANCH_TAG=$(git rev-parse --abbrev-ref HEAD | sed 's/\//./g')
ROCM_TAG="rocm5.7x"
export VERSION="2.1.0"

while [ $# -gt 0 ]; do
  if [ "$1" == "--version" ]; then
    export VERSION=$2
    shift 2
  elif [ "$1" == "--llvm_build_dir" ]; then
    export LLVM_BUILD_DIR=$2
    shift 2
  elif [ "$1" == "--release" ]; then
    export RELEASE="true"
    shift 1
  else
    echo -e "\nArgument Error!!!\n"
    usage
    cd - > /dev/null 2>&1
    return
  fi
done

VERSION_TAG="${VERSION}+${BRANCH_TAG}.${COMMIT_ID}.${ROCM_TAG}"

cp setup.py setup.py.bk

sed -i -r "s/version\=\"(2.*)\"/version=\"${VERSION_TAG}\"/g" setup.py.bk

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
      ${python_app} setup.py.bk bdist_wheel
    else
      LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
      LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
      LLVM_SYSPATH=$LLVM_BUILD_DIR \
      ${python_app} setup.py.bk bdist_wheel
    fi
  fi
done

WHL_FILE=$(ls -t dist/*.whl | head -n 1)
echo -e "\nThe wheel path is:"
echo -e "$(realpath $(dirname "${WHL_FILE}"))\n"

rm -rf setup.py.bk

cd - > /dev/null 2>&1