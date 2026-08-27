#!/bin/bash

CUR_PATH="$( cd $( dirname ${BASH_SOURCE} );pwd )"
cd $CUR_PATH/../../
export DEBUG=OFF

source ${CUR_PATH}/launch.sh

function usage() {
  echo "--release - To specify release the triton wheel package"
  echo "--llvm_repo - To specify the triton-llvm repo, such as ssh://git@10.65.42.70:8022/buhui/llvm-project.git"
  echo "--file_server_ip - To specify the ip of file server, such as 10.65.42.71"
  echo "--file_server_user - To specify the user of file server"
  echo "--file_server_pwd - To specify the password of file server user"
}

BRANCH_TAG=$(git rev-parse --abbrev-ref HEAD | sed 's/\//./g')
ROCM_TAG="rocm5.7x"
# export TRITON_WHEEL_VERSION_SUFFIX=".${BRANCH_TAG}.${ROCM_TAG}"

while [ $# -gt 0 ]; do
  if [ "$1" == "--release" ]; then
    RELEASE="true"
    shift 1
  elif [ "$1" == "--llvm_repo" ]; then
    LLVM_REPO=$2
    shift 2
  elif [ "$1" == "--file_server_ip" ]; then
    SERVER_IP=$2
    shift 2
  elif [ "$1" == "--file_server_user" ]; then
    SERVER_USER=$2
    shift 2
  elif [ "$1" == "--file_server_pwd" ]; then
    SERVER_PWD=$2
    shift 2
  else
    echo -e "\nArgument Error!!!\n"
    usage
    cd - > /dev/null 2>&1
    return
  fi
done

build_llvm "${LLVM_REPO}" "${SERVER_IP}" "${SERVER_USER}" "${SERVER_PWD}"

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

# local cache
mkdir ~/.triton/
cd ~/.triton/
wget http://10.16.1.201:8000/Jenkins/CompileDep/triton/nvidia.3.6.x.tar.gz
tar -zxf nvidia.3.6.x.tar.gz
wget http://10.16.1.201:8000/Jenkins/CompileDep/triton/json-v3.11.3.tar.gz
tar -zxf json-v3.11.3.tar.gz
cd $CUR_PATH/../../

for python_app in "${PYTHON_LIST[@]}"; do
  app_path=$(which ${python_app})
  if [[ -z ${app_path} ]]; then
    echo "Can't find python app for ${python_app}"
    exit 1
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
