# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.

add_triton_library(SIMTIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  SIMTTableGen
  SIMTAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)