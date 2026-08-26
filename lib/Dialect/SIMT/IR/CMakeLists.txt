# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

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