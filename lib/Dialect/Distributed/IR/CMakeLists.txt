# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

add_triton_library(DistributedIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  DistributedTableGen
  DistributedAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)