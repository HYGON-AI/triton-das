# Modified by Hygon Information Technology Co., Ltd., 2026.

add_triton_library(TritonAMDGPUIR
  Dialect.cpp

  DEPENDS
  TritonAMDGPUTableGen
  TritonAMDGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  # Dialect verify uses AMD::getElemCoordinatesFromRegisters / getRegFromCoordinates.
  TritonAMDUtils
)
