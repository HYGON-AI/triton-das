# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

add_triton_library(TritonDistributedToLLVM
    HCU/LibDeviceToLLVM.cpp
    HCU/DistributedOpToLLVM.cpp
    HCU/ConvertHCUDistributedToLLVM.cpp

    DEPENDS
    TritonDistributedToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    DistributedIR
    TritonAMDGPUIR
    TritonAMDGPUToLLVM
    TritonAMDGPUDialectToLLVM
    TritonAMDGPUTransforms
    TritonAMDUtils
    TritonAMDAnalysis
)
