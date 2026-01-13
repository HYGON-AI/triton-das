add_triton_library(TritonDistributedToLLVM
    AMD/LibDeviceToLLVM.cpp
    AMD/DistributedOpToLLVM.cpp
    AMD/ConvertAMDDistributedToLLVM.cpp

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
