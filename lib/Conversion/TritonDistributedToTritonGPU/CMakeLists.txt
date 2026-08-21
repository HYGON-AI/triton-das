# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.

add_triton_library(TritonDistributedToTritonGPU
    TritonDistributedToTritonGPU.cpp

    DEPENDS
    TritonDistributedConversionPassIncGen
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ProtonIR
    DistributedIR
    SIMTIR
    TritonGPUIR
    TritonGPUTransforms
)
