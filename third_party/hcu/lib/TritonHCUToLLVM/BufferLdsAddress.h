// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_BUFFER_LDS_ADDRESS_H
#define TRITON_HCU_BUFFER_LDS_ADDRESS_H

#include "TritonHCU/BufferLdsConfig.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::HCU {
// Logical element coordinates to physical LDS byte offset within a
// buffer-to-LDS tile.
Value emitBufferLdsByteOffset(TritonLLVMOpBuilder &b,
                              const BufferLdsConfig &copyConfig, Value row,
                              Value col, unsigned elementBytes);

// The shared encoding already interleaves this byte offset. Apply only wrap,
// preserving allocation slot and transfer bits; never encode M0 control bits.
Value emitBufferLdsWrappedByteOffset(TritonLLVMOpBuilder &b,
                                     const BufferLdsConfig &copyConfig,
                                     Value interleavedByteOffset);
} // namespace mlir::triton::HCU
#endif
