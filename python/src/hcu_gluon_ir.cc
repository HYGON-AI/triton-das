// HCU-specific Gluon builder bindings. Python API names stay on GluonOpBuilder.
#include "ir.h"
#include "hcu_gluon_ir.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonHCU/TargetUtils.h"
#include "TritonHCU/MlsGroup.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "pybind11/stl.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
namespace py = pybind11;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttag = triton::amdgpu;

void enableHcuGluonInlining(MLIRContext *ctx);

void init_hcu_gluon_ir(py::handle builder) {
  auto bind = [&](const char *name, auto method) {
    builder.attr(name) = py::cpp_function(method, py::name(name),
                                         py::is_method(builder));
  };
  bind("create_hcu_set_priority", [](TritonOpBuilder &self, int32_t priority) {
    enableHcuGluonInlining(self.getContext());
    self.create<ROCDL::SetPrioOp>(priority);
  });
  bind("create_hcu_wait_lgkmcnt", [](TritonOpBuilder &self, int32_t count) {
    self.create<ttag::MemoryCounterWaitOp>(
        /*load=*/nullptr, /*store=*/nullptr,
        /*ds=*/self.getBuilder().getI32IntegerAttr(count));
  });
  bind("create_hcu_wave_id", [](TritonOpBuilder &self) -> Value {
    // HW_REG_HW_ID1 = 23; extract its five-bit CTA wave index at offset zero.
    constexpr int32_t hwreg = 23 | ((5 - 1) << 11);
    Value selector = self.create<arith::ConstantIntOp>(hwreg, 32);
    return LLVM::createLLVMIntrinsicCallOp(
               self.getBuilder(), self.getLastLoc(), "llvm.amdgcn.s.getreg",
               TypeRange{self.getBuilder().getI32Type()}, ValueRange{selector})
        .getResult(0);
  });
  bind("get_hcu_mmac_instruction",
       [](TritonOpBuilder &self, Type a, Type b, Type c, unsigned maxK,
          std::vector<unsigned> tile, unsigned version, unsigned layout, const std::string &arch) {
         if (tile.size() != 2 || maxK == 0)
           throw py::value_error("MMAC query requires a rank-two tile and positive max_k");
         if (layout < 1 || layout > 4)
           throw py::value_error("requested MMAC layout must be in [1,4]");
         if (layout != 1 && !triton::HCU::supportsHCUISAFeature(arch, triton::HCU::HCUISAFeature::MMAC_LAYOUT))
           throw py::value_error("target does not support native MMAC layout modifiers");
         auto instruction = MfmaIntrinsic::selectFor(
             self.getLastLoc(), version, tile[0], tile[1], maxK, a, b, c,
             false, false, triton::HCU::deduceHCUISAFeature(arch));
         if (failed(instruction))
           throw py::value_error("no native MMAC instruction for the requested dtype/geometry");
         bool lit = layout == 2 || layout == 4;
         bool lts = layout == 3 || layout == 4;
         if ((lit && !instruction->supportsMmacLit) || (lts && !instruction->supportsMmacLts))
           throw py::value_error("selected native MMAC instruction does not support the requested layout");
         return py::make_tuple(
             std::vector<unsigned>{instruction->mDim, instruction->nDim, instruction->kDim},
             instruction->kBase);
       });
  bind("get_hcu_mmac_interleave2_layout",
       [](TritonOpBuilder &self, Attribute parent) -> Attribute {
         return ttg::HCUMmacInterleave2EncodingAttr::get(self.getContext(), parent);
       });
  bind("check_hcu_mls_store_configuration",
       [](TritonOpBuilder &, std::vector<unsigned> tile, unsigned bits,
          unsigned interleave, unsigned version, std::vector<unsigned> order) {
         if (tile.size() != 2 || order.size() != 2 ||
             failed(triton::HCU::MlsInsn::selectOrGetMatrixStoreInsn(
                 tile[0], tile[1], bits, version,
                 static_cast<triton::HCU::MlsInterleaveKind>(interleave), order[0] == 1)))
           throw py::value_error("unsupported HCU matrix_store tile, element width, version, order or interleave");
       });
  bind("create_hcu_mls_store_from_registers",
           [](TritonOpBuilder &self, Value base, Value value,
              std::vector<Value> shape, std::vector<Value> strides,
              std::vector<Value> indices, std::vector<unsigned> tile,
              unsigned interleave, unsigned version,
              std::vector<unsigned> order, std::vector<int32_t> boundaryCheck) {
             auto ty = cast<RankedTensorType>(value.getType());
             auto f = ttg::getUnderlyingMmacEncoding(ty.getEncoding());
             if (!f)
               throw std::invalid_argument("MLS store requires an MMAC result layout");
             auto ctx = self.getContext();
             auto tensorShape = DenseI32ArrayAttr::get(
                 ctx,
                 llvm::to_vector(llvm::map_range(
                     ty.getShape(), [](int64_t x) { return int32_t(x); })));
             auto op = self.create<ttag::MatrixStoreFromRegOp>(
                 base, value, shape, strides, tensorShape, indices,
                 DenseI32ArrayAttr::get(ctx, boundaryCheck), tt::CacheModifier::NONE,
                 tt::EvictionPolicy::NORMAL);
             auto attr = ttag::MlsStoreEncodingAttr::get(
                 ctx, tile, ty.getElementType().getIntOrFloatBitWidth(),
                 interleave, version, order,
                 f.getWarpsPerCTA());
             op->setAttr(ttag::MlsStoreEncodingAttr::getMnemonic(), attr);
           });
  bind("create_hcu_packed_fp32_scale_fma",
           [](TritonOpBuilder &self, Value dot, Value aScale, Value bScale,
              Value accumulator) -> Value {
             return self
                 .create<ttag::HCUPackedScaleFmaOp>(dot.getType(), dot, aScale,
                                                    bScale, accumulator)
                 .getResult();
           });
  bind("create_hcu_sched_barrier",
           [](TritonOpBuilder &self, int32_t mask) {
             enableHcuGluonInlining(self.getContext());
             self.create<ROCDL::SchedBarrier>(mask);
           });
  bind("create_hcu_sched_group_barrier",
           [](TritonOpBuilder &self, int32_t mask, int32_t count, int32_t sync) {
             enableHcuGluonInlining(self.getContext());
             self.create<ROCDL::SchedGroupBarrier>(mask, count, sync);
           });
  bind("create_hcu_ebarrier_arrive",
       [](TritonOpBuilder &self, Value id, int count) {
         enableHcuGluonInlining(self.getContext());
         self.create<ROCDL::HCUEbarrierArriveOp>(id, count);
       });
  bind("create_hcu_ebarrier_sync",
       [](TritonOpBuilder &self, Value id, int count) {
         enableHcuGluonInlining(self.getContext());
         self.create<ROCDL::HCUEbarrierSyncOp>(id, count);
       });
  bind("create_hcu_abarrier_try_wait",
           [](TritonOpBuilder &self, Value id, Value phase) -> Value {
             enableHcuGluonInlining(self.getContext());
             return self.create<ROCDL::HCUAbarrierTryWaitOp>(id, phase).getResult();
           });
  bind("create_hcu_abarrier_test_wait",
           [](TritonOpBuilder &self, Value id, Value phase) -> Value {
             enableHcuGluonInlining(self.getContext());
             return self.create<ROCDL::HCUAbarrierTestWaitOp>(id, phase).getResult();
           });
  bind("create_hcu_abarrier_seq",
           [](TritonOpBuilder &self, Value id) {
             enableHcuGluonInlining(self.getContext());
             self.create<ROCDL::HCUAbarrierSeqOp>(id);
           });
  bind("create_hcu_abarrier_arrive",
           [](TritonOpBuilder &self, Value id, int count) {
             enableHcuGluonInlining(self.getContext());
             self.create<ROCDL::HCUAbarrierArriveOp>(id, count);
           });
  bind("get_hcu_mls_shared_layout",
           [](TritonOpBuilder &self, unsigned opIdx, std::vector<unsigned> tile,
              unsigned bits, unsigned kind, unsigned alt, unsigned version,
              std::vector<unsigned> order) -> Attribute {
             auto *ctx = self.getContext();
             triton::LinearLayout::BasesT bases;
             bases[StringAttr::get(ctx, "block")] = {};
             auto cta = ttg::CTAEncodingAttr::get(
                 ctx, triton::LinearLayout(std::move(bases),
                                           triton::standardOutDimNames(ctx, 2)));
             return ttg::HCUMlsSharedEncodingAttr::get(
                 self.getContext(), opIdx, tile, bits,
                 static_cast<ttg::MlsElemBitTyKind>(kind), alt, version, order,
                 cta);
           });
  bind("create_hcu_mls_load_to_shared",
           [](TritonOpBuilder &self, Value dest, Value base,
              std::vector<Value> shape, std::vector<Value> strides,
              std::vector<Value> indices, std::vector<unsigned> warps,
              Value pred, std::vector<int32_t> boundaryCheck) {
             auto ty = cast<ttg::MemDescType>(dest.getType());
             auto a = cast<ttg::HCUMlsSharedEncodingAttr>(ty.getEncoding());
             auto tileShape = DenseI32ArrayAttr::get(
                 self.getContext(),
                 llvm::to_vector(llvm::map_range(
                     ty.getShape(), [](int64_t x) { return int32_t(x); })));
             auto op = self.create<ttag::MatrixLoadToLocalOp>(
                 base, shape, strides, tileShape, indices,
                 DenseI32ArrayAttr::get(self.getContext(), boundaryCheck),
                 tt::CacheModifier::NONE, tt::EvictionPolicy::NORMAL, false,
                 dest, pred);
             auto attr = ttag::MlsEncodingAttr::get(
                 self.getContext(), a.getOpIdx(), a.getMlsTile(),
                 a.getElemBitWidth(),
                 static_cast<unsigned>(a.getElemBitTyKind()), a.getAlt2Kind(),
                 a.getVersion(), a.getOrder(), warps);
             op->setAttr(ttag::MlsEncodingAttr::getMnemonic(), attr);
           });
  bind("create_hcu_abarrier_init",
       [](TritonOpBuilder &self, int id, int count) {
         enableHcuGluonInlining(self.getContext());
         self.create<ROCDL::HCUAbarrierInitOp>(id, count);
       });
  bind("create_hcu_abarrier_invalidate", [](TritonOpBuilder &self, int id) {
    enableHcuGluonInlining(self.getContext());
    self.create<ROCDL::HCUAbarrierInvOp>(id);
  });

  // Explicit Gluon layouts and descriptors.
  bind("get_hcu_scale_layout",
       [](TritonOpBuilder &self, std::vector<int64_t> blockShape,
          int64_t rowStride, int64_t bankXor) -> Attribute {
         auto layout = ttag::ScaleEncodingAttr::getChecked(
               [&] { return emitError(self.getLastLoc()); }, self.getContext(),
               llvm::ArrayRef<int64_t>(blockShape), rowStride, bankXor);
         if (!layout)
           throw py::value_error("invalid HCU Scale Buffer layout");
         return layout;
       });
  bind("get_hcu_scale_mem_desc_ty",
       [](TritonOpBuilder &self, std::vector<int64_t> shape,
          Attribute layout) -> Type {
         if (!isa<ttag::ScaleEncodingAttr>(layout))
           throw py::value_error("expected an HCU Scale Buffer layout");
         auto type = ttg::MemDescType::getChecked(
               [&] { return emitError(self.getLastLoc()); }, shape,
               self.getBuilder().getI8Type(), layout,
               ttag::ScaleMemorySpaceAttr::get(self.getContext()),
               /*mutableMemory=*/false);
         if (!type)
           throw py::value_error("invalid HCU Scale Buffer descriptor");
         return type;
       });
  bind("get_hcu_buffer_lds_shared_layout",
       [](TritonOpBuilder &self, Attribute linear, unsigned elementBytes,
          unsigned contiguousDim, unsigned rows, unsigned rowBytes,
          unsigned waves, unsigned copyBytesPerLane, unsigned rowsPerChunk,
          unsigned wrapCount, unsigned wrapStepBytes, unsigned wavesPerRowGroup,
          bool transferMajor, unsigned rowBitSwap) -> Attribute {
         auto component = dyn_cast<ttg::SharedLinearEncodingAttr>(linear);
         if (!component)
           throw py::value_error("expected a shared linear component");
         auto layout = ttg::HCUBufferLdsSharedEncodingAttr::getChecked(
             [&] { return emitError(self.getLastLoc()); }, self.getContext(),
             component, elementBytes, contiguousDim, rows, rowBytes, waves,
             copyBytesPerLane, rowsPerChunk, wrapCount, wrapStepBytes,
             wavesPerRowGroup, transferMajor, rowBitSwap);
         if (!layout)
           throw py::value_error("invalid buffer-to-LDS placement");
         return layout;
       });

  // Native scaled dot operations.
  bind("create_hcu_scale_write",
       [](TritonOpBuilder &self, Type resultType, Value src, Value pred,
          int32_t mode) -> Value {
         // An unconditional write must not acquire a predicated lifetime.
         if (matchPattern(pred, m_One()))
           pred = Value();
         return self.create<ttag::ScaleWriteOp>(
             resultType, src, pred, Value(),
             self.getBuilder().getI32IntegerAttr(mode), IntegerAttr());
       });
  bind("create_hcu_dot_scaled",
       [](TritonOpBuilder &self, Value a, Value b, Value acc, Value aScale,
          Value bScale, tt::ScaleDotElemType aFormat,
          tt::ScaleDotElemType bFormat, bool lhsKPack, bool rhsKPack) -> Value {
         return self.create<ttag::DotScaledOp>(
             acc.getType(), a, b, acc, aScale, bScale, aFormat, bFormat,
             /*fastMath=*/false, lhsKPack, rhsKPack);
       });
}
