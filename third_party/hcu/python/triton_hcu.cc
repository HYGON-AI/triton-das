#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonHCU/WaitCntHCUUtility.h"
#include "TritonHCU/Passes.h"
#include "TritonHCU/BufferLdsEncoding.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "third_party/hcu/include/TritonHCUToLLVM/Passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/TargetParser.h"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

void addBlockPingpongPipeline(mlir::PassManager &pm,
                              std::optional<unsigned> globalDistance,
                              unsigned ldsDistance) {
  // Schedule split-BK M0-D0-M1-D1 or whole-BK M-D, turn slot parity into a
  // compile-time property with late 2K unrolling, then expose the two LDS slots
  // as independent buffers.
  mlir::TritonHCUBlockPingpongOptions options;
  if (globalDistance)
    options.globalPrefetchDistance = *globalDistance;
  options.ldsPrefetchDistance = ldsDistance;
  pm.addPass(mlir::createTritonHCUBlockPingpong(options));
  pm.addPass(mlir::triton::createTritonLoopUnroll());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createTritonHCUFinalizeBlockPingpong());
}

void init_triton_hcu_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  // HCU-specific passes whose implementations live in
  // third_party/hcu/lib/TritonHCUTransforms/ and TritonHCUToLLVM/.
  ADD_PASS_OPTION_WRAPPER_4("add_accelerate_matmul",
                            mlir::createTritonHCUAccelerateMatmul,
                            const std::string, int, int, int);
  ADD_PASS_WRAPPER_0("add_select_buffer_lds_config",
                     mlir::createTritonHCUSelectBufferLdsConfig);
  m.def("add_prepare_block_pingpong_layout",
        [](mlir::PassManager &pm,
           std::optional<unsigned> globalDistance, unsigned ldsDistance) {
          mlir::TritonHCUPrepareBlockPingpongLayoutOptions options;
          if (globalDistance)
            options.globalPrefetchDistance = *globalDistance;
          options.ldsPrefetchDistance = ldsDistance;
          pm.addPass(mlir::createTritonHCUPrepareBlockPingpongLayout(options));
        }, py::arg("pm"),
        py::arg("global_prefetch_distance") = py::none(),
        py::arg("lds_prefetch_distance") = 1);
  ADD_PASS_OPTION_WRAPPER_1("add_prepare_wasp_split_plan",
                            mlir::createTritonHCUPrepareWaspSplitPlan, int);
  ADD_PASS_WRAPPER_0("add_consumer_pingpong",
                     mlir::createTritonHCUConsumerPingpong);
  ADD_PASS_OPTION_WRAPPER_1("add_optimize_partition_warps",
                            mlir::triton::gpu::createTritonGPUOptimizePartitionWarpsHCU,
                            bool);
  // HCU pipeline only: materialize known tt.hint.func.* then clean tt.hint.*.
  ADD_PASS_WRAPPER_0("add_apply_func_hints_and_clean",
                     mlir::createTritonHCUApplyFuncHintsAndClean);
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(mlir::triton::createConvertTritonHCUToLLVMPass(arch, ftz));
        });
  ADD_PASS_OPTION_WRAPPER_3("add_mls_stream_pipeline",
                            mlir::createTritonHCUMlsStreamPipeline, int, int,
                            bool);
  m.def("add_prepare_block_pingpong",
        [](mlir::PassManager &pm, std::optional<unsigned> globalDistance,
           unsigned ldsDistance) {
          mlir::TritonHCUPrepareBlockPingpongOptions options;
          if (globalDistance)
            options.globalPrefetchDistance = *globalDistance;
          options.ldsPrefetchDistance = ldsDistance;
          pm.addPass(mlir::createTritonHCUPrepareBlockPingpong(options));
        }, py::arg("pm"), py::arg("global_prefetch_distance") = py::none(),
        py::arg("lds_prefetch_distance") = 1);
  ADD_PASS_WRAPPER_0("add_utc_warmup", mlir::createTritonHCUUTCWarmup);
  ADD_PASS_WRAPPER_0("add_convert_buffer_lds_copies",
                     mlir::createTritonHCUConvertBufferLdsCopies);
  ADD_PASS_WRAPPER_0("add_synchronize_buffer_lds_copies",
                     mlir::createTritonHCUSynchronizeBufferLdsCopies);
  m.def("add_schedule_buffer_to_lds",
        [](mlir::PassManager &pm, unsigned ldsDistance) {
          pm.addPass(mlir::createTritonHCUScheduleBufferToLds({ldsDistance}));
        }, py::arg("pm"), py::arg("lds_prefetch_distance") = 1);
  m.def("add_block_pingpong", addBlockPingpongPipeline, py::arg("pm"),
        py::arg("global_prefetch_distance") = py::none(),
        py::arg("lds_prefetch_distance") = 1);
  ADD_PASS_WRAPPER_0("add_pack_block_pingpong_bit8",
                     mlir::createTritonHCUPackBlockPingpongBit8);
  ADD_PASS_WRAPPER_0("add_pack_bit8_loop_carriers",
                     mlir::createTritonHCUPackBit8LoopCarriers);
  ADD_PASS_OPTION_WRAPPER_1("add_scale_pipeline",
                            mlir::createTritonHCUScalePipeline, bool);
  ADD_PASS_WRAPPER_0("add_mls_encoding_insertion",
                     mlir::createTritonHCUMlsEncodingInsertion);
  ADD_PASS_WRAPPER_0("add_mls_lowering_pass",
                     mlir::createTritonHCUMlsLowering);
  ADD_PASS_OPTION_WRAPPER_1("add_update_async_wait_count",
                            mlir::createTritonHCUUpdateAsyncWaitCount,
                            std::string);
  ADD_PASS_WRAPPER_0("add_allocate_scale_buffer",
                     mlir::triton::HCU::createAllocateScaleBuffer);
  // HCU wrapper around AMD ConvertToBufferOps — passes extra HCU-specific options.
  ADD_PASS_OPTION_WRAPPER_6("add_convert_to_buffer_ops",
                            mlir::createTritonAMDGPUConvertToBufferOps,
                            const std::string &, bool, bool, bool, bool, bool);
  m.def("add_warp_specialize_to_llvm", [](mlir::PassManager &pm, const std::string &arch,
      bool wdraEnabled) {
    pm.addPass(createHCUConvertWarpSpecializeToLLVM(
        arch, wdraEnabled));
  });}

} // namespace

void init_triton_hcu(py::module &&m) {
  m.doc() = "Python bindings to the HCU Triton backend";
  m.def(
      "get_buffer_lds_producer_layout",
      [](std::vector<std::vector<int32_t>> offsetBases,
         std::vector<std::vector<int32_t>> blockBases, unsigned alignment,
         unsigned elementBytes, unsigned contiguousDim, unsigned rows,
         unsigned rowBytes, unsigned waves, unsigned copyBytesPerLane,
         unsigned rowsPerChunk, unsigned wrapCount, unsigned wrapStepBytes,
         unsigned wavesPerRowGroup, bool transferMajor, unsigned rowBitSwap) {
        using namespace mlir;
        namespace tt = mlir::triton;
        namespace ttg = mlir::triton::gpu;
        MLIRContext context(MLIRContext::Threading::DISABLED);
        context.getOrLoadDialect<ttg::TritonGPUDialect>();
        auto offset = StringAttr::get(&context, "offset");
        auto block = StringAttr::get(&context, "block");
        // Check structural invariants before construction; getChecked below
        // handles surjectivity and the remaining encoding constraints.
        if ((elementBytes != 1 && elementBytes != 2) || contiguousDim > 1 ||
            !llvm::isPowerOf2_32(rows) || !llvm::isPowerOf2_32(rowBytes) ||
            rowBytes < elementBytes ||
            uint64_t(rows) * (rowBytes / elementBytes) > (uint64_t(1) << 30) ||
            offsetBases.size() + blockBases.size() > 30)
          throw py::value_error("invalid buffer-to-LDS dimensions");
        SmallVector<int32_t> shape{int32_t(rows), int32_t(rowBytes / elementBytes)};
        if (contiguousDim == 0)
          std::swap(shape[0], shape[1]);
        for (const auto *bases : {&offsetBases, &blockBases})
          for (const auto &basis : *bases)
            if (basis.size() != 2 || basis[0] < 0 || basis[1] < 0 ||
                basis[0] >= shape[0] || basis[1] >= shape[1])
              throw py::value_error("invalid buffer-to-LDS basis");
        auto names = tt::standardOutDimNames(&context, 2);
        tt::LinearLayout::BasesT linearBases;
        linearBases[offset] = std::move(offsetBases);
        linearBases[block] = std::move(blockBases);
        tt::LinearLayout linearLayout(
            std::move(linearBases), {{names[0], shape[0]}, {names[1], shape[1]}},
            /*requireSurjective=*/false);
        auto linear = ttg::SharedLinearEncodingAttr::getChecked(
            [&] { return emitError(UnknownLoc::get(&context)); }, &context,
            linearLayout, alignment);
        if (!linear)
          throw py::value_error("invalid buffer-to-LDS linear component");
        auto placement = ttg::HCUBufferLdsSharedEncodingAttr::getChecked(
            [&] { return emitError(UnknownLoc::get(&context)); }, &context,
            linear, elementBytes, contiguousDim, rows, rowBytes, waves,
            copyBytesPerLane, rowsPerChunk, wrapCount, wrapStepBytes,
            wavesPerRowGroup, transferMajor, rowBitSwap);
        if (!placement)
          throw py::value_error("invalid buffer-to-LDS placement");
        auto layout =
            tt::HCU::getBufferLdsProducerEncoding(placement).getLinearLayout();
        auto bases = [&](StringRef name) {
          return layout.getBases().lookup(StringAttr::get(&context, name));
        };
        auto sizes = layout.getOutDimSizes();
        return std::make_tuple(
            bases("register"), bases("lane"), bases("warp"), bases("block"),
            std::vector<int32_t>(sizes.begin(), sizes.end()));
      });
  m.def("has_buffer_lds_encoding", [](mlir::ModuleOp mod) {
    return mlir::triton::HCU::hasBufferLdsEncoding(mod);
  });
  auto passes = m.def_submodule("passes");
  init_triton_hcu_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  m.def("add_wait_cnt", [](mlir::ModuleOp mod) {
    ::mlir::triton::HCU::addWaitCntHCU(mod);
  });
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::amdgpu::TritonAMDGPUDialect>();
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
