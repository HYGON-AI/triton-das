#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonHCU/WaitCntHCUUtility.h"
#include "TritonHCU/Passes.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/TargetParser.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

void init_triton_hcu_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  // HCU-specific passes whose implementations live in
  // third_party/hcu/lib/TritonHCUTransforms/ and TritonHCUToLLVM/.
  ADD_PASS_OPTION_WRAPPER_4("add_accelerate_matmul",
                            mlir::createTritonHCUAccelerateMatmul,
                            const std::string, int, int, int);
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(mlir::triton::createConvertTritonHCUToLLVMPass(arch, ftz));
        });
  ADD_PASS_OPTION_WRAPPER_3("add_mls_stream_pipeline",
                            mlir::createTritonHCUMlsStreamPipeline, int, int, int);
  ADD_PASS_WRAPPER_0("add_mls_encoding_insertion",
                     mlir::createTritonHCUMlsEncodingInsertion);
  ADD_PASS_WRAPPER_0("add_mls_lowering_pass",
                     mlir::createTritonHCUMlsLowering);
  // HCU wrapper around AMD ConvertToBufferOps — passes extra HCU-specific options.
  ADD_PASS_OPTION_WRAPPER_5("add_convert_to_buffer_ops",
                            mlir::createTritonAMDGPUConvertToBufferOps,
                            const std::string &, bool, bool, bool, bool);
  m.def("add_warp_specialize_to_llvm", [](mlir::PassManager &pm, const std::string &arch,
      int waspNumLoadWarps, int waspNumMmaWarps, bool wdraEnabled, int wdraNumLoadRegs,
      int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {
    pm.addPass(createHCUConvertWarpSpecializeToLLVM(
        arch,
        waspNumLoadWarps,
        waspNumMmaWarps,
        wdraEnabled,
        wdraNumLoadRegs,
        wdraNumMmaRegsMain,
        wdraNumMmaRegsTail));
  });
}

} // namespace

void init_triton_hcu(py::module &&m) {
  m.doc() = "Python bindings to the HCU Triton backend";
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
