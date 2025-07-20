from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, amd
from triton.runtime.errors import HSACOError
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def is_fma_supported(lhsType, rhsType):
        return lhsType == rhsType and (lhsType.is_fp16() or lhsType.is_fp32())

    def get_gfx94_limits(lhsType, rhsType):
        if is_fma_supported(lhsType.scalar, rhsType.scalar):
            return (1, 1, 1)
        return (16, 16, 1)

    def get_gfx9_limits(lhsType, rhsType):
        if is_fma_supported(lhsType.scalar, rhsType.scalar):
            return (1, 1, 1)
        return (16, 16, 1)

    arch_str = target.arch
    if "gfx94" in arch_str:
        return get_gfx94_limits
    if "gfx9" in arch_str:
        return get_gfx9_limits
    # gfx11 and gfx12 architectures will only support 16,16,16 with wmma instructions
    return lambda lhsType, rhsType: (1, 1, 1) if is_fma_supported(lhsType.scalar, rhsType.scalar) else (16, 16, 16)


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 1
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", )
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    enable_fp_fusion: bool = True
    # TODO: Implement cooperative grid launch for AMD:
    # See: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'
    optimize_epilogue: bool = False

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "none" variant preserves the default
    # instruction scheduling of the AMDGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    #
    # Current experimental scheduling variants:
    #
    # llvm-iglp-0: injects `llvm.amdgcn.iglp_opt` intrinsic call with value `0` to the GEMM's
    #              k-loop; i.e., "interleave DS and MFMA instructions for small GEMM kernels".
    # llvm-iglp-1: injects `llvm.amdgcn.iglp_opt` intrinsic call with value `1` to the GEMM's
    #              k-loop; i.e., "interleave DS and MFMA instructions for single wave small
    #              GEMM kernels.".
    # local-prefetch: implements instruction scheduling similar to the one from the ROCm Composable
    #                 Kernel library. Note, this variant requires the use of buffer load/store ops
    #                 and a special software pipelining style - i.e., 1x LDS and 1x register
    #                 prefetch buffers for each GEMM tile.
    # Extend options for HCU:
    # llvm-iglp-8: injects `llvm.amdgcn.iglp_opt` intrinsic call with value `8` to the GEMM's
    #              k-loop; i.e., "interleave DS and MFMA instructions for common GEMM kernels".
    instruction_sched_variant: str = 'none'

    # Extend options for HCU
    # 1. set scheduling latency for mmac and ds:
    #    - none: use default scheduling latency which equal mmac1-ds5
    #    - see get_options_args() to get more options.
    sched_latency: str = 'none'

    def __post_init__(self):
        default_libdir = os.path.join(HIPBackend.path_to_rocm(), 'amdgcn/bitcode/')
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        # Ignore user-defined warp size for gfx9
        warp_size = 32 if 'gfx10' in self.arch or 'gfx11' in self.arch or 'gfx12' in self.arch else 64
        object.__setattr__(self, 'warp_size', warp_size)
        libs = ["ocml", "ockl", "hip", "opencl"]
        for lib in libs:
            extern_libs[lib] = str(os.path.join(default_libdir, f'{lib}.bc'))
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'hip'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def parse_options(self, opts) -> Any:
        args = {'arch': os.getenv("TRITON_OVERRIDE_ARCH", self.target.arch)}

        if "supported_fp8_dtypes" not in opts:
            supported_fp8_dtypes = set(HIPOptions.supported_fp8_dtypes)
            if self.target.arch in ('gfx940', 'gfx941', 'gfx942'):
                supported_fp8_dtypes.update({'fp8e4nv', 'fp8e4b8', 'fp8e5b16'})
            if self.target.arch in ('gfx938', 'gfx946', 'gfx948'):
                supported_fp8_dtypes.update({'fp8e4b8', 'fp8e5b16'})
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"

        if "optimize_epilogue" not in opts:
            args["optimize_epilogue"] = os.getenv("OPTIMIZE_EPILOGUE", "0") == "1"

        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() if k in opts and opts[k] is not None})
        return HIPOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        amd.load_dialects(ctx)

    @staticmethod
    def is_within_2gb(arg):
        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= 2**31 - 1
        if "torch.Tensor" in str(type(arg)) and hasattr(arg, "untyped_storage"):
            return arg.untyped_storage().size() <= 2**31 - 1
        return False

    @staticmethod
    def parse_attr(desc):
        ret = BaseBackend.parse_attr(desc)
        if "S" in desc:
            ret += [["tt.pointer_range", 32]]
        return ret

    @staticmethod
    def get_arg_specialization(arg, ty, **kwargs):
        ret = BaseBackend.get_arg_specialization(arg, ty, **kwargs)
        if ty == "tensor" and HIPBackend.is_within_2gb(arg):
            ret += "S"
        return ret

    @staticmethod
    def path_to_rocm():
        rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
        return rocm_path

    @staticmethod
    def path_to_rocm_lld():
        rocm_path = HIPBackend.path_to_rocm()
        # Check env path for ld.lld
        lld_env_path = os.getenv("TRITON_HIP_LLD_PATH")
        if lld_env_path is not None:
            lld = Path(lld_env_path)
            if lld.is_file():
                return lld
        # Check backend for ld.lld (used for pytorch wheels)
        lld = Path(__file__).parent / "llvm/bin/ld.lld"
        if lld.is_file():
            return lld
        lld = Path(f"{rocm_path}/llvm/bin/ld.lld")
        if lld.is_file():
            return lld
        lld = Path("/usr/bin/ld.lld")
        if lld.is_file():
            return lld
        raise Exception(f"ROCm linker {rocm_path}/llvm/bin/ld.lld not found. Set 'TRITON_HIP_LLD_PATH' to its path.")

    @staticmethod
    def path_to_rocm_clang():
        rocm_path = HIPBackend.path_to_rocm()
        # Check env path for clang
        clang_env_path = os.getenv("TRITON_HIP_CLANG_PATH",
                                    # By default, use clang-18
                                   f"{rocm_path}/llvm/bin/clang-18")
        if clang_env_path is not None:
            clang = Path(clang_env_path)
            if clang.is_file():
                return clang
        clang = Path(f"{rocm_path}/llvm/bin/clang")
        if clang.is_file():
            return clang
        clang = Path("/usr/bin/clang")
        if clang.is_file():
            return clang
        raise Exception(f"ROCm compiler {rocm_path}/llvm/bin/clang not found. Set 'TRITON_HIP_CLANG_PATH' to its path.")

    @staticmethod
    def _get_clang_args(metadata, options):
        arch_args = {
            "gfx928": [
                        "-mllvm=-support-512-vgprs=true",
                      ],
            "gfx936": [
                        "-mllvm=-support-768-vgprs=true",
                      ],
        }
        if options.arch in arch_args:
            options_args = arch_args[options.arch]
        else:
            raise ValueError(f"Unknown arch: {options.arch}")

        if options.sched_latency != 'none':
            sched_latency_args = {
                "mmac5-ds10": ["-mllvm=-enable-latency-hack=true", "-mllvm=-mmac-latency=5", "-mllvm=-ds-load-store-latency=10"],
                "mmac5-ds6" : ["-mllvm=-enable-latency-hack=true", "-mllvm=-mmac-latency=5", "-mllvm=-ds-load-store-latency=6" ],
            }
            if options.sched_latency in sched_latency_args:
                options_args.extend(sched_latency_args[options.sched_latency])
            else:
                raise ValueError(f"Unsupported scheduling latency: {options.sched_latency}")

        if options.instruction_sched_variant == "llvm-iglp-8":
            if (options.num_warps >= 4 and metadata["shared"] <= 32*1024):
                options_args.extend(["-mllvm=-amdgpu-iglp8-advance-sched-group-cnt=2"])
            if options.kpack == 2:
                options_args.extend(["-mllvm=-amdgpu-iglp8-interleave-ds-cnt-per-mfma=1"])

        clang_args = [
            "-target", amd.TARGET_TRIPLE,
            f"-mcpu={options.arch}:xnack-",
            "-mllvm=-check-valu-data-forward-hazards=0",
            "-mllvm=-disable-cluster-lds-memops=true",
            *options_args,
            "-O3",
        ]

        return clang_args

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        pm.run(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        amd.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if options.optimize_epilogue:
            amd.passes.ttgpuir.add_optimize_epilogue(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)

        stream_prefetch = os.getenv("TRITON_HIP_STREAM_PREFETCH", "0") == "1"

        # The `local-prefetch` scheduling variant requires turning on buffer ops.
        if options.instruction_sched_variant == "local-prefetch":
            stream_prefetch = True

        if amd.has_matrix_core_feature(options.arch):
            assert options.num_stages != 0, ("Triton AMD backend pipeliner has been updated. "
                                             "We used to trigger software pipelining with "
                                             "num_stages == 0. Now it will not happen anymore; "
                                             "please update to use num_stages == 2 for "
                                             "equivalent behavior in the past.")
            amd.passes.ttgpuir.add_stream_pipeline(pm, options.num_stages, stream_prefetch)
            passes.common.add_canonicalizer(pm)
        amd.passes.ttgpuir.insert_instruction_sched_hints(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        if amd.has_matrix_core_feature(options.arch):
            amd.passes.ttgpuir.add_reorder_instructions(pm)
            use_block_pingpong = os.getenv("TRITON_HIP_USE_BLOCK_PINGPONG", "0") == "1"
            if use_block_pingpong and options.num_stages == 2:
                amd.passes.ttgpuir.add_block_pingpong(pm)

        # In most cases, buffer ops are beneficial for performance on HCUs.
        use_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "1") == "1"
        if use_buffer_ops:
            amd.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            amd.passes.ttgpuir.add_convert_to_buffer_ops(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_decompose_unsupported_conversions(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)

        passes.ttgpuir.add_allocate_shared_memory(pm)
        ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
        ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
        ##    of the value of kernel arg `allow_flush_denorm`.
        ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
        ##    depends on the value of kernel arg `allow_flush_denorm`.
        ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
        ##    For now it is used as a controller for developers only.
        __HIP_FTZ = True
        amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        amd.passes.ttgpuir.lower_instruction_sched_hints(pm, options.arch, options.num_stages,
                                                         options.instruction_sched_variant)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, __HIP_FTZ)
        pm.run(mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        amd.attach_target_triple(llvm_mod)
        target_features = ''
        if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
            target_features = '+xnack'
        llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, target_features)

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        amd.set_isa_version(llvm_mod, options.arch)
        amd.set_abi_version(llvm_mod, 500)
        amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_daz_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{options.num_warps*options.warp_size}")
        # LLVM AMDGPU backend supports the attribute "amdgpu-waves-per-eu"="<min>[, <max>]".
        # This attribute may be attached to a kernel function definition and is an optimization hint.
        # <min> parameter specifies the requested minimum number of waves per EU, and optional <max> parameter
        # specifies the requested maximum number of waves per EU (must be greater than <min> if specified).
        # If <max> is omitted, then there is no restriction on the maximum number of waves per EU other than
        # the one dictated by the hardware for which the kernel is compiled. Passing 0, 0 as <min>, <max>
        # implies the default behavior (no limits).
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)
        if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
            fns[0].add_fn_target_feature("+xnack")
            fns[0].add_fn_asan_attr()

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        amd.set_all_fn_arg_inreg(fns[0])

        if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
            default_libdir = Path(__file__).parent / 'lib'
            paths = [
                str(default_libdir / 'asanrtl.bc'),
                str(default_libdir / "ocml.bc"),
                str(default_libdir / "ockl.bc")
            ]
            llvm.link_extern_libs(llvm_mod, paths)
        elif options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if amd.need_extern_lib(llvm_mod, name)]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("ttg.shared")

        amd.cleanup_bitcode_metadata(llvm_mod)
        # Disable inlining of print related functions,
        # because inlining of these function could slow down compilation significantly
        amd.disable_print_inline(llvm_mod)
        return str(llvm_mod)

    @staticmethod
    def make_amdgcn(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accomodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]

        # llvm -> hsaco
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".ll", delete=False) as f:
                llir_file = f.name
                f.write(str(src))

            asm_file = tempfile.mktemp(suffix=".amdgcn")

            clang_path = HIPBackend.path_to_rocm_clang()
            clang_args = HIPBackend._get_clang_args(metadata, options)

            # Compile to ASM
            asm_command = [clang_path] + clang_args + [llir_file, "-S", "-o", asm_file]
            subprocess.run(asm_command, check=True, capture_output=True, text=True)

            with open(asm_file, "r") as fd_out:
                amdgcn = fd_out.read()

        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr}")
            raise HSACOError(f"Compilation failed: {e.stderr}") from e
        except IOError as e:
            print(f"File operation failed: {str(e)}")
            raise HSACOError(f"File operation failed: {str(e)}") from e
        finally:
            # Clean up temporary files
            for file in [llir_file, asm_file]:
                if os.path.exists(file):
                    os.remove(file)

        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".s", delete=False) as f:
                asm_file = f.name
                f.write(str(src))

            hsaco_file = tempfile.mktemp(suffix=".hsaco")

            clang_path = HIPBackend.path_to_rocm_clang()
            clang_args = HIPBackend._get_clang_args(metadata,options)

            # Compile to HSACO
            hsaco_command = [clang_path] + clang_args + [asm_file, "-x", "assembler", "-o", hsaco_file]
            subprocess.run(hsaco_command, check=True, capture_output=True, text=True)

            with open(hsaco_file, "rb") as fd_out:
                ret = fd_out.read()

        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr}")
            raise HSACOError(f"Compilation failed: {e.stderr}") from e
        except IOError as e:
            print(f"File operation failed: {str(e)}")
            raise HSACOError(f"File operation failed: {str(e)}") from e

        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([HIPBackend.path_to_rocm_clang(), "--version"], encoding='utf-8')
        return f'{version}-{self.target}'
