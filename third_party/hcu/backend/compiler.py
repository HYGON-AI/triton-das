from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, amd, hcu, distributed
from triton import knobs
from triton.runtime.errors import HSACOError
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import shutil
import tempfile
import time
import re
import subprocess
import functools
import warnings
import os
from pathlib import Path
 

def get_min_dot_size(target: GPUTarget):
    # We fallback to use FMA and cast arguments if certain configurations is
    # not supported natively by matrix core units.
    return lambda lhs_type, rhs_type: (1, 1, 1)


def is_pingpong_schedule_enabled(arch, use_async_copy):
    return (arch == "gfx942" or (arch == "gfx950" and use_async_copy is True)
            ) if knobs.amd.use_block_pingpong is None else knobs.amd.use_block_pingpong


def is_in_thread_transpose_enabled(arch):
    return (arch == "gfx942") if knobs.amd.use_in_thread_transpose is None else knobs.amd.use_in_thread_transpose


@dataclass(frozen=True)
class HIPOptions:
    num_warps: int = 4
    waves_per_eu: int = 1
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = None
    # We have native support for OCP fp8 variants since CDNA4/RDNA4. For earlier generations,
    # we software emulate the support for them.
    # For leagcy HCU, enable software emulation for fp8e4nv conversions.
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32")
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    matrix_instr_nonkdim: int = 0
    kpack: int = 1
    allow_flush_denorm: bool = False
    max_num_imprecise_acc_default: int = 0
    backend_name: str = 'hip'
    instrumentation_mode: str = ""
    optimize_epilogue: bool = True
    # 0: mfma
    # 1: mmac legacy
    # 2: mmac interleave
    # 3: mmac transpose
    # 4: mmac interleave and transpose
    mmac_layout_force: int = -1

    async_copy_use_single_buffer: bool = True

    # The following option provides hints to the AMDGPU backend regarding instruction scheduling
    # for all `tt.dot` operations in a kernel. The "none" variant preserves the default
    # instruction scheduling of the AMDGPU backend which aims at maximizing occupancy.
    # The option is experimental and may change at any time regarding its semantics and/or may
    # be gone entirely anytime.
    #
    # Current experimental scheduling variants:
    #
    # local-prefetch: implements instruction scheduling similar to the one from the ROCm Composable
    #                 Kernel library. Note, this variant requires the use of buffer load/store ops
    #                 and a special software pipelining style - i.e., 1x LDS and 1x register
    #                 prefetch buffers for each GEMM tile.
    # attention: enables a bunch of optimizations for attention kernels, including:
    #            - iglp 2 and sched.barrier around it
    #            - sink-insts-to-avoid-spills flag to avoid register spills
    # memory-bound-attention: enables custom scheduling strategy in llvm backend,
    #            This option targets special FA variant, which is memory bound and
    #            has a lot of elementwise operations from fused operand dequantizations.
    #            Note that this option is highly experimental,
    #            and will be removed as soon as default sceduler algorithm is fixed.
    #
    # Option allows to set multiple variants divided by commas:
    # schedule_hint="attention,memory-bound-attention"
    schedule_hint: str = 'none'

    # Extend options for HCU(legacy), used for old arch like gfx928, gfx936
    # 1. set scheduling latency for mmac and ds:
    #    - none: use default scheduling latency which equal mmac1-ds5
    #    - see get_options_args() to get more options.
    sched_latency: str = 'none'

    # Extend options for HCU, used for buffer ops with cache swizzle enable or disalbe.
    # mls ignore this and enable default due to better performance.
    buffer_cache_swizzle: bool = False

    # wasp options
    wasp_enabled: bool = False
    wdra_enabled: bool = False
    wasp_num_load_warps: int = None
    wasp_num_mma_warps: int = None
    wdra_num_load_regs: int = None
    wdra_num_mma_regs_main: int = None
    wdra_num_mma_regs_tail: int = None
    # Delay empty abarrier arrive until after mmac (+ SchedBarrier fence).
    # Autotune-friendly; hashed into the kernel cache via HIPOptions.hash().
    empty_arrive_after_mmac: bool = False
    # LLVM HCUMMACCluster mode:
    #   0 — off
    #   1 — enable cluster (+ SchedBarrier before mmac and between A/B loads)
    #   2 — enable cluster + cross-region A/B load analysis (implies 1)
    enable_v_mmac_cluster: int = 0

    def __post_init__(self):
        gfx_major = int(self.arch[3:-2])  # Drop "gfx" prefix and minor/patch number
        warp_size = 32 if gfx_major >= 10 else 64
        object.__setattr__(self, 'warp_size', warp_size)
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

        if (self.arch == 'gfx950') and (self.kpack != 1):
            warnings.warn(
                f"kpack is deprecated starting from gfx950 and will be removed in later releases. So for now kpack = {self.kpack} will be overwritten to 1 to make transitioning easier."
            )
            object.__setattr__(self, 'kpack', 1)

        # default_libdir = Path(__file__).parent / 'lib'
        # HCU toolchain uses this path.
        default_libdir = Path(HIPBackend.path_to_rocm()) / 'amdgcn/bitcode/'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        for lib in ["ocml", "ockl", "hip", "opencl"]:   # Add hip and opencl for HCU toolchain.
            extern_libs[lib] = str(default_libdir / f'{lib}.bc')
        # rocshmem_device_lib = str(default_libdir / 'librocshmem_device.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        # object.__setattr__(self, 'rocshmem_device_lib', rocshmem_device_lib)

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class HIPBackend(BaseBackend):
    instrumentation = None
    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        # Support both 'hcu' (native) and 'hip' (legacy) identifiers.
        # Disambiguation from AMD HIPBackend is handled by make_backend().
        return target.backend in ('hcu', 'hip')

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"
        self.is_hcu = target.arch in ('gfx926', 'gfx928', 'gfx936', 'gfx938', 'gfx92a', 'gfx946')

    def get_target_name(self, options) -> str:
        return f"hip:{options.arch}"

    def parse_options(self, opts) -> Any:
        legacy_sched_variant = opts.pop("instruction_sched_variant", None)
        args = {'arch': knobs.runtime.override_arch or self.target.arch}

        if opts.get("num_ctas", 1) > 1 and not amd.supports_multi_cta_launch(self.target.arch):
            raise ValueError(f"num_ctas > 1 not supported on {self.target.arch}")


        if "supported_fp8_dtypes" not in opts:
            args["supported_fp8_dtypes"] = tuple(sorted(HIPOptions.supported_fp8_dtypes))

        if self.target.arch == 'gfx950':
            deprecated_fp8_dot_operand_dtypes = set(HIPOptions.deprecated_fp8_dot_operand_dtypes)
            deprecated_fp8_dot_operand_dtypes.update({"fp8e5b16", "fp8e4b8"})
            args["deprecated_fp8_dot_operand_dtypes"] = tuple(sorted(deprecated_fp8_dot_operand_dtypes))

        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        if "optimize_epilogue" not in opts:
            args["optimize_epilogue"] = knobs.amd.optimize_epilogue

        # dtk triton compatibility: consume legacy compiler.py options and ignore.
        opts.pop("num_ldmatrixes", None)
        opts.pop("enable_mmacfuse", None)
        opts.pop("reorder_instr", None)

        args.update({k: opts[k] for k in HIPOptions.__dataclass_fields__.keys() \
                     if k in opts and opts[k] is not None})

        cluster = int(args.get("enable_v_mmac_cluster", 0))
        if cluster not in (0, 1, 2):
            raise ValueError(
                f"enable_v_mmac_cluster must be 0, 1, or 2; got {cluster}")
        args["enable_v_mmac_cluster"] = cluster

        # Consume the legacy `instruction_sched_variant` option and map it to
        # `schedule_hint`. If both are present, keep the explicit schedule_hint.
        if args.get("schedule_hint", "none") == "none" and legacy_sched_variant is not None:
            args["schedule_hint"] = legacy_sched_variant

        if "buffer_cache_swizzle" not in opts:
            args["buffer_cache_swizzle"] = knobs.amd.buffer_cache_swizzle

        if args.get("wasp_enabled"):
            if args.get("wdra_enabled"):
                # WDRA topologies (Load-first after OptimizePartitionWarps):
                #   load=4, mma=8 → 1P+2C (12-wave): [Load, MMA_main, MMA_tail]
                #   load=8, mma=8 → 2P+2C (16-wave): [Load0, Load1, MMA0, MMA1]
                #   load=4, mma=4 → [Load, MMA] (no DataPartition)
                assert args["wasp_num_load_warps"] in [4, 8]
                assert args["wasp_num_mma_warps"] in [4, 8]
                if args["wasp_num_load_warps"] == 8:
                    assert args["wasp_num_mma_warps"] == 8, (
                        "wdra 2P+2C requires wasp_num_mma_warps == 8")
            else:
                args.pop("wdra_num_load_regs", None)
                args.pop("wdra_num_mma_regs_main", None)
                args.pop("wdra_num_mma_regs_tail", None)
        else:
            assert not args.get("wdra_enabled"), "wdra_enabled is only supported when wasp_enabled is True"
            args.pop("wasp_num_load_warps", None)
            args.pop("wasp_num_mma_warps", None)
            args.pop("wdra_num_load_regs", None)
            args.pop("wdra_num_mma_regs_main", None)
            args.pop("wdra_num_mma_regs_tail", None)

        return HIPOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": get_min_dot_size(self.target)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice
        from triton.language.extra.hip import librocshmem_device
        # from triton.language.extra.hip import libnvshmem_device

        return {
            "triton.language.extra.libdevice": libdevice,
            # "triton.language.extra.libshmem_device": libnvshmem_device
            "triton.language.extra.libshmem_device": librocshmem_device
        }

    def load_dialects(self, ctx):
        distributed.ir.load_dialects(ctx)
        amd.load_dialects(ctx)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def get_tensor_physical_size(t) -> int: 
        if t.numel() == 0:
            return 0

        offset = sum((size_i - 1) * stride_i for size_i, stride_i in zip(t.shape, t.stride()))
        elements = offset + 1
        return elements * t.element_size()

    @staticmethod
    def is_within_4gb(arg):
        import torch

        if hasattr(arg, "ptr_range"):
            return arg.ptr_range() <= 2**32 - 8
        if isinstance(arg, torch.Tensor) and hasattr(arg, "untyped_storage"):
            return HIPBackend.get_tensor_physical_size(arg) <= 2**32 - 8
        return False

    @staticmethod
    def parse_attr(desc):
        ret = BaseBackend.parse_attr(desc)
        if "S" in desc:
            ret += [["tt.pointer_range", 32]]
        return ret

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        ret = BaseBackend.get_tensor_specialization(arg, **kwargs)
        if knobs.amd.use_buffer_ops and HIPBackend.is_within_4gb(arg):
            ret += "S"
        return ret

    @staticmethod
    def path_to_rocm():
        rocm_path = os.getenv("ROCM_PATH")
        if rocm_path is not None:
            return rocm_path

        candidates = [Path("/opt/rocm"), Path("/opt/dtk")]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)

        return str(candidates[0])

    @staticmethod
    def path_to_rocm_llvm():
        rocm_path = Path(HIPBackend.path_to_rocm())
        name = rocm_path.name.lower()
        if name == "dtk" or "dtk" in name:
            llvm_path = rocm_path / "aillvm"
            if not llvm_path.is_dir():
                print(
                    "\n"
                    "========================================================================\n"
                    "HCU/DTK TOOLCHAIN ERROR: AI LLVM (aillvm) not found\n"
                    "========================================================================\n"
                    f"  DTK root : {rocm_path}\n"
                    f"  Expected : {llvm_path}\n"
                    "========================================================================\n",
                    file=os.sys.stderr,
                    flush=True,
                )
                return rocm_path / "llvm"
            return llvm_path
        return rocm_path / "llvm"

    @staticmethod
    def _archive_failed_compilation(stage, metadata, artifact_paths, clang_path, clang_command, error):
        rocm_path = HIPBackend.path_to_rocm()
        llvm_path = HIPBackend.path_to_rocm_llvm()
        kernel_name = metadata.get("name", "unknown_kernel")
        safe_name = re.sub(r'[^\w.-]', '_', kernel_name)
        context = (
            f"stage      : {stage}\n"
            f"kernel     : {kernel_name}\n"
            f"rocm_path  : {rocm_path}\n"
            f"llvm_path  : {llvm_path}\n"
            f"clang      : {clang_path}\n"
        )
        if clang_command is not None:
            context += f"command    : {' '.join(map(str, clang_command))}\n"
        context += f"error      : {error}\n"
        print(f"[triton-hcu] {stage} failed\n{context}", flush=True)

        dest_dir = (
            Path.home() / ".triton" / "cache" / "failed_kernels"
            / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name}_{stage.replace('->', '_')}"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "context.txt").write_text(context)
        for path in artifact_paths:
            if path and os.path.isfile(path):
                shutil.copy2(path, dest_dir / os.path.basename(path))
        print(f"[triton-hcu] archived failed kernel artifacts to {dest_dir}", flush=True)

    @staticmethod
    def path_to_rocm_lld():
        llvm_path = HIPBackend.path_to_rocm_llvm()
        # Check env path for ld.lld
        lld_env_path = knobs.amd.lld_path
        if lld_env_path is not None:
            lld = Path(lld_env_path)
            if lld.is_file():
                return lld
        # Check backend for ld.lld (used for pytorch wheels)
        lld = Path(__file__).parent / "llvm/bin/ld.lld"
        if lld.is_file():
            return lld
        lld = llvm_path / "bin/ld.lld"
        if lld.is_file():
            return lld
        lld = Path("/usr/bin/ld.lld")
        if lld.is_file():
            return lld
        raise Exception(
            f"ROCm linker not found under {llvm_path}/bin/ld.lld. "
            "Set 'TRITON_HIP_LLD_PATH' to its path."
        )

    @staticmethod
    def path_to_rocm_clang():
        llvm_path = HIPBackend.path_to_rocm_llvm()
        # Check env path for clang
        clang_env_path = os.getenv("TRITON_HIP_CLANG_PATH",
                                    # By default, use clang-18
                                   str(llvm_path / "bin/clang-18"))
        if clang_env_path is not None:
            clang = Path(clang_env_path)
            if clang.is_file():
                return clang
        clang = llvm_path / "bin/clang"
        if clang.is_file():
            return clang
        clang = Path("/usr/bin/clang")
        if clang.is_file():
            return clang
        raise Exception(
            f"ROCm compiler not found under {llvm_path}/bin/clang. "
            "Set 'TRITON_HIP_CLANG_PATH' to its path."
        )

    @staticmethod
    def _get_clang_args(metadata, options):
        arch_args = {
            "gfx928": [
                        "-mllvm=-support-512-vgprs=true",
                      ],
            "gfx936": [
                        "-mllvm=-support-768-vgprs=true",
                      ],
            "gfx938": [
                        "-mllvm=-support-768-vgprs=true",
                      ],
            "gfx946": [
                        "-mllvm=-support-512-vgprs=true",
                      ],
            "gfx92a": [
                        "-mllvm=-support-512-vgprs=true",
                      ],
        }
        if options.arch in arch_args:
            options_args = arch_args[options.arch]
        else:
            raise ValueError(f"Unknown arch: {options.arch}")

        if options.arch != "gfx928":
            # s_barrier not include s_waitcnt vmcnt(0), if not use this, compiler will auto insert s_waitcnt vmcnt(0) with s_barrier
            # The deployed PMD/gem5 clang (1464499b74) predates this flag and rejects
            # it as an unknown argument, aborting the amdgcn stage. Skip it on PMD;
            # the s_barrier waitcnt behavior is not modeled by PMD anyway.
            if not os.environ.get("PMD_PATH"):
                options_args.append("-mllvm=-amdgpu-disable-backoff-barrier=false")


        version_args = {
            "18": [
                "-mllvm=-enable-hcu-approx-func-fp-math=true",
                "-mllvm=-hcu-update-wait-by-reverse-search=true",
            ],
        }
        clang_out = subprocess.check_output([HIPBackend.path_to_rocm_clang(), "--version"])
        match = re.search(r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", clang_out.decode())
        clang_major = match.group("major")
        clang_minor = match.group("minor")
        if clang_major in version_args:
            options_args.extend(version_args[clang_major])

        if options.sched_latency != 'none':
            sched_latency_args = {
                "mmac5-ds10": ["-mllvm=-enable-latency-hack=true", "-mllvm=-mmac-latency=5", "-mllvm=-ds-load-store-latency=10"],
                "mmac5-ds6" : ["-mllvm=-enable-latency-hack=true", "-mllvm=-mmac-latency=5", "-mllvm=-ds-load-store-latency=6" ],
            }
            if options.sched_latency in sched_latency_args:
                options_args.extend(sched_latency_args[options.sched_latency])
            else:
                raise ValueError(f"Unsupported scheduling latency: {options.sched_latency}")

        if options.enable_fp_fusion:
            options_args.extend(["-mllvm=-fp-fuse=fast"])

        clang_args = [
            "-target", amd.TARGET_TRIPLE,
            f"-mcpu={options.arch}:xnack-",
            "-mllvm=-check-valu-data-forward-hazards=0",
            "-mllvm=-disable-cluster-lds-memops=true",
            # Note: when register spill after ds_read_matrix, result is wrong for compiler backend. disable current.
            "-mllvm=-hcu-pre-emit-load-store-opt=false",
            "-mllvm=-vgpr-greedy-alloc-mode=local-wave" if options.wdra_enabled else "",
            # On the model (PMD/gem5), s_trap is not implemented; this tells the
            # HCUInsertWDRAInit pass to skip emitting the s_trap-based wdra init
            # prologue while keeping the rest of the WDRA setup.
            "-mllvm=-turn-off-wdra-trap-handler=true"
            if (options.wdra_enabled and os.environ.get("PMD_PATH")) else "",
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
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        # TritonDistributed Extension
        # distributed.passes.ttir.add_convert_to_ttgpuir_ext(pm, f"hip:{options.arch}", options.num_warps,
        #                                                    options.warp_size, options.num_ctas)
        pm.run(mod, 'make_ttgir_early')
        # Attach sched/empty-arrive knobs as module attrs for WASP + MMAC lowering.
        # Sched barriers are implied by enable_v_mmac_cluster != 0.
        b = ir.builder(mod.context)
        mmac_cluster_on = int(options.enable_v_mmac_cluster) >= 1
        mod.set_attr("hcu.sched_barrier_before_mmac",
                     b.get_bool_attr(mmac_cluster_on))
        mod.set_attr("hcu.empty_arrive_after_mmac",
                     b.get_bool_attr(bool(options.empty_arrive_after_mmac)))
        mod.set_attr("hcu.sched_barrier_between_a_b_loads",
                     b.get_bool_attr(mmac_cluster_on))
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        emuTF32 = False
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        # Record the same per-dot WDRA split intent that DataPartition will
        # execute later. Accelerate/MLS consume it while their inputs are
        # still full-tile IR, so selection is constrained by the eventual
        # per-consumer shape rather than repaired after partitioning.
        if options.wasp_enabled and options.wdra_enabled:
            hcu.passes.ttgpuir.add_prepare_wdra_split_plan(pm, 2)
        hcu.passes.ttgpuir.add_accelerate_matmul(pm, options.arch,
                                                 options.matrix_instr_nonkdim,
                                                 options.kpack,
                                                 options.mmac_layout_force)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        if options.optimize_epilogue:
            amd.passes.ttgpuir.add_optimize_epilogue(pm)
        amd.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)

        hcu.passes.ttgpuir.add_mls_encoding_insertion(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        amd.passes.ttgpuir.add_hoist_layout_conversions(pm)

        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)

        global_prefetch = getattr(knobs.amd, "global_prefetch", 0)
        local_prefetch = getattr(knobs.amd, "local_prefetch", 0)
        use_async_copy = knobs.amd.use_async_copy

        # The `local-prefetch` scheduling variant requires turning on buffer ops.
        if options.schedule_hint == "local-prefetch":
            global_prefetch = 1

        async_copy_single_buffer = options.async_copy_use_single_buffer and (options.num_stages == 2 and not global_prefetch)
        # When WASP is enabled, skip MLS software pipelining (same rationale as
        # skipping amd stream_pipeline): WASP owns multi-buffering / sync for
        # matrix_load_to_local. Running mls_stream_pipeline first injects scf.if
        # epilogue peeling that PartitionLoopsHCU cannot handle.
        if not options.wasp_enabled:
            hcu.passes.ttgpuir.add_mls_stream_pipeline(pm, options.num_stages, global_prefetch, async_copy_single_buffer)

        use_block_pingpong = is_pingpong_schedule_enabled(options.arch, use_async_copy)
        amd.passes.ttgpuir.add_schedule_loops(pm, options.num_stages)

        if not options.wasp_enabled:
            if hasattr(amd.passes.ttgpuir, "add_stream_pipeline"):
                amd.passes.ttgpuir.add_stream_pipeline(
                    pm, options.num_stages, global_prefetch, local_prefetch, use_async_copy, use_block_pingpong
                )
            else:
                amd.passes.ttgpuir.add_pipeline(pm, use_async_copy, use_block_pingpong)
        if use_async_copy:
            amd.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
        passes.common.add_canonicalizer(pm)
        if options.schedule_hint.lower() != "none":
            for hint in options.schedule_hint.split(","):
                amd.passes.ttgpuir.insert_instruction_sched_hints(pm, hint)
        passes.ttgpuir.add_remove_layout_conversions(pm)

        hcu.passes.ttgpuir.add_mls_lowering_pass(pm)

        passes.ttgpuir.add_reduce_data_duplication(pm)
        if is_in_thread_transpose_enabled(options.arch):
            amd.passes.ttgpuir.add_in_thread_transpose(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
        amd.passes.ttgpuir.add_reorder_instructions(pm)
        if use_block_pingpong and options.num_stages > 1:
            amd.passes.ttgpuir.add_block_pingpong(pm, options.num_stages)

        if options.wasp_enabled:
            passes.ttgpuir.add_warp_specialize_hcu(pm, 2, options.wdra_enabled, options.wasp_num_load_warps, options.wasp_num_mma_warps)
            hcu.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack, options.mmac_layout_force)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            if options.optimize_epilogue:
                amd.passes.ttgpuir.add_optimize_epilogue(pm)
            passes.ttgpuir.add_optimize_dot_operands(pm, True)
            amd.passes.ttgpuir.add_hoist_layout_conversions(pm)

        if knobs.amd.use_buffer_ops:
            amd.passes.ttgpuir.add_canonicalize_pointers(pm)
            passes.common.add_canonicalizer(pm)
            hcu.passes.ttgpuir.add_convert_to_buffer_ops(
                pm,
                options.arch,
                knobs.amd.use_buffer_atomics,
                knobs.amd.buffer_ops_analyze_small_tensor_range,
                knobs.amd.emit_buffer_ops_offset_assert,
                options.buffer_cache_swizzle,
            )

        amd.passes.ttgpuir.add_fold_true_cmpi(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if 1:#use_async_copy:
            amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def gluon_to_ttgir(src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
        # custom_lds_size is an experimental parameter that defines amount of LDS available
        # for one thread block. Measured in bytes.
        #
        # If custom_lds_size = 0, pass will consider all LDS is available for one threads block,
        # LDS size is determined by provided arch name.
        custom_lds_size = 0
        amd.passes.ttgpuir.add_optimize_lds_usage(pm, options.arch, custom_lds_size)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        passes.convert.add_index_to_llvmir(pm)

        amd.passes.ttgpuir.add_allocate_shared_memory(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as follows:
        ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
        ##    of the value of kernel arg `allow_flush_denorm`.
        ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
        ##    depends on the value of kernel arg `allow_flush_denorm`.
        ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
        ##    For now it is used as a controller for developers only.
        __HIP_FTZ = True
        # Before TTGIR->LLVMIR: materialize func.* (noalias / attr passthrough)
        # and drop FuncOp/ModuleOp tt.hint.* (incl. mod.flag.*).
        hcu.passes.ttgpuir.add_apply_func_hints_and_clean(pm)
        hcu.passes.ttgpuir.add_to_llvmir(pm, options.arch, __HIP_FTZ)
        # TritonDistributed Extension: distributed -> llvm
        distributed.passes.ttgpuir.amd.add_distributed_to_llvm(pm, options.arch, __HIP_FTZ)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)

        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if options.wasp_enabled:
            hcu.passes.ttgpuir.add_warp_specialize_to_llvm(pm, options.arch, options.wasp_num_load_warps,
                options.wasp_num_mma_warps, options.wdra_enabled, options.wdra_num_load_regs or 0,
                options.wdra_num_mma_regs_main or 0, options.wdra_num_mma_regs_tail or 0)
            passes.convert.add_arith_to_llvmir(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)

        if options.schedule_hint.lower() != "none":
            amd.passes.ttgpuir.lower_instruction_sched_hints(pm, options.arch, options.num_stages)

        # This can not be moved below the di_scope pass
        if HIPBackend.instrumentation:
            HIPBackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, __HIP_FTZ)
        # TritonDistributed Extension: libdevice -> llvm
        distributed.passes.ttgpuir.amd.add_lib_device_to_llvmir(pm, __HIP_FTZ)
        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        amd.attach_target_triple(llvm_mod)
        target_features = ''
        if knobs.compilation.enable_asan:
            target_features = '+xnack'
        llvm.attach_datalayout(llvm_mod, amd.TARGET_TRIPLE, options.arch, target_features)

        # Set various control constants on the LLVM module so that device
        # libraries can resolve references to them.
        amd.set_isa_version(llvm_mod, options.arch)
        amd.set_abi_version(llvm_mod, 500)
        amd.set_bool_control_constant(llvm_mod, "__oclc_finite_only_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_correctly_rounded_sqrt32", True)
        amd.set_bool_control_constant(llvm_mod, "__oclc_unsafe_math_opt", False)
        amd.set_bool_control_constant(llvm_mod, "__oclc_wavefrontsize64", options.warp_size == 64)

        # WarpSpecialize Passes would set this attribute
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        total_num_warps = total_num_warps if total_num_warps is not None else options.num_warps

        # Set kernel attributes first given this may affect later optimizations.
        fns = [fn for fn in llvm_mod.get_functions() if not fn.is_declaration()]
        # If wdra is enabled, this attribute is required by the LLVM backend.
        if options.wdra_enabled:
            fns[0].add_fn_attr("hcu-wdra-waves-per-tg", str(total_num_warps))
        mmac_cluster = int(options.enable_v_mmac_cluster)
        if mmac_cluster >= 1:
            fns[0].add_fn_attr("hcu-enable-v-mmac-cluster", "true")
        if mmac_cluster >= 2:
            fns[0].add_fn_attr(
                "hcu-enable-v-mmac-cluster-cross-region-analysis", "true")
        # The public kernel should be kernel 0.
        fns[0].set_calling_conv(amd.CALLING_CONV_AMDGPU_KERNEL)
        fns[0].add_fn_attr("amdgpu-flat-work-group-size", f"1,{total_num_warps*options.warp_size}")
        if "memory-bound-attention" in options.schedule_hint.split(','):
            fns[0].add_fn_attr("amdgpu-sched-strategy", "iterative-ilp")
        fns[0].add_fn_attr("uniform-work-group-size", "true")
        # LLVM AMDGPU backend supports the attribute "amdgpu-waves-per-eu"="<min>[, <max>]".
        # This attribute may be attached to a kernel function definition and is an optimization hint.
        # <min> parameter specifies the requested minimum number of waves per EU, and optional <max> parameter
        # specifies the requested maximum number of waves per EU (must be >= <min> if specified).
        # If <max> is omitted, then there is no restriction on the maximum number of waves per EU other than
        # the one dictated by the hardware for which the kernel is compiled. Passing 0, 0 as <min>, <max>
        # implies the default behavior (no limits).
        # Specifying N, N forces LLVM to focus on a single register count, simplifies some heuristics
        # and may improve scheduling.
        # FIXME: Keep legacy HCU behavior from Triton 3.1.x, need check triton-llvm waves_per_eu {0} or {0, 0} why not eliminate
        # fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}, {options.waves_per_eu}")
        fns[0].add_fn_attr("amdgpu-waves-per-eu", f"{options.waves_per_eu}")
        denormal_mode = "preserve-sign" if options.allow_flush_denorm else "ieee"
        fns[0].add_fn_attr("denormal-fp-math-f32", denormal_mode)
        if knobs.compilation.enable_asan:
            fns[0].add_fn_target_feature("+xnack")
            fns[0].add_fn_asan_attr()

        # Hint the compiler that we'd like the firmware to set the kernel arguments
        # to user SGPRs so that the kernel does not need to s_load its arguments
        # from memory.
        amd.set_all_fn_arg_inreg(fns[0])
        metadata['use_nvshmem'] = False
        metadata['use_rocshmem'] = False
        for k in llvm_mod.get_functions():
            if "rocshmem" in k.name and k.is_declaration():
                metadata['use_rocshmem'] = True
                break
            elif "nvshmem" in k.name and k.is_declaration():
                metadata['use_nvshmem'] = True
                break

        if knobs.compilation.enable_asan:
            default_libdir = Path(__file__).parent / 'lib'
            paths = [
                str(default_libdir / 'asanrtl.bc'),
                str(default_libdir / "ocml.bc"),
                str(default_libdir / "ockl.bc")
            ]
            llvm.link_extern_libs(llvm_mod, paths)
        elif options.extern_libs:
            paths = [path for (name, path) in options.extern_libs if amd.need_extern_lib(llvm_mod, name)]
            if len(paths) > 0:
                llvm.link_extern_libs(llvm_mod, paths)

        # if options.rocshmem_device_lib and metadata['use_rocshmem']:
        if metadata['use_rocshmem']:
            default_libdir = Path(__file__).parent / 'lib'
            llvm.link_extern_libs(llvm_mod, [str(default_libdir / "librocshmem_device.bc")])
        elif metadata['use_nvshmem']:
            default_libdir = Path(__file__).parent / 'lib'
            llvm.link_extern_libs(llvm_mod, [str(default_libdir / "libnvshmem_device.bc")])

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, options.arch, '', [], options.enable_fp_fusion)

        # Architectures with architected SGPRs store the workgroup id in ttmp9 (X) and ttmp7 (Y[15:0], Z[31:16]).
        # These attributes are used to determine if Z should be masked out when loading Y. They are inferred during
        # optimize_module from calls to @llvm.amdgcn.workgroup.id.x/y/z(). We cannot rely on this because a
        # dispatch dimensions might be used even if there is no program_id() call for it.
        if amd.has_architected_sgprs(options.arch):
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-x")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-y")
            fns[0].remove_fn_attr("amdgpu-no-workgroup-id-z")

        if knobs.amd.scalarize_packed_fops:
            amd.add_scalarize_packed_fops_llvm_pass(fns[0])

        # Get some metadata
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

        # warp-specialization mutates num_warps: the launcher must dispatch all
        # warp groups (load + mma), so use the total warp count, not options.num_warps.
        metadata["num_warps"] = total_num_warps

        amd.cleanup_bitcode_metadata(llvm_mod)
        # Disable inlining of print related functions,
        # because inlining of these function could slow down compilation significantly
        amd.disable_print_inline(llvm_mod)
        return str(llvm_mod)

    ## TODO: [AMD] integrate with rocshmem
    @staticmethod
    def make_amdgcn(src, metadata, options):
        # Find kernel names (there should only be one)
        # We get the name at the last possible step to accommodate `triton.compile`
        # on user-provided LLVM
        names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
        assert len(names) == 1
        metadata["name"] = names[0]
        # llvm -> hsaco
        flags = []
        # The sink-insts-to-avoid-spills flag asks LLVM backend to sink instructions
        # into loops to avoid register spills in the MachineSinking pass, while it
        # can also lead to regression in some cases. But from current observation,
        # the regression is not significant. It would be better to have some heuristics.
        # if options.schedule_hint == 'attention':
        #     flags.append('sink-insts-to-avoid-spills')
        # features = '-real-true16' if 'gfx11' in options.arch else ''
        # amdgcn = llvm.translate_to_asm(src, amd.TARGET_TRIPLE, options.arch, features, flags, options.enable_fp_fusion,
        #                                False)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".ll", delete=False) as f:
                llir_file = f.name
                f.write(str(src))

            asm_file = tempfile.mktemp(suffix=".amdgcn")

            clang_path = HIPBackend.path_to_rocm_clang()
            clang_args = HIPBackend._get_clang_args(metadata, options) + flags

            # Compile to ASM
            asm_command = [clang_path] + clang_args + [llir_file, "-S", "-o", asm_file]
            result = subprocess.run(asm_command, check=True, capture_output=True, text=True)
            if options.wdra_enabled:
                log = result.stdout + result.stderr
                print(log, flush=True)

            with open(asm_file, "r") as fd_out:
                amdgcn = fd_out.read()

        except subprocess.CalledProcessError as e:
            HIPBackend._archive_failed_compilation(
                "llir->amdgcn", metadata, [llir_file, asm_file], clang_path, asm_command, e.stderr)
            print(f"Compilation failed: {e.stderr}")
            raise HSACOError(f"Compilation failed: {e.stderr}") from e
        except IOError as e:
            HIPBackend._archive_failed_compilation(
                "llir->amdgcn", metadata, [llir_file, asm_file], clang_path, asm_command, str(e))
            print(f"File operation failed: {str(e)}")
            raise HSACOError(f"File operation failed: {str(e)}") from e
        finally:
            # Clean up temporary files
            for file in [llir_file, asm_file]:
                if os.path.exists(file):
                    os.remove(file)
        if knobs.amd.dump_amdgcn:
            print("// -----// AMDGCN Dump //----- //")
            print(amdgcn)
        return amdgcn

    @staticmethod
    def make_hsaco(src, metadata, options):
        # target_features = ''
        # if knobs.compilation.enable_asan:
        #     target_features = '+xnack'
        # hsaco = amd.assemble_amdgcn(src, options.arch, target_features)
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
            HIPBackend._archive_failed_compilation(
                "amdgcn->hsaco", metadata, [asm_file, hsaco_file], clang_path, hsaco_command, e.stderr)
            print(f"Compilation failed: {e.stderr}")
            raise HSACOError(f"Compilation failed: {e.stderr}") from e
        except IOError as e:
            HIPBackend._archive_failed_compilation(
                "amdgcn->hsaco", metadata, [asm_file, hsaco_file], clang_path, hsaco_command, str(e))
            print(f"File operation failed: {str(e)}")
            raise HSACOError(f"File operation failed: {str(e)}") from e

        return ret

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        version = subprocess.check_output([HIPBackend.path_to_rocm_clang(), "--version"], encoding='utf-8')
        return f'{version}-{self.target}'
