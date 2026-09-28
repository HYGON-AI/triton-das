"""WASP options, IR planning and spill feedback; no GPU execution required.

These tests explicitly compile for gfx946, independent of the host GPU.
Only an HCU-enabled Triton build is required.
"""
from itertools import product
from pathlib import Path

import pytest
from types import SimpleNamespace
from triton.runtime.jit import JITFunction

libtriton = pytest.importorskip("triton._C.libtriton")
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler support is not built", allow_module_level=True)

from triton._C.libtriton import hcu, ir, passes
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler import HIPBackend
from triton.backends.hcu import wasp


@pytest.fixture
def backend():
    return HIPBackend(GPUTarget("hcu", "gfx946", 64))


def parse_module(path, backend=None):
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend = backend or HIPBackend(GPUTarget("hcu", "gfx946", 64))
    backend.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(path), ctx)
    mod.context = ctx
    return mod


def module(tmp_path, backend, count):
    loops = '\n'.join('''scf.for %i = %lo to %hi step %step {
      scf.yield
    } {tt.warp_specialize}''' for _ in range(count))
    path = tmp_path / 'requests.ttir'
    path.write_text('''module {
      tt.func public @kernel(%lo: index, %hi: index, %step: index) {
      ''' + loops + '''
        tt.return
      }
    }''')
    return parse_module(path, backend)


def test_defaults_and_ir_detection(tmp_path, backend):
    opts = backend.parse_options({})
    assert opts.wasp_partition_warps == (4, 4, 4)
    assert opts.wasp_wdra
    assert opts.wasp_partition_regs is None
    for count in (0, 1):
        opts = backend.parse_options({})
        metadata = {}
        wasp.configure_wasp(module(tmp_path, backend, count), metadata, opts)
        assert metadata['wasp_enabled'] == bool(count)
        assert metadata['wasp_wdra'] == bool(count)
        assert opts.wasp_wdra is True


def test_two_independent_loops_not_rejected_at_detection(tmp_path, backend):
    for wdra in (False, True):
        metadata = {}
        wasp.configure_wasp(module(tmp_path, backend, 2), metadata, backend.parse_options({'wasp_wdra': wdra}))
        assert metadata['wasp_enabled']
        assert metadata['wasp_wdra'] == wdra


@pytest.mark.parametrize('waves', [None, 12, (), (4,), (1, 1, 1, 1, 1),
                                   (0, 4), (-2, 4), (3, 4), (True, 4),
                                   (4.0, 4), (16, 16), (4, 2, 4), (2, 4, 4, 4)])
def test_invalid_partition_warps(backend, waves):
    with pytest.raises(ValueError):
        backend.parse_options(dict(wasp_wdra=False, wasp_partition_warps=waves))


# Representative uniform and mixed tuples; exhaustive legality is below.
@pytest.mark.parametrize('waves', [
    (2, 2), (4, 4), (2, 2, 2), (4, 4, 4), (2, 2, 2, 2), (4, 4, 4, 4),
    (1, 1), (1, 4, 4), (1, 8), (2, 2, 4, 4),
])
def test_partition_tuple_contract(backend, waves):
    options = backend.parse_options(dict(num_warps=8, wasp_wdra=False,
                                         wasp_partition_warps=list(waves)))
    assert options.wasp_partition_warps == waves
    assert options.hash() == backend.parse_options(dict(
        num_warps=8, wasp_wdra=False, wasp_partition_warps=waves)).hash()


@pytest.mark.parametrize('wdra', [False, True])
def test_exact_supported_partition_space(backend, wdra):
    accepted = set()
    for count in (2, 3, 4):
        for waves in product((1, 2, 4, 8), repeat=count):
            try:
                backend.parse_options(dict(wasp_wdra=wdra, wasp_partition_warps=waves))
            except ValueError:
                continue
            accepted.add(waves)
    sizes = (4,) if wdra else (1, 2, 4, 8)
    expected = set()
    for producers, consumers in ((1, 1), (1, 2), (2, 2)):
        for p, c in product(sizes, repeat=2):
            waves = (p,) * producers + (c,) * consumers
            if c >= p and sum(waves) <= 16:
                expected.add(waves)
    assert accepted == expected


def test_removed_total_option_rejected(backend):
    with pytest.raises(ValueError, match='was removed'):
        backend.parse_options({'wasp_total_num_warps': 12})
    with pytest.raises(ValueError, match='was removed'):
        JITFunction._pack_args(SimpleNamespace(params=[]), backend,
                              {'wasp_total_num_warps': 12}, {}, [], None)


@pytest.mark.parametrize('name', ['wasp_enabled', 'wdra_enabled', 'wasp_total_warps',
                                'wasp_consumer_pingpang'])
def test_unknown_options_rejected_by_jit(backend, name):
    # No HCU compatibility layer: old names follow the ordinary JIT error path.
    with pytest.raises(KeyError, match='unrecognised'):
        JITFunction._pack_args(SimpleNamespace(params=[]), backend,
                              {name: True}, {}, [], None)


def test_explicit_quota_contract(backend):
    with pytest.raises(ValueError, match='conflicts'):
        backend.parse_options({'wasp_wdra': False, 'wasp_partition_regs': (88, 144, 140)})
    for regs in ((), (88, 144), (True, 144, 140), (-4, 144, 140), (8.5, 144, 140)):
        with pytest.raises(ValueError):
            backend.parse_options({'wasp_partition_regs': regs})
    assert backend.parse_options({'wasp_partition_regs': [88, 144, 140]}).wasp_partition_regs == (88, 144, 140)


@pytest.mark.parametrize('requested,effective', [
    ((88, 144, 140), (88, 144, 140)),
    ((88, 88, 144, 140), (92, 88, 144, 140)),
    ((88, 140), (92, 140)),
    ((88, 144, 144), (96, 144, 144)),
    ((256, 8, 4), (256, 16, 4)),
    ((256, 80, 80, 76), (256, 84, 80, 76)),
    ((252, 4, 4, 4), (256, 8, 4, 4)),
    ((80, 96, 144, 144), (80, 96, 144, 144)),
    ((85, 141, 139), (88, 144, 140)),
    ((4, 256, 244), (4, 256, 244)),
    ((4, 4, 256), (4, 4, 256)),
])
def test_normalization(requested, effective):
    assert wasp.normalize_regs(requested, "gfx946") == effective
    assert wasp.normalize_regs(effective, "gfx946") == effective


@pytest.mark.parametrize('requested', [(256, 256, 4), (260, 4), (256, 256, 256, 256)])
def test_capacity_rejected(requested):
    with pytest.raises(ValueError):
        wasp.normalize_regs(requested, "gfx946")


@pytest.mark.parametrize('arch', ['gfx946', 'gfx938', 'unknown'])
def test_quota_capacity_defaults_to_512(arch):
    # Capacity fallback does not enable WASP on these other architectures.
    assert wasp.normalize_regs((256, 256), arch) == (256, 256)
    with pytest.raises(ValueError, match='budget'):
        wasp.normalize_regs((256, 256, 4), arch)


def test_no_request_with_explicit_quota(tmp_path, backend):
    with pytest.raises(ValueError, match='requires a warp_specialize'):
        wasp.configure_wasp(module(tmp_path, backend, 0), {},
                     backend.parse_options({'wasp_partition_regs': (88, 144, 140)}))


def test_ordinary_gluon_is_not_wasp(tmp_path, backend):
    metadata = {}
    backend.gluon_to_ttgir(module(tmp_path, backend, 0), metadata, backend.parse_options({}))
    assert not metadata['wasp_enabled']
    assert not metadata['wasp_wdra']


def test_compilation_options_remain_unchanged(tmp_path, backend, monkeypatch):
    from triton.backends.compiler import Language

    original = backend.parse_options({})
    original_hash = original.hash()
    seen = []

    def ttgir(mod, metadata, options):
        wasp.configure_wasp(mod, metadata, options)
        assert options is original
        seen.append(metadata["wasp_wdra"])
        return mod

    def llir(mod, metadata, options):
        assert options is original
        seen.append(metadata["wasp_wdra"])
        return mod

    monkeypatch.setattr(backend, 'make_ttgir', ttgir)
    monkeypatch.setattr(backend, 'make_llir', llir)
    for count in (0, 1):
        stages, metadata = {}, {}
        backend.add_stages(stages, original, Language.TRITON)
        mod = stages['ttgir'](module(tmp_path, backend, count), metadata)
        stages['llir'](mod, metadata)
        assert original.wasp_wdra is True
        assert original.hash() == original_hash
    assert seen == [False, False, True, True]


def test_wdra_explicitly_disabled(tmp_path, backend):
    opts = backend.parse_options({'wasp_wdra': False})
    metadata = {}
    wasp.configure_wasp(module(tmp_path, backend, 1), metadata, opts)
    assert metadata['wasp_enabled']
    assert not opts.wasp_wdra
    assert not metadata['wasp_wdra']


def test_clang_flags_follow_wasp_request_and_user_options(tmp_path, backend):
    for count in (0, 1):
        opts = backend.parse_options({})
        metadata = {}
        wasp.configure_wasp(module(tmp_path, backend, count), metadata, opts)
        args = backend._get_clang_args(metadata, opts)
        assert ('-mllvm=-vgpr-greedy-alloc-mode=local-wave' in args) == bool(count)


def test_wdra_uses_effective_metadata(backend, monkeypatch):
    monkeypatch.setenv('PMD_PATH', '/test/pmd')
    opts = backend.parse_options({})
    for metadata in ({}, {'wasp_wdra': False}, {'wasp_wdra': True}):
        enabled = metadata.get('wasp_wdra', False)
        args = backend._get_clang_args(metadata, opts)
        assert ('-mllvm=-vgpr-greedy-alloc-mode=local-wave' in args) == enabled
        assert ('-mllvm=-turn-off-wdra-trap-handler=true' in args) == enabled
        if not enabled:
            assert wasp.check_and_adjust_wasp_wdra_regs('', '', metadata, opts) is None
    assert opts.wasp_wdra is True


def mock_assembler(monkeypatch, compile_once):
    from pathlib import Path
    from triton.backends.hcu import compiler

    monkeypatch.setattr(HIPBackend, 'path_to_rocm_clang', staticmethod(lambda: '/test/clang'))
    monkeypatch.setattr(HIPBackend, '_get_clang_args', staticmethod(lambda metadata, options: []))
    def run(command, **kwargs):
        llir_path = command[1]
        asm_path = command[command.index('-o') + 1]
        Path(asm_path).write_text(compile_once(Path(llir_path).read_text()))
        return SimpleNamespace(stdout='', stderr='')
    monkeypatch.setattr(compiler.subprocess, 'run', run)


def feedback_llir(regs):
    init = ', '.join(f'i16 {n}' for n in (*regs, *((0,) * (4 - len(regs)))))
    cases = '\n'.join(f'i32 {w}, label %part{w // 4}' for w in range(4 * len(regs)))
    # Reverse block order and allow equal quotas: identity must come from dispatch.
    blocks = '\n'.join(f'''part{i}:
  call void @llvm.hcu.s.set.vgpr.size(i16 {regs[i]})
  br label %exit''' for i in reversed(range(len(regs))))
    return f'''define amdgpu_kernel void @kernel() {{
  call void @llvm.hcu.wdra.init({init})
  %wave = call i32 (...) @llvm.hcu.get.wave.id()
  switch i32 %wave, label %exit [
{cases}
  ]
{blocks}
exit:
  ret void
}}
declare void @llvm.hcu.wdra.init(i16, i16, i16, i16)
declare void @llvm.hcu.s.set.vgpr.size(i16)
'''


def partition_feedback(required, spilling=(0,)):
    return {i: {'required_vgprs': n, 'vgpr_spill_count': int(i in spilling)}
            for i, n in enumerate(required)}


def branch_feedback_asm(regs, required, spilling=()):
    flags = ', '.join('true' if i in spilling else 'false' for i in range(len(regs)))
    return (f'; BranchSpillInfo: ({flags})\n'
            f'; BranchRealOrSuggestedVgprs: ({", ".join(map(str, required))})\n'
            f'; BranchAvailableVgprs: ({", ".join(map(str, regs))})\n')


@pytest.mark.parametrize('regs,required,spilling,expected', [
    ((16, 124, 124), (2, 228, 228), (1, 2), (32, 236, 236)),
    ((4, 4, 248, 256), (1, 1, 256, 243), (2,), (8, 4, 256, 244)),
])
def test_backend_branch_feedback_examples(backend, regs, required, spilling, expected):
    # Exact feedback tuples supplied by the GEMM/FA backend artifacts.
    gcn = branch_feedback_asm(regs, required, spilling)
    demand = wasp.read_partition_registers(gcn)
    assert tuple(demand[i]['required_vgprs'] for i in range(len(regs))) == required
    assert tuple(demand[i]['available_vgprs'] for i in range(len(regs))) == regs
    for i in range(len(regs)):
        assert demand[i]['vgpr_spill_count'] == int(i in spilling)
    opts = backend.parse_options({'wasp_partition_warps': (4,) * len(regs)})
    metadata = {'wasp_wdra': True}
    assert wasp.check_and_adjust_wasp_wdra_regs(gcn, feedback_llir(regs), metadata, opts) == expected
    assert metadata['wasp_partition_regs_actual'] == required


@pytest.mark.parametrize('gcn', [
    '; BranchSpillInfo: (false, true)',
    branch_feedback_asm((4, 252), (1, 240)).replace('(false, false)', '(false, 1)'),
    branch_feedback_asm((4, 252), (1, 240)).replace('(1, 240)', '(1, -1)'),
    branch_feedback_asm((4, 252), (1, 240)).replace('(1, 240)', '(1, 2.5)'),
    branch_feedback_asm((4, 252), (1,)),
    branch_feedback_asm((4,), (1,)),
    branch_feedback_asm((4, 252), (1, 240)) * 2,
    branch_feedback_asm((4, 252), (1, 240)).replace('(1, 240)', '1, 240'),
])
def test_malformed_backend_branch_feedback(gcn):
    with pytest.raises(RuntimeError, match='backend partition register feedback'):
        wasp.read_partition_registers(gcn)


def test_backend_branch_feedback_no_spill_and_consistency(backend):
    regs, required = (32, 236, 236), (2, 231, 229)
    opts = backend.parse_options({})
    metadata = {'wasp_wdra': True}
    gcn = branch_feedback_asm(regs, required)
    assert wasp.check_and_adjust_wasp_wdra_regs(gcn, feedback_llir(regs), metadata, opts) is None
    assert metadata['wasp_partition_regs'] == regs
    assert metadata['wasp_partition_regs_actual'] == required  # Keep unaligned demand.
    with pytest.raises(RuntimeError, match='BranchAvailableVgprs'):
        wasp.check_and_adjust_wasp_wdra_regs(gcn, feedback_llir((16, 244, 244)), metadata, opts)
    with pytest.raises(RuntimeError, match='no corresponding spilling partition'):
        wasp.check_and_adjust_wasp_wdra_regs(gcn + '.vgpr_spill_count: 1', feedback_llir(regs), metadata, opts)
    with pytest.raises(RuntimeError, match='Invalid or incomplete'):
        wasp.check_and_adjust_wasp_wdra_regs(branch_feedback_asm((4, 252), (1, 240)),
                                 feedback_llir(regs), metadata, opts)
    assert wasp.read_partition_registers('.vgpr_spill_count: 0') is None


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('final_spill', [False, True])
def test_real_feedback_format_two_adjustments(backend, monkeypatch, explicit, final_spill):
    quotas = [(16, 124, 124), (32, 236, 236), (16, 244, 244)]
    requirements = [(2, 228, 228), (2, 236, 235), (2, 239, 238)]
    opts = backend.parse_options({'wasp_partition_regs': quotas[0] if explicit else None})
    metadata = {'wasp_wdra': True}
    compiled = []

    def compile_once(src):
        i = len(compiled)
        assert src == feedback_llir(quotas[i])
        compiled.append(src)
        spilling = (1, 2) if i < 2 or final_spill else ()
        # Even if a legacy global counter is zero, branch spill flags must win.
        return branch_feedback_asm(quotas[i], requirements[i], spilling) + '.vgpr_spill_count: 0'

    mock_assembler(monkeypatch, compile_once)
    if final_spill:
        with pytest.raises(RuntimeError, match='after 2 adjustments'):
            backend.make_amdgcn(feedback_llir(quotas[0]), metadata, opts)
    else:
        backend.make_amdgcn(feedback_llir(quotas[0]), metadata, opts)
    assert len(compiled) == 3
    assert metadata['wasp_partition_regs'] == quotas[-1]
    assert metadata['wasp_partition_regs_actual'] == requirements[-1]
    assert opts.wasp_partition_regs == (quotas[0] if explicit else None)


@pytest.mark.parametrize('regs,demands,adjustments,final_spill', [
    ((88, 144), [(92, 148)], [(100, 252)], False),
    ((88, 144, 140), [(92, 148, 144)], [(100, 204, 200)], False),
    ((80, 80, 144, 144), [(84, 84, 148, 148)], [(92, 100, 160, 160)], False),
    ((88, 144, 140), [(92, 144, 140), (104, 196, 196)],
     [(100, 204, 200), (112, 196, 196)], False),
    ((88, 144, 140), [(92, 144, 140), (104, 196, 196), (116, 192, 192)],
     [(100, 204, 200), (112, 196, 196)], True),
])
@pytest.mark.parametrize('explicit', [False, True])
def test_feedback_recompilation(backend, monkeypatch, regs, demands, adjustments, final_spill, explicit):
    opts = backend.parse_options({'wasp_partition_warps': (4,) * len(regs),
                                  'wasp_partition_regs': regs if explicit else None})
    metadata = {'wasp_wdra': True, 'wasp_enabled': True}
    feedback = iter(demands)
    monkeypatch.setattr(wasp, 'read_partition_registers',
                        lambda _: partition_feedback(next(feedback)))
    compiled = []
    def compile_once(src):
        compiled.append(src)
        if len(compiled) > 1:
            assert metadata['wasp_partition_regs'] == adjustments[len(compiled) - 2]
            assert metadata['wasp_partition_regs_actual'] == demands[len(compiled) - 2]
        spill = len(compiled) <= len(adjustments) or final_spill
        return f'.vgpr_spill_count: {int(spill)}'
    mock_assembler(monkeypatch, compile_once)
    if final_spill:
        with pytest.raises(RuntimeError, match='after 2 adjustments'):
            backend.make_amdgcn(feedback_llir(regs), metadata, opts)
    else:
        assert backend.make_amdgcn(feedback_llir(regs), metadata, opts) == '.vgpr_spill_count: 0'
    assert len(compiled) == 1 + len(adjustments) <= 3
    assert next(feedback, None) is None
    assert metadata['wasp_partition_regs'] == adjustments[-1]
    assert metadata['wasp_partition_regs_actual'] == demands[-1]
    assert opts.wasp_partition_regs == (regs if explicit else None)
    # Each retry updates init and all branches, including equal quotas and
    # reverse block order. Never identify a partition by quota value alone.
    for source, quotas in zip(compiled[1:], adjustments):
        init = ', '.join(f'i16 {n}' for n in (*quotas, *((0,) * (4 - len(quotas)))))
        assert f'@llvm.hcu.wdra.init({init})' in source
        for i, n in enumerate(quotas):
            assert f'part{i}:\n  call void @llvm.hcu.s.set.vgpr.size(i16 {n})' in source


@pytest.mark.parametrize('demand', [
    {0: 88, 1: 144, 2: 140},  # Demand alone cannot identify which partition spills.
    partition_feedback((88,)),
    partition_feedback((88, -1, 140)),
    partition_feedback((88, 144, 140, 20)),
    {**partition_feedback((88, 144, 140)), 0: {'required_vgprs': 88}},
    {**partition_feedback((88, 144, 140)), 0: {'required_vgprs': 88, 'vgpr_spill_count': -1}},
    {**partition_feedback((88, 144, 140)), 0: {'required_vgprs': 88, 'vgpr_spill_count': True}},
    [],
])
def test_invalid_feedback(backend, monkeypatch, demand):
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: demand)
    with pytest.raises(RuntimeError, match='Invalid or incomplete'):
        wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', feedback_llir((88, 144, 140)),
                                 {'wasp_wdra': True}, backend.parse_options({}))


def test_explicit_spill_without_demand_does_not_retry(backend, monkeypatch):
    opts = backend.parse_options({'wasp_partition_regs': (88, 144, 140)})
    metadata = {'wasp_wdra': True, 'wasp_enabled': True, 'wasp_partition_regs': (88, 144, 140)}
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: None)
    compiled = []
    def compile_once(src):
        compiled.append(src)
        return '.vgpr_spill_count: 1'
    mock_assembler(monkeypatch, compile_once)
    with pytest.raises(RuntimeError, match='feedback is missing'):
        backend.make_amdgcn(feedback_llir((88, 144, 140)), metadata, opts)
    assert len(compiled) == 1
    assert opts.wasp_partition_regs == (88, 144, 140)
    assert metadata['wasp_partition_regs'] == (88, 144, 140)
    assert 'wasp_partition_regs_actual' not in metadata


@pytest.mark.parametrize('requested', [None, (88, 144, 140)])
def test_no_spill_skips_quota_and_feedback_processing(backend, monkeypatch, requested):
    opts = backend.parse_options({'wasp_partition_regs': requested})
    metadata = {'wasp_wdra': True, 'wasp_enabled': True}
    if requested is not None:
        metadata['wasp_partition_regs'] = requested
    original_metadata = metadata.copy()
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: pytest.fail('No spill: no feedback needed'))
    compiled = []
    def compile_once(src):
        compiled.append(src)
        return '.vgpr_spill_count: 0'
    mock_assembler(monkeypatch, compile_once)
    # No init instruction: the no-spill fast path must not parse quotas either.
    backend.make_amdgcn('define amdgpu_kernel void @kernel() { ret void }', metadata, opts)
    assert len(compiled) == 1
    assert metadata == {**original_metadata, 'name': 'kernel'}
    assert opts.wasp_partition_regs == requested
    assert 'wasp_regs_automatic' not in metadata


def test_feedback_compares_each_partition_not_tuple_order(backend, monkeypatch):
    regs = (88, 144, 140)
    required = (80, 140, 144)  # Tuple compares smaller, but the last quota is insufficient.
    metadata = {'wasp_wdra': True,
                'wasp_partition_regs': regs}
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback(required, (2,)))
    adjusted = wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', feedback_llir(regs),
                                        metadata, backend.parse_options({}))
    assert adjusted is not None
    assert all(quota >= n for quota, n in zip(adjusted, required))
    assert wasp.normalize_regs(adjusted, "gfx946") == adjusted
    assert metadata['wasp_partition_regs_actual'] == required
    # Only the caller updates the effective quotas when recompiling the LLIR.
    assert metadata['wasp_partition_regs'] == regs


def test_feedback_runs_only_in_amdgcn_stage(backend, monkeypatch):
    from triton.backends.compiler import Language
    original = feedback_llir((88, 144, 140))
    monkeypatch.setattr(backend, 'make_llir', lambda src, metadata, options: original)
    compiled = []
    def compile_once(src):
        compiled.append(src)
        return '.vgpr_spill_count: ' + ('1' if len(compiled) == 1 else '0')
    mock_assembler(monkeypatch, compile_once)
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback((92, 144, 140)))
    stages = {}
    backend.add_stages(stages, backend.parse_options({}), Language.TRITON)
    metadata = {'wasp_wdra': True, 'wasp_enabled': True}
    llir = stages['llir'](None, metadata)
    assert llir == original
    assert compiled == []
    assert 'wasp_partition_regs' not in metadata
    assert stages['amdgcn'](llir, metadata) == '.vgpr_spill_count: 0'
    assert len(compiled) == 2
    assert compiled[0] == original and compiled[1] != original
    assert metadata['wasp_partition_regs'] == (100, 204, 200)


def test_spill_requires_init_in_amdgcn_stage(backend, monkeypatch):
    compiled = []
    def compile_once(src):
        compiled.append(src)
        assert 'wasp_partition_regs' not in metadata
        return '.vgpr_spill_count: 1'
    mock_assembler(monkeypatch, compile_once)
    metadata = {'wasp_wdra': True}
    with pytest.raises(RuntimeError, match='require llvm.hcu.wdra.init'):
        backend.make_amdgcn('define amdgpu_kernel void @kernel() { ret void }',
                            metadata, backend.parse_options({}))
    assert len(compiled) == 1


@pytest.mark.parametrize('src,message', [
    (feedback_llir((88, 144, 140)) * 2, 'exactly once'),
    (feedback_llir((88, 144, 140)).replace('i16 0)', 'i16 4)'), 'partition count'),
    (feedback_llir((85, 144, 140)), 'unaligned'),
])
def test_spill_rejects_invalid_llir_quotas(backend, src, message):
    with pytest.raises(RuntimeError, match=message):
        wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', src,
                                 {'wasp_wdra': True}, backend.parse_options({}))


def test_private_scratch_is_not_a_spill(backend):
    opts = backend.parse_options({'wasp_partition_regs': (88, 144, 140)})
    metadata = {'wasp_wdra': True, 'wasp_enabled': True, 'wasp_partition_regs': (88, 144, 140)}
    gcn = '.sgpr_spill_count: 0\n.vgpr_spill_count: 0\n.amdhsa_private_segment_fixed_size 16'
    assert wasp.check_and_adjust_wasp_wdra_regs(gcn, '', metadata, opts) is None
    with pytest.raises(RuntimeError, match='metadata is missing'):
        wasp.check_and_adjust_wasp_wdra_regs('.amdhsa_private_segment_fixed_size 0', '', metadata, opts)


@pytest.mark.parametrize('metadata', [{}, {'wasp_wdra': False}])
def test_amdgcn_skips_wdra_checks_when_disabled(backend, monkeypatch, metadata):
    compiled = []
    def compile_once(src):
        compiled.append(src)
        return 'ordinary assembly'
    mock_assembler(monkeypatch, compile_once)
    def unexpected_check(*args):
        pytest.fail('WDRA checker must not run for ordinary compilation')
    monkeypatch.setattr(wasp, 'check_and_adjust_wasp_wdra_regs', unexpected_check)
    result = backend.make_amdgcn('define amdgpu_kernel void @kernel() { ret void }',
                                metadata, backend.parse_options({}))
    assert result == 'ordinary assembly'
    assert len(compiled) == 1


@pytest.mark.parametrize('requested', [None, (88, 144, 140)])
def test_explicit_quotas_override_cpp_automatic_allocation(tmp_path, backend, monkeypatch, requested):
    opts = backend.parse_options({'wasp_partition_regs': requested})
    metadata = {'wasp_enabled': True, 'wasp_wdra': True}
    mod = module(tmp_path, backend, 0)
    mod.set_attr("ttg.total-num-warps", ir.builder(mod.context).get_int32_attr(12))
    mod.set_attr("hcu.wasp_partition_regs", ir.builder(mod.context).get_dense_i32_array_attr([88, 208, 208]))
    wasp.configure_partitions(mod, metadata, opts)
    expected = "88, 208, 208" if requested is None else "88, 144, 140"
    assert f"hcu.wasp_partition_regs = array<i32: {expected}>" in str(mod)
    assert 'wasp_regs_automatic' not in metadata


@pytest.mark.parametrize('regs, demand', [
    ((88, 208, 208), (96, 196, 200)),
    ((64, 80, 184, 184), (72, 88, 172, 176)),
])
def test_feedback_redistributes_from_demand_when_initial_budget_is_full(backend, monkeypatch, regs, demand):
    metadata = {'wasp_wdra': True,
                'wasp_partition_regs': regs}
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback(demand))
    opts = backend.parse_options({'wasp_partition_warps': (4,) * len(regs)})
    adjusted = wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', feedback_llir(regs), metadata, opts)
    assert sum(adjusted) == sum(regs)
    assert all(quota >= required for quota, required in zip(adjusted, demand))
    assert wasp.normalize_regs(adjusted, "gfx946") == adjusted
    assert any(new < old for new, old in zip(adjusted, regs))
    metadata['wasp_partition_regs'] = adjusted
    assert wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 0', feedback_llir(adjusted), metadata, opts) is None


@pytest.mark.parametrize('spill_field', ['.vgpr_spill_count: 1', 'NumVgprSpills: 1', 'NumVgprSpills = 1'])
def test_spill_with_underestimated_demand_gets_optional_margin(backend, monkeypatch, spill_field):
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback((80, 140, 140)))
    metadata = {'wasp_wdra': True}
    adjusted = wasp.check_and_adjust_wasp_wdra_regs(spill_field, feedback_llir((88, 144, 140)),
                                        metadata, backend.parse_options({}))
    assert adjusted == (88, 208, 208)
    assert metadata['wasp_partition_regs_actual'] == (80, 140, 140)


@pytest.mark.parametrize('spill_field', ['.sgpr_spill_count: 1', 'NumSgprSpills: 1', 'NumSgprSpills = 1'])
def test_sgpr_spill_does_not_adjust_vgpr_quotas(backend, monkeypatch, spill_field):
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: pytest.fail('SGPR spill is not a VGPR quota issue'))
    with pytest.raises(RuntimeError, match='SGPR spills'):
        wasp.check_and_adjust_wasp_wdra_regs(spill_field, '', {'wasp_wdra': True}, backend.parse_options({}))


@pytest.mark.parametrize('regs,required,spilling,expected', [
    # Total-alignment padding must not consume the spilling Consumer's margin.
    ((96, 192, 216), (96, 196, 204), (1,), (96, 204, 204)),
    # A spilling Producer also gets at most +8, not padding followed by +8.
    ((88, 144, 140), (92, 144, 140), (0,), (100, 204, 200)),
    # Extra budget goes first to the spilling Consumer, not always Producer.
    ((88, 144, 140), (88, 152, 140), (1,), (148, 160, 196)),
    # Multiple spills: round-robin up to eight each, then the other partition.
    ((88, 144, 140), (88, 144, 140), (0, 2), (96, 248, 148)),
    # Only four per spill available after satisfying all reported demands.
    ((88, 200, 204), (88, 200, 204), (0, 1, 2), (92, 204, 208)),
    # Saturated Producer: skip it when distributing the non-spill remainder.
    ((256, 64, 64), (256, 64, 68), (2,), (256, 172, 76)),
    # Spilling partition at 252 can receive only four; others take the rest.
    ((252, 120, 120), (252, 120, 120), (0,), (256, 124, 124)),
    # All partitions spill: margins capped at eight, unused budget is allowed.
    ((88, 144, 140), (88, 144, 140), (0, 1, 2), (96, 152, 148)),
    # A formerly full budget does not prevent extra margin above the new demand.
    ((88, 208, 208), (96, 196, 200), (0,), (104, 200, 200)),
    # Demand fills the budget: even a spilling partition may shrink vs. old.
    ((88, 208, 208), (80, 212, 212), (0,), (80, 212, 212)),
    # Zero/unaligned reported demand still receives hardware minimum/alignment.
    ((88, 144, 140), (0, 0, 0), (0,), (12, 248, 244)),
    ((88, 144, 140), (85, 141, 139), (0,), (96, 208, 200)),
])
def test_spill_allocation_priorities(backend, monkeypatch, regs, required, spilling, expected):
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback(required, spilling))
    opts = backend.parse_options({'wasp_partition_warps': (4,) * len(regs)})
    adjusted = wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', feedback_llir(regs),
                                        {'wasp_wdra': True}, opts)
    assert adjusted == expected
    assert wasp.normalize_regs(adjusted, opts.arch) == adjusted
    assert all(n >= need for n, need in zip(adjusted, required))
    for i in spilling:
        baseline = max(4, (required[i] + 3) // 4 * 4)
        assert adjusted[i] <= baseline + 8


@pytest.mark.parametrize('regs,required,spilling,message', [
    ((88, 208, 208), (88, 208, 208), (0,), 'unchanged quotas'),
    ((88, 208, 208), (88, 208, 208), (), 'no corresponding spilling partition'),
    ((88, 208, 208), (96, 208, 208), (0,), 'budget'),
    ((88, 208, 208), (260, 100, 100), (0,), 'per-partition limits'),
])
def test_spill_allocation_rejects_unusable_feedback(backend, monkeypatch, regs, required, spilling, message):
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback(required, spilling))
    with pytest.raises((RuntimeError, ValueError), match=message):
        wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', feedback_llir(regs),
                                 {'wasp_wdra': True}, backend.parse_options({}))


@pytest.mark.parametrize('regs', [(88, 144, 140), (120, 192, 192)])
@pytest.mark.parametrize('explicit', [False, True])
def test_spill_baseline_does_not_depend_on_old_quotas(backend, monkeypatch, regs, explicit):
    # Same demand and spill identity must give the same new quotas, including
    # when the old spilling quota is larger than the new demand plus margin.
    demand = (92, 140, 140)
    monkeypatch.setattr(wasp, 'read_partition_registers', lambda _: partition_feedback(demand))
    options = backend.parse_options({'wasp_partition_regs': regs if explicit else None})
    metadata = {'wasp_wdra': True}
    adjusted = wasp.check_and_adjust_wasp_wdra_regs('.vgpr_spill_count: 1', feedback_llir(regs), metadata, options)
    assert adjusted == (100, 204, 200)
    assert metadata['wasp_partition_regs'] == regs  # Old values for LLIR patch validation.
    assert metadata['wasp_partition_regs_actual'] == demand
    assert options.wasp_partition_regs == (regs if explicit else None)


# Split planning and explicit-partition contracts.

@pytest.fixture
def empty_module(tmp_path):
    path = tmp_path / "empty.ttgir"
    path.write_text('''module attributes {
        "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
        ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32
    } {}''')
    return parse_module(path)


@pytest.mark.parametrize("wdra", [False, True])
@pytest.mark.parametrize("waves", [None, (), (4, 2, 2), (8, 8, 8)])
def test_missing_or_invalid_topology_rejected(empty_module, wdra, waves, capfd):
    if waves is not None:
        empty_module.set_attr("hcu.wasp_partition_warps", ir.builder(empty_module.context).get_dense_i32_array_attr(waves))
    pm = ir.pass_manager(empty_module.context)
    passes.ttgpuir.add_warp_specialize_hcu(pm, 2, wdra)
    with pytest.raises(RuntimeError):
        pm.run(empty_module, "test_invalid_wasp_topology")
    assert "missing or invalid hcu.wasp_partition_warps" in capfd.readouterr().err


def test_prepare_wasp_split_plan():
    root = Path(__file__).resolve().parents[3]
    mod = parse_module(root / "test/TritonGPU/hcu/Inputs/hcu-gemm-wasp-wdra-pre.mlir")
    pm = ir.pass_manager(mod.context)
    hcu.passes.ttgpuir.add_prepare_wasp_split_plan(pm, 2)
    pm.run(mod, "test_prepare_wasp_split_plan")
    result = str(mod)
    assert "hcu.wasp_split = array<i32: 0, 2, 0>" in result
    assert "hcu.wasp_split_policy = 2 : i32" in result
    assert "hcu.wdra_split" not in result


def partition_module(tmp_path, waves=(4, 4, 4)):
    path = tmp_path / "partitions.mlir"
    partitions = '\n'.join(f'partition{i}() num_warps({n}) {{ ttg.warp_return }}'
                           for i, n in enumerate(waves))
    path.write_text('''module attributes {
      "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
      ttg.target = "hip:gfx946", "ttg.threads-per-warp" = 64 : i32
    } {
      tt.func @test() {
        ttg.warp_specialize()
        default { ttg.warp_yield }
        ''' + partitions + '''
        : () -> ()
        tt.return
      }
    }''')
    return parse_module(path)


@pytest.mark.parametrize("actual,requested", [
    ((2, 2, 2, 2), (4, 4)),
    ((4, 4), (2, 2, 2, 2)),
    ((2, 2, 2, 2), (2, 2, 2, 2)),
    ((4, 4), (4, 4)),
])
def test_direct_ttgir_checks_each_partition_not_total(tmp_path, actual, requested):
    mod = partition_module(tmp_path, actual)
    b = ir.builder(mod.context)
    mod.set_attr("ttg.total-num-warps", b.get_int32_attr(sum(actual)))
    mod.set_attr("hcu.wasp_partition_warps", b.get_dense_i32_array_attr(actual))
    options = HIPBackend(GPUTarget("hcu", "gfx946", 64)).parse_options(
        {"wasp_partition_warps": requested, "wasp_wdra": False})
    metadata = {}
    if actual != requested:
        original = str(mod)
        with pytest.raises(ValueError, match="do not match wasp_partition_warps"):
            wasp.configure_partitions(mod, metadata, options)
        assert str(mod) == original
        assert metadata == {}
    else:
        wasp.configure_partitions(mod, metadata, options)
        assert metadata["wasp_enabled"]
        assert not metadata["wasp_wdra"]


@pytest.mark.parametrize("explicit,with_tuple", [(False, True), (True, True), (True, False)])
def test_partition_pass_uses_tuple_without_fallback(tmp_path, capfd, explicit, with_tuple):
    mod = partition_module(tmp_path)
    b = ir.builder(mod.context)
    if explicit:
        mod.set_attr("hcu.wasp_explicit_partitions", b.get_bool_attr(True))
    if with_tuple:
        mod.set_attr("hcu.wasp_partition_warps", b.get_dense_i32_array_attr([2, 2, 2]))
    pm = ir.pass_manager(mod.context)
    hcu.passes.ttgpuir.add_optimize_partition_warps(pm, False)
    if explicit:
        with pytest.raises(RuntimeError):
            pm.run(mod, "test_partition_tuple")
        expected = ("do not match wasp_partition_warps" if with_tuple else
                    "missing or invalid hcu.wasp_partition_warps")
        assert expected in capfd.readouterr().err
        return
    pm.run(mod, "test_partition_tuple")
    assert mod.get_int_attr("ttg.total-num-warps") == 6
    assert "warpGroupStartIds = array<i32: 0, 2, 4>" in str(mod)


@pytest.mark.parametrize("waves", [
    (0, 1, 1), (2, 4, 8), (8, 2, 2), (8, 8, 8), (4, 2, 2), (8, 4, 4),
])
def test_direct_partition_pass_rejects_out_of_scope_waves(tmp_path, capfd, waves):
    mod = partition_module(tmp_path)
    mod.set_attr("hcu.wasp_partition_warps", ir.builder(mod.context).get_dense_i32_array_attr(waves))
    pm = ir.pass_manager(mod.context)
    hcu.passes.ttgpuir.add_optimize_partition_warps(pm, False)
    with pytest.raises(RuntimeError):
        pm.run(mod, "test_bounded_partition_tuple")
    assert "missing or invalid hcu.wasp_partition_warps" in capfd.readouterr().err


@pytest.mark.parametrize("waves", [(2, 2, 2), (4, 8)])
def test_direct_partition_pass_rejects_non_four_wave_wdra(tmp_path, capfd, waves):
    mod = partition_module(tmp_path, waves)
    mod.set_attr("hcu.wasp_partition_warps", ir.builder(mod.context).get_dense_i32_array_attr(waves))
    pm = ir.pass_manager(mod.context)
    hcu.passes.ttgpuir.add_optimize_partition_warps(pm, True)
    with pytest.raises(RuntimeError):
        pm.run(mod, "test_two_wave_wdra")
    assert "HCU WDRA requires 4 waves per partition" in capfd.readouterr().err


@pytest.mark.parametrize("waves", [(1, 4, 4), (1, 8), (2, 2, 4, 4), (1, 1)])
def test_mixed_partition_starts(tmp_path, waves):
    mod = partition_module(tmp_path, waves)
    mod.set_attr("hcu.wasp_partition_warps", ir.builder(mod.context).get_dense_i32_array_attr(waves))
    pm = ir.pass_manager(mod.context)
    hcu.passes.ttgpuir.add_optimize_partition_warps(pm, False)
    pm.run(mod, "test_mixed_partition_starts")
    starts = ', '.join(str(sum(waves[:i])) for i in range(len(waves)))
    assert f"warpGroupStartIds = array<i32: {starts}>" in str(mod)
    assert mod.get_int_attr("ttg.total-num-warps") == sum(waves)


@pytest.mark.parametrize("waves,extent", [
    ((2, 2, 2), 64), ((2, 2, 2, 2), 64), ((4, 4, 4, 4), 64),
    ((4, 4, 4), 64), ((4, 4), 16), ((2, 2), 64),
    ((2, 2, 2), 16), ((4, 4, 4), 16),
    ((2, 2, 2, 2), 16), ((4, 4, 4, 4), 16),
])
def test_early_budget_plan(tmp_path, capfd, backend, waves, extent):
    kind = len(waves) - 2
    path = tmp_path / "plan.mlir"
    path.write_text(f'''module {{
      tt.func @test(%lo: index, %hi: index, %step: index,
                   %a: tensor<{extent}x32xf16>, %b: tensor<32x{extent}xf16>,
                   %c: tensor<{extent}x{extent}xf32>) {{
        scf.for %i = %lo to %hi step %step {{
          %d = tt.dot %a, %b, %c : tensor<{extent}x32xf16> *
               tensor<32x{extent}xf16> -> tensor<{extent}x{extent}xf32>
          scf.yield
        }} {{tt.warp_specialize}}
        tt.return
      }}
    }}''')
    mod = parse_module(path, backend)
    options = backend.parse_options(dict(num_warps=4, wasp_wdra=False,
                                         num_stages=2,
                                         wasp_partition_warps=waves))
    wasp.configure_wasp(mod, {}, options)
    pm = ir.pass_manager(mod.context)
    hcu.passes.ttgpuir.add_prepare_wasp_split_plan(pm, 2)
    if kind != 0 and extent == 16:
        with pytest.raises(RuntimeError):
            pm.run(mod, "test_unsplittable_explicit_plan")
        assert "CTA tile cannot be split" in capfd.readouterr().err
        assert f"hcu.wasp_partition_warps = array<i32: {', '.join(map(str, waves))}>" in str(mod)
        return
    pm.run(mod, "test_early_budget_plan")
    assert f"hcu.wasp_partition_warps = array<i32: {', '.join(map(str, waves))}>" in str(mod)
    assert "hcu.wasp_topo" not in str(mod)
    assert ("hcu.wasp_split =" in str(mod)) == (kind != 0)
