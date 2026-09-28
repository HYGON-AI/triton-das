"""HCU WASP configuration, resource planning and cross-stage validation."""
import re

from triton._C.libtriton import ir


def check_options(args):
    if type(args.get("wasp_wdra", True)) is not bool:
        raise ValueError("wasp_wdra must be a bool")
    waves = args.get("wasp_partition_warps", (4, 4, 4))
    if not isinstance(waves, (tuple, list)) or len(waves) not in (2, 3, 4):
        raise ValueError("wasp_partition_warps requires 2, 3, or 4 partition sizes")
    if any(type(n) is not int or n not in (1, 2, 4, 8) for n in waves):
        raise ValueError("WASP requires 1, 2, 4, or 8 waves per partition")
    if sum(waves) > 16:
        raise ValueError("WASP requires at most 16 total waves")
    producers = 2 if len(waves) == 4 else 1
    if len(set(waves[:producers])) != 1 or len(set(waves[producers:])) != 1:
        raise ValueError("WASP partitions of the same role must have equal wave counts")
    if waves[producers] < waves[0]:
        raise ValueError("WASP requires Consumer waves >= Producer waves per partition")
    if args.get("wasp_wdra", True) and any(n != 4 for n in waves):
        raise ValueError("HCU WDRA requires 4 waves per partition")
    args["wasp_partition_warps"] = tuple(waves)
    regs = args.get("wasp_partition_regs")
    if regs is not None:
        if not args.get("wasp_wdra", True):
            raise ValueError("wasp_partition_regs conflicts with wasp_wdra=False")
        if not isinstance(regs, (tuple, list)) or len(regs) != len(waves):
            raise ValueError(f"wasp_partition_regs requires {len(waves)} partition quotas")
        if any(type(n) is not int or n <= 0 for n in regs):
            raise ValueError("wasp_partition_regs must contain positive integers")
        args["wasp_partition_regs"] = tuple(regs)


def configure_wasp(mod, metadata, options):
    """Detect the IR request and select the execution topology once."""
    requested = False
    explicit = False

    def visit(op):
        nonlocal requested, explicit
        if (op.get_name() == "scf.for" and op.has_attr("tt.warp_specialize")) or op.get_name() == "ttg.warp_specialize":
            requested = True
        if op.get_name() == "ttg.warp_specialize":
            explicit = True

    mod.walk(visit)
    metadata["wasp_enabled"] = requested
    metadata["wasp_wdra"] = requested and options.wasp_wdra
    if not requested:
        if options.wasp_partition_regs is not None:
            raise ValueError("wasp_partition_regs requires a warp_specialize=True loop")
        return
    if options.arch != "gfx946":
        raise ValueError(f"HCU WASP is not supported on {options.arch}")
    # WASP owns LDS multi-buffering through LoadMMASpecialization; num_stages
    # is the LDS buffer depth, not the ordinary AMD software-pipeline depth.
    # The current stage/phase and abarrier lowering supports depths 2 and 4.
    if options.num_stages not in (2, 4):
        raise ValueError("WASP requires num_stages in (2, 4)")
    # The current 2P2C plan budgets 2 load groups * 2 (empty/ready) *
    # 2 producer-consumer pairs * num_stages = 8 * num_stages abarrier IDs.
    # Four stages would require 32 IDs, exceeding the hardware limit of 16.
    # This limit belongs to the automatic topology, not the total wave count.
    if not explicit and len(options.wasp_partition_warps) == 4 and options.num_stages != 2:
        raise ValueError("2P2C WASP requires num_stages=2 (abarrier ID limit)")

    b = ir.builder(mod.context)
    if explicit:
        # Preserve Gluon's explicit layouts. OptimizePartitionWarps verifies
        # that each frontend partition matches wasp_partition_warps.
        mod.set_attr("hcu.wasp_explicit_partitions", b.get_bool_attr(True))
    mod.set_attr("hcu.wasp_partition_warps", b.get_dense_i32_array_attr(options.wasp_partition_warps))


def normalize_regs(regs, arch):
    # WDRA hardware constraints: each four-wave partition contributes one wave
    # per SIMD. B = len(regs) is 2/3/4 for an 8/12/16-wave TG.
    # For per-lane VGPR quotas N[i] and S = sum(N[i]):
    #   4 <= N[i] <= 256, N[i] % 4 == 0;
    #   S <= 512 on gfx946 (Wave64 VGPR units per SIMD);
    #   S % (4 * B) == 0, so S / B is integral and also 4-aligned.
    # Compare quotas directly with capacity, without multiplying by 64 lanes
    # or 4 SIMDs. Unused capacity is allowed; infeasible quotas are rejected.
    # WASP currently supports gfx946 only.
    vgpr_size = {"gfx946": 512}.get(arch, 512)
    if len(regs) not in (2, 3, 4) or any(type(n) is not int or n <= 0 for n in regs):
        raise ValueError("WASP requires 2, 3, or 4 positive integer partition quotas")
    # WDRA allocation rules belong here, not in the target capacity query.
    gran, min_regs, max_regs = 4, 4, 256
    values = [((n + gran - 1) // gran) * gran for n in regs]
    if any(n < min_regs or n > max_regs for n in values):
        raise ValueError("WASP partition quota exceeds hardware per-partition limits")
    # Minimal upward adjustment in Load-first order: prefer Producers, then
    # Consumers only when earlier partitions have reached the per-wave limit.
    padding = (-sum(values)) % (gran * len(values))
    for i in range(len(values)):
        extra = min(padding, max_regs - values[i])
        values[i] += extra
        padding -= extra
    if padding or sum(values) > vgpr_size:
        raise ValueError("WASP partition quotas exceed the per-SIMD VGPR budget")
    return tuple(values)


def configure_partitions(mod, metadata, options):
    """Validate final partitions and set quotas before LLVM lowering."""
    # Direct TTGIR bypasses OptimizePartitionWarps. Check each realized tuple
    # before configure_wasp can overwrite the Module request; equal totals do
    # not imply equal partitions (e.g. (4,4) versus (2,2,2,2)).
    def check_partition_warps(op):
        if op.get_name() == "ttg.warp_specialize":
            actual = op.get_dense_i32_array_attr("partitionNumWarps")
            if actual != list(options.wasp_partition_warps):
                raise ValueError(f"WASP partition waves {actual} do not match "
                                 f"wasp_partition_warps={options.wasp_partition_warps}")

    mod.walk(check_partition_warps)
    if "wasp_enabled" not in metadata:  # Direct TTGIR input.
        configure_wasp(mod, metadata, options)
    if not metadata.get("wasp_enabled"):
        return
    total = mod.get_int_attr("ttg.total-num-warps")
    if total != sum(options.wasp_partition_warps):
        raise ValueError(f"WASP launch waves {total} do not match requested topology")
    if not metadata["wasp_wdra"]:
        return
    requested = options.wasp_partition_regs
    metadata["wasp_partition_regs_requested"] = requested
    if requested is not None:
        regs = normalize_regs(requested, options.arch)
        metadata["wasp_partition_regs"] = regs
        mod.set_attr("hcu.wasp_partition_regs", ir.builder(mod.context).get_dense_i32_array_attr(regs))
    # Automatic quotas are already on the module, written by the C++ pass.


def read_partition_registers(gcn):
    """Read the backend's Load-first branch feedback, or None if absent.

    Keep raw demand unaligned; WDRA alignment belongs to quota allocation.
    vgpr_spill_count is a 0/1 spill indicator here, not a measured spill count.
    available_vgprs is the quota used for THIS compilation (including retries),
    not necessarily the user's original option.
    IDs follow Load-first wave groups, not AMDGCN block printing order.
    Spill identity must come from the backend, not required_vgprs > old_quota:
    reported demand may underestimate the actual requirement by up to eight.
    """
    fields = ("BranchSpillInfo", "BranchRealOrSuggestedVgprs", "BranchAvailableVgprs")
    values = {}
    for field in fields:
        matches = re.findall(r"^\s*;\s*" + field + r"\s*:\s*([^\r\n]*)", gcn, re.M)
        if not matches:
            continue
        if len(matches) != 1 or not re.fullmatch(r"\([^()]*\)\s*", matches[0]):
            raise RuntimeError(f"Invalid backend partition register feedback: {field}")
        items = [item.strip() for item in matches[0].strip()[1:-1].split(',')]
        pattern = r"true|false" if field == "BranchSpillInfo" else r"[0-9]+"
        if any(not re.fullmatch(pattern, item) for item in items):
            raise RuntimeError(f"Invalid backend partition register feedback: {field}")
        values[field] = ([int(item == 'true') for item in items] if field == "BranchSpillInfo"
                         else [int(item) for item in items])
    if not values:
        return None
    if (len(values) != len(fields) or len(values[fields[0]]) not in (2, 3, 4) or
            len({len(items) for items in values.values()}) != 1):
        raise RuntimeError("Invalid or incomplete backend partition register feedback")
    return {i: {"required_vgprs": required, "vgpr_spill_count": spill,
                "available_vgprs": available}
            for i, (spill, required, available) in enumerate(zip(*(values[f] for f in fields)))}


def update_llir_regs(src, previous, regs):
    """Update entry initialization and wave-dispatch branch quotas together."""
    init = r'(\bcall\s+void\s+@llvm\.hcu\.wdra\.init\()([^)]*)(\))'
    matches = list(re.finditer(init, src))
    if len(matches) != 1:
        raise RuntimeError("Expected one WDRA initialization in LLVM IR")
    old = tuple(map(int, re.findall(r'i16\s+(\d+)', matches[0][2])))
    if old != tuple(previous) + (0,) * (4 - len(previous)):
        raise RuntimeError("LLVM WDRA initialization does not match partition quotas")

    wave = re.search(r'(%[\w.]+)\s*=.*@llvm\.hcu\.get\.wave\.id\(', src)
    dispatch = re.search(r'switch\s+i32\s+' + re.escape(wave[1]) +
                         r',\s*label\s+%[\w.]+\s*\[([^]]*)\]', src) if wave else None
    if dispatch is None:
        raise RuntimeError("Cannot locate LLVM WASP wave dispatch")
    targets = {int(w): label for w, label in
               re.findall(r'i32\s+(\d+),\s*label\s+%([\w.]+)', dispatch[1])}
    edits = []
    seen = set()
    for part, quota in enumerate(regs):
        labels = {targets.get(w) for w in range(part * 4, part * 4 + 4)}
        if None in labels or len(labels) != 1:
            raise RuntimeError("LLVM WASP partition does not have a unique dispatch target")
        label = labels.pop()
        if label in seen:
            raise RuntimeError("LLVM WASP partitions share a register-setting block")
        seen.add(label)
        block = re.search(r'^' + re.escape(label) + r':[^\n]*\n(.*?)(?=^[\w.]+:|^\})',
                          src, re.M | re.S)
        calls = list(re.finditer(r'@llvm\.hcu\.s\.set\.vgpr\.size\(i16\s+(\d+)\)',
                                 block[1])) if block else []
        if len(calls) != 1 or int(calls[0][1]) != previous[part]:
            raise RuntimeError("LLVM partition register-setting instruction does not match quota")
        start, end = calls[0].span(1)
        edits.append((block.start(1) + start, block.start(1) + end, str(quota)))
    padded = tuple(regs) + (0,) * (4 - len(regs))
    edits.append((*matches[0].span(2), ', '.join(f'i16 {n}' for n in padded)))
    for start, end, value in sorted(edits, reverse=True):
        src = src[:start] + value + src[end:]
    return src


def check_and_adjust_wasp_wdra_regs(gcn, src, metadata, options):
    """Check spill feedback; return None or adjusted quotas for the caller to recompile."""
    if not metadata.get("wasp_wdra", False):
        return None
    # Private scratch is not a spill count. Older backends without branch
    # feedback can still take the global no-spill fast path.
    spills = re.findall(r"(?:\.[sv]gpr_spill_count\s*:\s*|Num[SV]gprSpills\s*[:=]\s*)(\d+)", gcn)
    has_branch_feedback = re.search(r"^\s*;\s*Branch(?:SpillInfo|RealOrSuggestedVgprs|AvailableVgprs)\s*:", gcn, re.M)
    if not spills and not has_branch_feedback:
        raise RuntimeError("Cannot verify HCU WDRA no-spill requirement: backend spill metadata is missing")
    if not any(int(n) for n in spills) and not has_branch_feedback:
        return None
    sgpr_spills = re.findall(r"(?:\.sgpr_spill_count\s*:\s*|NumSgprSpills\s*[:=]\s*)(\d+)", gcn)
    if any(int(n) for n in sgpr_spills):
        raise RuntimeError("HCU WDRA does not allow SGPR spills; changing VGPR quotas cannot repair them")
    # Branch feedback is authoritative for spill identity. Validate its current
    # quotas against LLIR, and retain raw demand even on a successful final run.
    # Both explicit and automatic quotas may be adjusted; options stay intact.
    init = re.findall(r"call\s+void\s+@llvm\.hcu\.wdra\.init\(([^)]*)\)", src)
    if len(init) != 1:
        raise RuntimeError("WASP spill checks require llvm.hcu.wdra.init exactly once in LLIR")
    padded = tuple(map(int, re.findall(r"i16\s+(\d+)", init[0])))
    count = len(options.wasp_partition_warps)
    if len(padded) != 4 or any(padded[count:]):
        raise RuntimeError("LLVM WDRA initialization does not match partition count")
    regs = padded[:count]
    if normalize_regs(regs, options.arch) != regs:
        raise RuntimeError("LLVM WDRA initialization contains unaligned partition quotas")
    metadata["wasp_partition_regs"] = regs
    demand = read_partition_registers(gcn)
    if demand is None:
        raise RuntimeError("Backend per-partition register-demand feedback is missing")
    if (not isinstance(demand, dict) or set(demand) != set(range(len(regs))) or
            any(type(k) is not int for k in demand) or
            any(not isinstance(entry, dict) or
                any(type(entry.get(field)) is not int or entry[field] < 0
                    for field in ("required_vgprs", "vgpr_spill_count"))
                for entry in demand.values())):
        raise RuntimeError("Invalid or incomplete backend partition register feedback")
    if has_branch_feedback:
        available = tuple(demand[i]["available_vgprs"] for i in range(len(regs)))
        if available != regs:
            raise RuntimeError("Backend BranchAvailableVgprs does not match LLVM WDRA quotas")
    required = tuple(demand[i]["required_vgprs"] for i in range(len(regs)))
    spilling = [i for i in range(len(regs)) if demand[i]["vgpr_spill_count"] > 0]
    others = [i for i in range(len(regs)) if demand[i]["vgpr_spill_count"] == 0]
    metadata["wasp_partition_regs_actual"] = required
    if not spilling:
        if any(int(n) for n in spills):
            raise RuntimeError("Backend VGPR spill count has no corresponding spilling partition")
        return None

    # Spill-retry allocation (explicit and automatic quotas use the same policy):
    # 1. Baseline = per-partition 4-aligned demand (minimum 4), NOT
    #    max(old quota, demand). Redistribute from demand, even if that reduces
    #    a spilling partition's old quota. Do not pad the total here: that would
    #    give spare registers to Producers before the spilling partitions.
    #    Reject an infeasible baseline.
    # 2. Only with remaining budget, add up to 8 per spilling partition in
    #    round-robin steps of 4. Never cut another partition's baseline for +8.
    # 3. Give the rest to non-spilling partitions, round-robin in fours, skipping
    #    quotas at 256. If all recipients are capped, leave capacity unused.
    # 4. Trim only optional additions for total alignment, validate, and return
    #    changed quotas. The caller updates init + branch quotas and allows at
    #    most two adjustments (three compilations), using fresh feedback each
    #    time. Demand may shift as allocation changes; even two retries do not
    #    guarantee success, so the final output must also pass the spill check.
    # Example (3 partitions, backend spill counts=(1,0,0), SIMD budget 504):
    # old=(88,208,208), demand=(96,196,200): Producer needs 96 but had only 88.
    # baseline=(96,196,200), sum=492, remaining=12
    # -> Producer +8: (104,196,200), remaining=4
    # -> first non-spilling Consumer +4: (104,200,200), sum=504.
    # Old quotas are used only for LLIR consistency checks and no-change detection,
    # not as allocation floors.
    # No demand > old check: an underestimated demand can still have real spills.
    #
    # Keep sum(N[i]) within the arch's capacity and divisible by 4 * B:
    # gfx946 allows totals 512/504/512 for B = 2/3/4. normalize_regs enforces
    # each quota's 4..256 bounds and 4-alignment before a retry.
    vgpr_size = {"gfx946": 512}.get(options.arch, 512)
    budget = vgpr_size // (4 * len(regs)) * (4 * len(regs))

    # 1. Round each demand independently. Total alignment is handled only after
    # the priority allocation below, so all spare registers remain available.
    values = [max(4, (n + 3) // 4 * 4) for n in required]
    if any(n > 256 for n in values):
        raise ValueError("WASP partition quota exceeds hardware per-partition limits")
    if sum(values) > budget:
        raise ValueError("WASP partition quotas exceed the per-SIMD VGPR budget")

    # 2. With genuine remaining capacity, give spilling partitions up to eight
    # extra VGPRs each, round-robin in fours. No unconditional +8 reservation.
    remaining = budget - sum(values)
    additions = []
    for _ in range(2):
        for i in spilling:
            if remaining >= 4 and values[i] < 256:
                values[i] += 4
                remaining -= 4
                additions.append(i)

    # 3. Distribute the rest among non-spilling partitions, also round-robin.
    while remaining >= 4:
        candidates = [i for i in others if values[i] < 256]
        if not candidates:
            break
        for i in candidates:
            if remaining < 4:
                break
            values[i] += 4
            remaining -= 4
            additions.append(i)

    # If recipients hit their caps (or every partition spills), leave unused
    # capacity. Undo only optional additions to retain total 4*B alignment;
    # never increase a spill margin beyond eight just to align the total.
    for _ in range((sum(values) % (4 * len(regs))) // 4):
        values[additions.pop()] -= 4
    adjusted = normalize_regs(values, options.arch)
    if adjusted == regs:
        raise RuntimeError("HCU WDRA spill feedback produces unchanged quotas; cannot retry")
    return adjusted
