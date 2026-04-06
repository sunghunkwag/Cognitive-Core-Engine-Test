"""
Task benchmark and strict structural detector for the OMEGA_FORGE engine.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    ExecutionState,
    ProgramGenome,
)
from cognitive_core_engine.omega_forge.cfg import ControlFlowGraph
from cognitive_core_engine.omega_forge.vm import VirtualMachine


class TaskBenchmark:
    """Evaluates genomes against practical computational tasks."""

    TASKS = [
        # (name, inputs, expected_output_location, expected_value)
        ("SUM_SIMPLE", [1.0, 2.0, 3.0, 4.0, 5.0], "reg0", 15.0),
        ("SUM_SMALL", [2.0, 3.0, 5.0], "reg0", 10.0),
        ("MAX_FIND", [3.0, 7.0, 2.0, 9.0, 1.0], "reg0", 9.0),
        ("COUNT", [1.0, 1.0, 1.0, 1.0], "reg0", 4.0),
        ("DOUBLE_FIRST", [5.0, 0.0, 0.0, 0.0], "mem0", 10.0),
    ]

    @staticmethod
    def evaluate(genome: "ProgramGenome", vm: "VirtualMachine") -> float:
        """Returns task score 0.0-1.0 based on practical task performance."""
        passed = 0
        total = len(TaskBenchmark.TASKS)

        for name, inputs, out_loc, expected in TaskBenchmark.TASKS:
            try:
                st = vm.execute(genome, inputs)
                if out_loc == "reg0":
                    result = st.regs[0]
                elif out_loc == "mem0":
                    result = st.memory.get(0, 0.0)
                else:
                    result = 0.0

                # Check if result matches expected (with tolerance)
                if abs(result - expected) < 0.01:
                    passed += 1
                # Partial credit for being close
                elif abs(result - expected) < expected * 0.1:
                    passed += 0.5
            except:
                pass

        return passed / total


# ==============================================================================
# 6) Detector + evidence writer
# ==============================================================================

@dataclass
class DetectorParams:
    # Target: 0.5-5% successes. Use a curriculum so the search has a gradient early,
    # then harden constraints to avoid "linear cheats".
    K_initial: int = 6               # strict CFG edit distance (post-warmup)
    L_initial: int = 10              # strict active subseq length (post-warmup)
    C_coverage: float = 0.55         # min coverage (post-warmup)
    f_rarity: float = 0.001          # rarity threshold (post-warmup)
    N_repro: int = 4                 # reproducibility trials

    require_both: bool = True        # strict mode requires CFG + subseq
    min_loops: int = 2               # STRICT: Require at least 2 loops (Multi-Stage)
    min_scc: int = 2                 # STRICT: Require at least 2 SCCs (Complex Topology)

    allow_cfg_variants: int = 2      # reproducibility CFG variants
    max_cov_span: float = 0.30       # reproducibility coverage stability
    max_loop_span: int = 5           # reproducibility loop stability

    # Warmup curriculum (first warmup_gens generations)
    warmup_gens: int = 100
    warmup_K: int = 3
    warmup_L: int = 8
    warmup_cov: float = 0.45
    warmup_require_both: bool = True # Strict warmup
    warmup_min_loops: int = 1        # Ban linear code even in warmup
    warmup_min_scc: int = 1          # Ban acyclic graphs even in warmup


class StrictStructuralDetector:
    def __init__(self, params: Optional[DetectorParams] = None) -> None:
        self.p = params or DetectorParams()
        self.parent_cfgs: Dict[str, ControlFlowGraph] = {}
        self.subseq_counts: Counter = Counter()
        self.subseq_total: int = 0
        self.seen_success_hashes: Set[str] = set()

    def _in_warmup(self, gen: int) -> bool:
        return gen <= self.p.warmup_gens

    def _K(self, gen: int) -> int:
        # Curriculum: easier early, strict later.
        if self._in_warmup(gen):
            return max(1, int(self.p.warmup_K))
        return max(3, int(self.p.K_initial))

    def _L(self, gen: int) -> int:
        if self._in_warmup(gen):
            return max(4, int(self.p.warmup_L))
        return max(6, int(self.p.L_initial))

    def _anti_cheat(self, st: ExecutionState, code_len: int, gen: int) -> Tuple[bool, str]:
        if st.error:
            return False, f"ERR:{st.error}"
        if not st.halted_cleanly:
            return False, "DIRTY_HALT"
        cov = st.coverage(code_len)
        if cov < self.p.C_coverage:
            return False, f"LOW_COVERAGE:{cov:.3f}"
        min_loops = self.p.warmup_min_loops if self._in_warmup(gen) else self.p.min_loops
        if st.loops_count < min_loops:
            return False, "NO_LOOPS"
        return True, f"ANTI_OK cov={cov:.3f} loops={st.loops_count}"

    def _repro(self, genome: ProgramGenome, vm: VirtualMachine) -> Tuple[bool, str]:
        cfgs: List[str] = []
        covs: List[float] = []
        loops: List[int] = []
        fixed_inputs = [
            [0.0]*8,
            [1.0]*8,
            [2.0]*8,
            [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],
        ]
        for i in range(self.p.N_repro):
            inputs = fixed_inputs[i % len(fixed_inputs)]
            st = vm.execute(genome, inputs)
            cfgs.append(ControlFlowGraph.from_trace(st.trace, len(genome.instructions)).canonical_hash())
            covs.append(st.coverage(len(genome.instructions)))
            loops.append(st.loops_count)

        if len(set(cfgs)) > self.p.allow_cfg_variants:
            return False, "CFG_UNSTABLE"
        if max(covs) - min(covs) > self.p.max_cov_span:
            return False, "COV_UNSTABLE"
        if max(loops) - min(loops) > self.p.max_loop_span:
            return False, "LOOP_UNSTABLE"
        return True, f"REPRO_OK N={self.p.N_repro}"

    def evaluate(
        self,
        genome: ProgramGenome,
        parent: Optional[ProgramGenome],
        st: ExecutionState,
        vm: VirtualMachine,
        generation: int,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        reasons: List[str] = []
        diag: Dict[str, Any] = {}

        ok, msg = self._anti_cheat(st, len(genome.instructions), generation)
        if not ok:
            return False, [f"ANTI_FAIL:{msg}"], diag
        reasons.append(msg)

        cfg = ControlFlowGraph.from_trace(st.trace, len(genome.instructions))
        diag["cfg_hash"] = cfg.canonical_hash()
        # Track CFG for every genome so children can be compared against their parents (prevents deadlock)
        self.parent_cfgs[genome.gid] = cfg
        p_cfg = self.parent_cfgs.get(parent.gid) if parent else None
        if p_cfg is None and parent is not None:
            # Fallback: compute parent's CFG directly (robust even if parent was never a success)
            pst = vm.execute(parent, [1.0] * 8)
            p_cfg = ControlFlowGraph.from_trace(pst.trace, len(parent.instructions))

        cfg_ok = False
        cfg_msg = "CFG_NO_PARENT"
        if p_cfg is not None:
            dist = cfg.edit_distance_to(p_cfg)
            K = self._K(generation)
            cfg_ok = dist >= K
            cfg_msg = f"CFG dist={dist} K={K}"
            diag["cfg_dist"] = dist
        else:
            diag["cfg_dist"] = None

        scc_n = len(cfg.sccs())
        diag["scc_n"] = scc_n
        min_scc = self.p.warmup_min_scc if self._in_warmup(generation) else self.p.min_scc
        if scc_n < min_scc:
            cfg_ok = False
            cfg_msg = "CFG_NO_SCC"

        # subsequence novelty (only executed pcs, contiguous window in instruction index-space)
        L = self._L(generation)
        ops = genome.op_sequence()
        active: List[Tuple[str, ...]] = []
        visited = st.visited_pcs
        for i in range(0, max(0, len(ops) - L + 1)):
            window_pcs = set(range(i, i + L))
            if window_pcs.issubset(visited):
                active.append(tuple(ops[i : i + L]))

        subseq_ok = False
        subseq_msg = "SUBSEQ_NONE"
        if active:
            # rarity by empirical frequency in archive
            for seq in active:
                freq = (self.subseq_counts.get(seq, 0) / max(1, self.subseq_total))
                if freq < self.p.f_rarity:
                    subseq_ok = True
                    subseq_msg = f"SUBSEQ rarity={freq:.6f} L={L}"
                    # Defer archive updates until AFTER full success (CFG+SUBSEQ+REPRO+UNIQUENESS),
                    # otherwise near-misses rapidly poison rarity and can suppress discovery.
                    diag["_candidate_subseq"] = list(seq)
                    diag["subseq"] = list(seq)
                    diag["subseq_freq"] = freq
                    break
        diag["active_subseq_windows"] = len(active)

        # require both or at least one
        require_both = self.p.warmup_require_both if self._in_warmup(generation) else self.p.require_both
        if require_both:
            if not (cfg_ok and subseq_ok and parent is not None):
                return False, [f"REQUIRE_BOTH_FAIL cfg={cfg_ok}({cfg_msg}) subseq={subseq_ok}({subseq_msg})"], diag
        else:
            if not (cfg_ok or subseq_ok):
                return False, [f"NO_STRUCT_CHANGE {cfg_msg}; {subseq_msg}"], diag

        reasons.append(cfg_msg if cfg_ok else cfg_msg)
        reasons.append(subseq_msg if subseq_ok else subseq_msg)

        # reproducibility
        r_ok, r_msg = self._repro(genome, vm)
        if not r_ok:
            return False, [f"REPRO_FAIL:{r_msg}"], diag
        reasons.append(r_msg)

        # global uniqueness on successes (prevents repeated printing of same "success")
        succ_hash = cfg.canonical_hash() + "|" + genome.code_hash()
        if succ_hash in self.seen_success_hashes:
            return False, ["DUP_SUCCESS_HASH"], diag

        self.seen_success_hashes.add(succ_hash)
        diag["success_hash"] = succ_hash

        # Commit subsequence rarity archive ONLY on confirmed success
        if "subseq" in diag:
            key = tuple(diag["subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1
        elif "_candidate_subseq" in diag:
            key = tuple(diag["_candidate_subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1

        # store cfg for parent tracking
        self.parent_cfgs[genome.gid] = cfg
        return True, reasons, diag
