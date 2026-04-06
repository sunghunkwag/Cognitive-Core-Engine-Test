"""
OmegaForgeV13 main engine class.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    CONTROL_OPS,
    Instruction,
    ProgramGenome,
)
from cognitive_core_engine.omega_forge.cfg import ControlFlowGraph
from cognitive_core_engine.omega_forge.vm import MacroLibrary, VirtualMachine
from cognitive_core_engine.omega_forge.concepts import rand_inst
from cognitive_core_engine.omega_forge.benchmark import (
    StrictStructuralDetector,
    TaskBenchmark,
)
from cognitive_core_engine.omega_forge.evidence import EngineConfig, EvidenceWriter


class OmegaForgeV13:
    def __init__(
        self,
        seed: int = 42,
        detector: Optional[StrictStructuralDetector] = None,
        vm: Optional[VirtualMachine] = None,
        config: Optional[EngineConfig] = None,
    ) -> None:
        random.seed(seed)
        self.seed = seed
        self.vm = vm or VirtualMachine()
        self.detector = detector or StrictStructuralDetector()
        self.cfg = config or EngineConfig()

        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        self.parents_index: Dict[str, ProgramGenome] = {}

    def init_population(self) -> None:
        self.population = []
        for i in range(self.cfg.pop_size):
            L = random.randint(self.cfg.init_len_min, self.cfg.init_len_max)
            insts = [rand_inst() for _ in range(L)]
            g = ProgramGenome(gid=f"init_{i}", instructions=insts, parents=[], generation=0)
            self.population.append(g)
        self._reindex()

    def _reindex(self) -> None:
        self.parents_index = {g.gid: g for g in self.population}

    def _get_parent_obj(self, g: ProgramGenome) -> Optional[ProgramGenome]:
        if not g.parents:
            return None
        pid = g.parents[0]
        return self.parents_index.get(pid)

    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"g{self.generation}_{random.randint(0, 999999)}"

        # structural mutation mixture
        # 1) splice macro sometimes
        roll = random.random()
        if roll < 0.20 and len(child.instructions) + 5 < self.cfg.max_code_len:
            macro = MacroLibrary.loop_skeleton() if random.random() < 0.7 else MacroLibrary.call_skeleton()
            pos = random.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [m.clone() for m in macro]
        elif roll < 0.45 and child.instructions:
            # replace a random instruction with a control op to encourage CFG change
            pos = random.randint(0, len(child.instructions) - 1)
            op = random.choice(list(CONTROL_OPS))
            child.instructions[pos] = Instruction(op, random.randint(-8, 8), random.randint(0, 7), random.randint(0, 7))
        elif roll < 0.75 and len(child.instructions) < self.cfg.max_code_len:
            # insert random instruction
            pos = random.randint(0, len(child.instructions))
            child.instructions.insert(pos, rand_inst())
        else:
            # delete
            if len(child.instructions) > 6:
                pos = random.randint(0, len(child.instructions) - 1)
                child.instructions.pop(pos)

        return child

    def step(self, writer: Optional[EvidenceWriter] = None) -> Tuple[int, int]:
        self.generation += 1
        successes_this_gen = 0

        # Evaluate all genomes
        for g in self.population:
            parent = self._get_parent_obj(g)
            st = self.vm.execute(g, [1.0] * 8)
            passed, reasons, diag = self.detector.evaluate(g, parent, st, self.vm, self.generation)
            if passed:
                successes_this_gen += 1
                if writer is not None:
                    ev = {
                        "type": "evidence",
                        "gen": self.generation,
                        "gid": g.gid,
                        "parent": parent.gid if parent else None,
                        "code_hash": g.code_hash(),
                        "reasons": reasons,
                        "diag": diag,
                        "metrics": {
                            "steps": st.steps,
                            "coverage": st.coverage(len(g.instructions)),
                            "loops": st.loops_count,
                            "branches": st.conditional_branches,
                            "scc_n": diag.get("scc_n", 0),
                        },
                    }
                    writer.write(ev)
                    writer.flush_fsync()


        # Reproduce: score-based elite selection + CFG-diversity (prevents random drift / stagnation)
        # Score rewards "structural potential" even when not yet passing strict detector gates.
        for g in self.population:
            parent = self._get_parent_obj(g)
            # we already executed VM above in this step's loop, but we don't store st; re-execute cheaply on fixed input
            st2 = self.vm.execute(g, [1.0] * 8)
            cfg2 = ControlFlowGraph.from_trace(st2.trace, len(g.instructions))
            cov = st2.coverage(len(g.instructions))
            scc_n = len(cfg2.sccs())

            # STRUCTURAL score (original): coverage + loops/branches/calls + SCC
            struct_score = cov + 0.02 * min(st2.loops_count, 50) + 0.01 * min(st2.conditional_branches, 50) + 0.03 * min(st2.max_call_depth, 10) + 0.08 * min(scc_n, 6)
            if st2.error or (not st2.halted_cleanly):
                struct_score -= 0.5

            # TASK-AWARE score (NEW): practical problem-solving ability
            task_score = TaskBenchmark.evaluate(g, self.vm)

            # Combined score: 50% structure + 50% task performance
            score = 0.5 * struct_score + 0.5 * task_score * 2.0  # task_score scaled to ~1.0 max

            g.last_score = float(score)
            g.last_cfg_hash = cfg2.canonical_hash()

        # Sort by score
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)

        # Diversity filter: prefer unique CFG hashes among the top band
        elites: List[ProgramGenome] = []
        seen_cfg: Set[str] = set()
        band = ranked[: max(self.cfg.elite_keep * 3, self.cfg.elite_keep)]
        for g in band:
            if len(elites) >= self.cfg.elite_keep:
                break
            if g.last_cfg_hash not in seen_cfg:
                elites.append(g)
                seen_cfg.add(g.last_cfg_hash)

        # If not enough elites due to diversity constraint, fill from ranked
        if len(elites) < self.cfg.elite_keep:
            for g in ranked:
                if len(elites) >= self.cfg.elite_keep:
                    break
                elites.append(g)

        next_pop: List[ProgramGenome] = []
        for e in elites:
            kept = e.clone()
            next_pop.append(kept)
            for _ in range(self.cfg.children_per_elite):
                next_pop.append(self.mutate(e))

        # trim to pop size
        self.population = next_pop[: self.cfg.pop_size]
        self._reindex()

        return successes_this_gen, len(getattr(self.detector, "seen_success_hashes", set()))
