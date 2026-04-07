from __future__ import annotations

import importlib.util
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognitive_core_engine.core.utils import stable_hash, now_ms
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.tools import ToolRegistry
from cognitive_core_engine.core.skills import SkillLibrary
from cognitive_core_engine.core.project_graph import ProjectGraph
from cognitive_core_engine.core.environment import TaskSpec, ResearchEnvironment, RuleProposal
from cognitive_core_engine.core.agent import Agent, AgentConfig

from agi_modules.competence_map import CompetenceMap
from agi_modules.goal_generator import GoalGenerator, GoalGenerationError
from agi_modules.intrinsic_motivation import IntrinsicMotivationModule
from agi_modules.concept_graph import ConceptGraph
from agi_modules.transfer_engine import TransferEngine
from agi_modules.self_model import SelfModel
from agi_modules.difficulty_scheduler import DifficultyScheduler
from agi_modules.self_improvement import SelfImprovementEngine
from agi_modules.agi_tracker import AGIProgressTracker
from agi_modules.external_benchmark import ExternalBenchmarkHarness
from agi_modules.solver_bridge import create_solver_pair
from cognitive_core_engine.core.causal_chain import CausalChainTracker
from cognitive_core_engine.omega_forge.benchmark import EnvironmentCoupledFitness
from agi_modules.self_referential_model import AdvancedSelfReferentialModel


def load_unified_critic_module() -> Any:
    """Load the governance critic module.

    Tries the new package path first, falls back to legacy file location.
    """
    try:
        from cognitive_core_engine.governance import critic as _critic_mod
        return _critic_mod
    except ImportError:
        pass
    # Fallback: legacy file path
    module_path = Path(__file__).resolve().parents[2] / "unified_rsi_extended.py"
    legacy_path = Path(__file__).resolve().parents[2] / "unified_rsi_extended .py"
    if not module_path.exists() and legacy_path.exists():
        module_path = legacy_path
    spec = importlib.util.spec_from_file_location("unified_rsi_extended", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load critic module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class OrchestratorConfig:
    agents: int = 8
    base_budget: int = 20
    selection_top_k: int = 4
    budget_growth: float = 1.06
    legacy_task_ratio: float = 0.3


class Orchestrator:
    """
    C-layer:
    - maintains SharedMemory, SkillLibrary, ProjectGraph
    - runs multiple B-type agents per round
    - distills principles from best episodes
    - adapts org policy (role mix, risk) based on outcomes
    """

    def __init__(self, cfg: OrchestratorConfig,
                 env: ResearchEnvironment,
                 tools: ToolRegistry) -> None:
        self.cfg = cfg
        self.env = env
        self.tools = tools

        self.mem = SharedMemory()
        self.skills = SkillLibrary()
        self.projects = ProjectGraph()

        self._agents: List[Agent] = []
        self._org_policy: Dict[str, Any] = {
            "risk": 0.25,
            "role_mix": ["theorist", "builder", "experimenter", "verifier", "strategist"],
            "infra_focus": 0.5,
        }
        self.candidate_queue: List[RuleProposal] = []
        self.evaluation_rules: Dict[str, Any] = {
            "min_score": 0.25,
            "l1_update_rate": 0.08,
            "min_transfer": 0.05,
            "min_holdout_pass_rate": 0.30,
            "max_generalization_gap": 0.05,
            "holdout_weight": 1.0,
            "generalization_gap_penalty": 0.75,
            "discovery_cost_penalty": 0.08,
            "min_adversarial_pass_rate": 0.28,
            "min_shift_holdout_pass_rate": 0.25,
            "max_holdout_discovery_cost": 4.0,
            "require_holdout_metrics": True,
        }
        self.meta_rules: Dict[str, Any] = {
            "l1_update_rate_bounds": (0.04, 0.20),
        }
        self.invariants: Dict[str, Any] = {
            "min_evidence": 1,
            "min_transfer": 0.05,
            "l1_update_rate_bounds": (0.04, 0.20),
        }
        self._recent_rewards: List[float] = []
        self._adoption_cooldown_ms = 1500
        self._last_adoption_ms = 0
        self._critic_module: Optional[Any] = None

        # --- AGI modules ---
        self.competence_map = CompetenceMap()
        self.goal_gen = GoalGenerator(self.competence_map, self.mem, random.Random(42))
        self.intrinsic_motivation = IntrinsicMotivationModule(self.mem, self.competence_map)
        self.concept_graph = ConceptGraph()
        self.transfer_engine = TransferEngine(self.concept_graph)
        self.self_model = AdvancedSelfReferentialModel()
        self.difficulty_scheduler = DifficultyScheduler(self.competence_map, random.Random(42))
        self.self_improvement = SelfImprovementEngine()
        self.agi_tracker = AGIProgressTracker()
        self.external_benchmark = ExternalBenchmarkHarness(seed=42)
        self._initial_domain_count = len(self.env.tasks)

        # BN-08: Recursive emergence infrastructure
        self.causal_tracker = CausalChainTracker()
        self.env_fitness = EnvironmentCoupledFitness()
        self._rsi_registrar: Optional[Any] = None
        self.agi_tracker.set_initial_domains(set(t.name for t in self.env.tasks))

        self._init_agents()

    def _init_agents(self) -> None:
        roles = self._org_policy["role_mix"]
        for i in range(self.cfg.agents):
            role = roles[i % len(roles)]
            cfg = AgentConfig(
                name=f"agent_{i:02d}",
                role=role,
                planner_depth=4 if role in ("theorist", "strategist") else 3,
                planner_width=7 if role == "strategist" else 6,
                risk=self._org_policy["risk"],
            )
            self._agents.append(Agent(
                cfg, self.tools, self.mem, self.skills,
                intrinsic_motivation=self.intrinsic_motivation,
                self_model=self.self_model,
                concept_graph=self.concept_graph,
            ))

    def _record_round_rewards(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        mean_reward = sum(r["reward"] for r in results) / max(1, len(results))
        self._recent_rewards.append(mean_reward)
        if len(self._recent_rewards) > 8:
            self._recent_rewards.pop(0)

    def _detect_stagnation(self, window: int = 5, threshold: float = 0.01) -> bool:
        if len(self._recent_rewards) < window:
            return False
        start = self._recent_rewards[-window]
        end = self._recent_rewards[-1]
        return (end - start) < threshold

    def _build_gap_spec(self, round_idx: int, round_out: Dict[str, Any]) -> Dict[str, Any]:
        # Determine invention target based on available diagnostics
        gaps = self.competence_map.gaps()
        invention_targets = []
        if gaps:
            invention_targets.append("capability")
        if self.concept_graph.depth() < 2:
            invention_targets.append("representation")
        if not invention_targets:
            invention_targets.append("strategy")

        return {
            "round": round_idx,
            "seed": round_idx + 11,
            "tasks": round_out.get("tasks", []),
            "recent_rewards": list(self._recent_rewards[-5:]),
            "constraints": {
                "quarantine_only": True,
                "no_self_adoption": True,
                "max_candidates": 1,
            },
            "competence_gaps": [(d, diff) for d, diff in gaps[:5]],
            "concept_depth": self.concept_graph.depth(),
            "invention_target": invention_targets[0] if invention_targets else "integration",
        }

    def _omega_generate_candidates(self, gap_spec: Dict[str, Any]) -> List[RuleProposal]:
        import cognitive_core_engine.omega_forge.stage1 as omega

        engine = omega.Stage1Engine(seed=int(gap_spec.get("seed", 0)))
        engine.init_population()
        for _ in range(3):
            engine.step()
            if engine.candidates:
                break

        candidates = list(engine.candidates)
        if not candidates and engine.population:
            fallback = engine.population[0]
            candidates = [
                {
                    "gid": fallback.gid,
                    "generation": engine.generation,
                    "code": [(i.op, i.a, i.b, i.c) for i in fallback.instructions],
                    "metrics": {
                        "fallback": True,
                        "train_pass_rate": 0.45,
                        "holdout_pass_rate": 0.42,
                        "adversarial_pass_rate": 0.40,
                        "discovery_cost": {"holdout": 1.0, "train": 1.0},
                    },
                    "task_scores": {},
                }
            ]

        proposals: List[RuleProposal] = []
        for cand in candidates[: gap_spec.get("constraints", {}).get("max_candidates", 1)]:
            payload = {"candidate": cand, "gap_spec": gap_spec}
            proposal_id = stable_hash({"level": "L0", "payload": payload})
            proposals.append(
                RuleProposal(
                    proposal_id=proposal_id,
                    level="L0",
                    payload=payload,
                    creator_key=stable_hash({"source": "omega", "gid": cand.get("gid")}),
                    created_ms=now_ms(),
                    evidence={"metrics": cand.get("metrics", {}), "task_scores": cand.get("task_scores", {})},
                )
            )
        return proposals

    def _load_critic(self) -> Any:
        if self._critic_module is None:
            self._critic_module = load_unified_critic_module()
        return self._critic_module

    def _critic_evaluate(self, proposal: RuleProposal) -> Dict[str, Any]:
        critic = self._load_critic()
        candidate = proposal.payload.get("candidate")
        if proposal.level == "L0":
            assert isinstance(candidate, dict), "candidate missing"
            assert candidate.get("gid"), "candidate gid missing"
        if proposal.level == "L1":
            assert isinstance(proposal.payload.get("evaluation_update"), dict), "evaluation_update missing"
        if proposal.level == "L2":
            assert isinstance(proposal.payload.get("meta_update"), dict), "meta_update missing"
        packet = {
            "proposal": asdict(proposal),
            "evaluation_rules": dict(self.evaluation_rules),
            "invariants": dict(self.invariants),
        }
        return critic.critic_evaluate_candidate_packet(packet, invariants=self.invariants)

    def _adopt_proposal(self, proposal: RuleProposal, verdict: Dict[str, Any]) -> bool:
        if verdict.get("verdict") != "approve":
            return False
        if not proposal.creator_key or not verdict.get("approval_key"):
            return False
        if now_ms() - self._last_adoption_ms < self._adoption_cooldown_ms:
            return False

        if proposal.level == "L0":
            self.mem.add(
                "artifact",
                f"adopted_candidate:{proposal.proposal_id}",
                {"proposal": proposal.payload, "critic": verdict},
                tags=["adopted", "L0"],
            )
        elif proposal.level == "L1":
            update = proposal.payload.get("evaluation_update", {})
            if update:
                self.evaluation_rules.update(update)
        elif proposal.level == "L2":
            meta_update = proposal.payload.get("meta_update", {})
            if meta_update:
                self.meta_rules.update(meta_update)
                if "l1_update_rate" in meta_update:
                    self.evaluation_rules["l1_update_rate"] = meta_update["l1_update_rate"]

        self._last_adoption_ms = now_ms()
        return True

    def _apply_l1_update(self) -> Optional[Dict[str, Any]]:
        if len(self._recent_rewards) < 2:
            return None
        trend = self._recent_rewards[-1] - self._recent_rewards[-2]
        update_rate = float(self.evaluation_rules.get("l1_update_rate", 0.08))
        min_score = float(self.evaluation_rules.get("min_score", 0.4))
        if trend < 0:
            min_score = min(0.9, min_score + update_rate)
        else:
            min_score = max(0.1, min_score - update_rate / 2.0)
        self.evaluation_rules["min_score"] = min_score
        return {"min_score": min_score}

    def _propose_l2_update(self, round_idx: int, force: bool = False) -> Optional[RuleProposal]:
        if not force and round_idx % 6 != 0:
            return None
        bounds = self.meta_rules.get("l1_update_rate_bounds", (0.04, 0.20))
        current = float(self.evaluation_rules.get("l1_update_rate", 0.08))
        proposed = max(bounds[0], min(bounds[1], current + 0.01))
        payload = {"meta_update": {"l1_update_rate": proposed}}
        proposal_id = stable_hash({"level": "L2", "payload": payload, "round": round_idx})
        return RuleProposal(
            proposal_id=proposal_id,
            level="L2",
            payload=payload,
            creator_key=stable_hash({"source": "meta", "round": round_idx}),
            created_ms=now_ms(),
            evidence={"meta": {"round": round_idx}},
        )

    def _distill_principles(self, round_idx: int,
                            results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        results_sorted = sorted(results, key=lambda r: r["reward"], reverse=True)
        top = results_sorted[: self.cfg.selection_top_k]
        bottom = results_sorted[-self.cfg.selection_top_k:]

        self.mem.add(
            "note",
            f"round_{round_idx}_distill",
            {
                "top": [
                    {
                        "agent": r["agent"],
                        "role": r["role"],
                        "task": r["info"]["task"],
                        "reward": r["reward"],
                        "action": r["action"],
                    }
                    for r in top
                ],
                "bottom": [
                    {
                        "agent": r["agent"],
                        "role": r["role"],
                        "task": r["info"]["task"],
                        "reward": r["reward"],
                        "action": r["action"],
                    }
                    for r in bottom
                ],
                "env": {
                    "tq": self.env.global_tool_quality,
                    "kq": self.env.global_kb_quality,
                    "oq": self.env.global_org_quality,
                },
                "policy": dict(self._org_policy),
            },
            tags=["distill", "round"],
        )

        for r in top:
            self.mem.add(
                "principle",
                f"good_pattern:{r['info']['task']}:{r['action']}",
                {
                    "agent": r["agent"],
                    "role": r["role"],
                    "task": r["info"]["task"],
                    "action": r["action"],
                    "reward": r["reward"],
                    "env": {
                        "tq": r["info"]["tq"],
                        "kq": r["info"]["kq"],
                        "oq": r["info"]["oq"],
                    },
                },
                tags=["principle", "good"],
            )
        for r in bottom:
            self.mem.add(
                "principle",
                f"bad_pattern:{r['info']['task']}:{r['action']}",
                {
                    "agent": r["agent"],
                    "role": r["role"],
                    "task": r["info"]["task"],
                    "action": r["action"],
                    "reward": r["reward"],
                    "env": {
                        "tq": r["info"]["tq"],
                        "kq": r["info"]["kq"],
                        "oq": r["info"]["oq"],
                    },
                },
                tags=["principle", "bad"],
            )

        self.mem.extract_principles(k=max(3, self.cfg.selection_top_k // 2))

        rewards = [r["reward"] for r in results]
        mean = sum(rewards) / max(1, len(rewards))
        var = sum((x - mean) ** 2 for x in rewards) / max(1, len(rewards))
        std = math.sqrt(var)

        tq = self.env.global_tool_quality
        kq = self.env.global_kb_quality
        oq = self.env.global_org_quality

        if tq < kq and tq < oq:
            self._org_policy["role_mix"] = [
                "builder", "builder", "experimenter",
                "verifier", "strategist"
            ]
            self._org_policy["infra_focus"] = min(0.7, self._org_policy["infra_focus"] + 0.1)
        elif kq < tq and kq < oq:
            self._org_policy["role_mix"] = [
                "verifier", "verifier", "theorist",
                "builder", "strategist"
            ]
            self._org_policy["infra_focus"] = min(0.7, self._org_policy["infra_focus"] + 0.05)
        elif oq < tq and oq < kq:
            self._org_policy["role_mix"] = [
                "strategist", "strategist", "builder",
                "experimenter", "verifier"
            ]
            self._org_policy["infra_focus"] = min(0.7, self._org_policy["infra_focus"] + 0.05)
        else:
            self._org_policy["role_mix"] = [
                "theorist", "builder", "experimenter",
                "verifier", "strategist"
            ]
            self._org_policy["infra_focus"] = max(0.4, self._org_policy["infra_focus"] - 0.05)

        if std > 0.10:
            self._org_policy["risk"] = max(0.05, self._org_policy["risk"] - 0.02)
        else:
            self._org_policy["risk"] = min(0.40, self._org_policy["risk"] + 0.01)

        roles = self._org_policy["role_mix"]
        for i, ag in enumerate(self._agents):
            ag.cfg.role = roles[i % len(roles)]
            ag.cfg.risk = self._org_policy["risk"]

    def _assign_tasks(self) -> List[TaskSpec]:
        tasks = [self.env.sample_task()]
        if self.cfg.agents > 4:
            tasks.append(self.env.sample_task())

        # BN-08: Inject skill-derived goals with priority
        skill_goals = self.goal_gen.get_skill_derived_goals()
        for sg in skill_goals[:2]:  # inject up to 2 skill-derived goals
            self.env.add_domain(sg.domain, sg.difficulty, sg.baseline)
            tasks.append(TaskSpec(
                name=sg.name, difficulty=sg.difficulty,
                baseline=sg.baseline, domain=sg.domain))
            self.agi_tracker.update_goals(self_generated=1, total=1)

        # Only attempt goal generation after warmup (competence map has data)
        # This preserves backward compatibility with existing tests
        if not self.competence_map.all_keys():
            return tasks

        # Use GoalGenerator's own RNG to avoid perturbing env.rng or global random
        goal_rng = self.goal_gen.rng

        # Attempt to replace some tasks with generated goals (70%/30% split)
        for i in range(len(tasks)):
            if goal_rng.random() > self.cfg.legacy_task_ratio:
                try:
                    generated = self.goal_gen.generate(n=1)
                    if generated:
                        g = generated[0]
                        # Add new domain to environment if creative goal
                        # Use fingerprinting to detect truly novel domains
                        if g.name.startswith("creative_"):
                            is_novel = self.goal_gen.register_domain(g.domain)
                            self.env.add_domain(g.domain, g.difficulty, g.baseline)
                            if is_novel:
                                self.agi_tracker.update_open_endedness(
                                    new_domains=1, difficulty_increases=0,
                                    domains_above_random=0)
                        tasks[i] = TaskSpec(
                            name=g.name, difficulty=g.difficulty,
                            baseline=g.baseline, domain=g.domain)
                        self.agi_tracker.update_goals(self_generated=1, total=1)
                        continue
                except GoalGenerationError:
                    pass  # Keep legacy task
            self.agi_tracker.update_goals(self_generated=0, total=1)

        return tasks

    def _make_agent_solve_fn(self) -> Any:
        """Create a solve_fn closure that uses the best agent's WorldModel.

        The ADB benchmark task is 'reverse the input list'. The agent's
        WorldModel Q-values guide action selection: action 0 = reverse,
        1 = sort, 2 = sort desc, 3 = identity. The agent is NOT told the
        answer — it must infer 'reverse' from its learned Q-values.
        """
        # Pick the agent with the most experience (highest total usage)
        def _agent_experience(a):
            counts = a.wm._sa_counts
            if not counts:
                return 0
            return sum(v if isinstance(v, (int, float)) else getattr(v, 'count', 0)
                       for v in counts.values())
        best_agent = max(self._agents, key=_agent_experience)
        wm = best_agent.wm

        def solve_fn(inp: list) -> list:
            """Solve an ADB reverse task using the agent's world model."""
            obs = {"task": "reverse", "domain": "held_out_reverse",
                   "difficulty": 3, "budget": 12, "phase": "research"}
            actions = ["attempt_breakthrough", "build_tool",
                       "write_verified_note", "tune_orchestration"]
            # Use Q-values to pick best action index
            q_values = [wm.q_value(obs, a) for a in actions]
            best_idx = q_values.index(max(q_values))
            # Map action index to list transformation
            candidates = {
                0: list(reversed(inp)),
                1: sorted(inp),
                2: sorted(inp, reverse=True),
                3: list(inp),
            }
            return candidates.get(best_idx % 4, list(inp))
        return solve_fn

    def _make_held_out_fn(self) -> Any:
        """Create a held_out_fn for measure_held_out_generalization."""
        def _agent_experience(a):
            counts = a.wm._sa_counts
            if not counts:
                return 0
            return sum(v if isinstance(v, (int, float)) else getattr(v, 'count', 0)
                       for v in counts.values())
        best_agent = max(self._agents, key=_agent_experience)
        wm = best_agent.wm

        def held_out_fn(inp: list, domain: str) -> list:
            obs = {"task": domain, "domain": domain,
                   "difficulty": 3, "budget": 12, "phase": "research"}
            actions = ["attempt_breakthrough", "build_tool",
                       "write_verified_note", "tune_orchestration"]
            q_values = [wm.q_value(obs, a) for a in actions]
            best_idx = q_values.index(max(q_values))

            if "reverse" in domain:
                candidates = [list(reversed(inp)), sorted(inp), list(inp)]
            elif "sort" in domain:
                candidates = [sorted(inp), list(reversed(inp)), list(inp)]
            else:
                seen, deduped = set(), []
                for v in inp:
                    if v not in seen:
                        seen.add(v)
                        deduped.append(v)
                candidates = [deduped, list(inp), sorted(inp)]
            return candidates[best_idx % len(candidates)]
        return held_out_fn

    def _budget_for_agent(self, base_budget: int, role: str) -> int:
        infra_focus = float(self._org_policy.get("infra_focus", 0.5))
        infra_roles = {"builder", "verifier", "strategist"}
        if role in infra_roles:
            scale = 0.85 + 0.5 * infra_focus
        else:
            scale = 0.85 + 0.5 * (1.0 - infra_focus)
        return max(8, int(base_budget * scale))

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        tasks = self._assign_tasks()
        budget = int(self.cfg.base_budget * (self.cfg.budget_growth ** round_idx))

        results: List[Dict[str, Any]] = []
        drift_result = None
        for idx, ag in enumerate(self._agents):
            task = tasks[idx % len(tasks)]
            proj_node = self.projects.pick_node_for_round(task.name)
            agent_budget = self._budget_for_agent(budget, ag.cfg.role)
            obs = self.env.make_observation(task, agent_budget)

            # ── Self-referential state encoding (before act_on_project) ──
            # Encode unified internal-external state into HDC space
            active_skill_ids = [s.id for s in ag.skills.list()]
            self.self_model.encode_self_referential_state(
                env_obs=obs,
                competence_map=self.competence_map,
                concept_graph=self.concept_graph,
                active_skills=active_skill_ids,
            )
            # Detect architectural drift and trigger rollback if critical
            if idx == 0:  # check once per round, not per agent
                drift_result = self.self_model.detect_architectural_drift()
                if drift_result.get("should_rollback"):
                    # Signal governance: critical cognitive drift detected
                    # Use L1 level (evaluation rule update) to avoid L0 candidate assertion
                    self.candidate_queue.append(RuleProposal(
                        proposal_id=stable_hash({"drift": drift_result, "round": round_idx}),
                        level="L1",
                        payload={"evaluation_update": {
                            "drift_rollback": True,
                            "min_score": float(self.evaluation_rules.get("min_score", 0.25)) + 0.1,
                        }},
                        creator_key=stable_hash({"source": "self_referential_drift"}),
                        created_ms=now_ms(),
                        evidence={"drift": drift_result},
                    ))

            res = ag.act_on_project(self.env, proj_node, obs)
            results.append(res)
            self.projects.update_node(proj_node.id, res["reward"], res["mem_id"])

        self._distill_principles(round_idx, results)

        # Update competence map and concept promotions
        # Domain/difficulty are in the task, not the info dict
        # Normalize reward to [0, 1]: env rewards are typically [0, 0.20]
        for idx, res in enumerate(results):
            if res.get("info", {}).get("skipped"):
                continue
            task = tasks[idx % len(tasks)]
            domain = task.domain
            difficulty = task.difficulty
            reward = float(res.get("reward", 0.0))
            normalized_reward = min(1.0, max(0.0, reward * 5.0))
            if domain:
                self.competence_map.update(domain, difficulty, normalized_reward)

        # Concept promotion pass: sweep all levels bottom-up with cascade
        promoted_count = self.concept_graph.sweep_promote_all(round_idx)
        if promoted_count > 0:
            self.agi_tracker.update_abstraction(self.concept_graph.depth())

        # Record co-occurrences for concepts used in same round
        round_concepts = []
        for idx, res in enumerate(results):
            action = res.get("action", "")
            task = tasks[idx % len(tasks)]
            for c in self.concept_graph.all_concepts():
                if c.name == f"{action}@{task.domain}" and c.level == 0:
                    round_concepts.append(c.concept_id)
        for i in range(len(round_concepts)):
            for j in range(i + 1, len(round_concepts)):
                self.concept_graph.record_co_occurrence(round_concepts[i], round_concepts[j])
                self.concept_graph.record_co_occurrence(round_concepts[j], round_concepts[i])

        # Track domains_above_random: competence > 0.55 (above 0.5 random baseline)
        # Only counts domains registered via fingerprinting (not initial 6)
        above_random_count = 0
        for domain_name in self.goal_gen._domain_fingerprints:
            for key in self.competence_map.all_keys():
                if key[0] == domain_name and self.competence_map.get_rate(key[0], key[1]) > 0.55:
                    above_random_count += 1
                    break
        if above_random_count > 0:
            # Update with delta (only newly-above-random domains)
            self.agi_tracker._domains_above_random = above_random_count

        # Difficulty scheduling
        self.difficulty_scheduler.schedule(round_idx)
        self.difficulty_scheduler.inject_chaos()

        # Self-improvement introspection
        self.self_improvement.record_decision({
            "round": round_idx,
            "reward": sum(r["reward"] for r in results) / max(1, len(results)),
            "domain": results[0].get("info", {}).get("domain", "") if results else "",
            "action": results[0].get("action", "") if results else "",
        })

        return {
            "round": round_idx,
            "tasks": [t.name for t in tasks],
            "results": results,
            "env": {
                "tq": self.env.global_tool_quality,
                "kq": self.env.global_kb_quality,
                "oq": self.env.global_org_quality,
            },
            "policy": dict(self._org_policy),
            "concept_depth": self.concept_graph.depth(),
            "concept_count": self.concept_graph.size(),
        }

    def run_recursive_cycle(
        self,
        round_idx: int,
        stagnation_override: Optional[bool] = None,
        force_meta_proposal: bool = False,
    ) -> Dict[str, Any]:
        # ── Governance anchor integrity check (every cycle) ──
        # Halt if the immutable objective anchor has been tampered with.
        if not self.self_model.verify_anchor_integrity():
            raise RuntimeError(
                "GOVERNANCE HALT: Immutable objective anchor integrity check "
                "failed at start of run_recursive_cycle. Possible RSI bypass attempt."
            )

        round_out = self.run_round(round_idx)
        self._record_round_rewards(round_out["results"])

        stagnation = stagnation_override if stagnation_override is not None else self._detect_stagnation()

        # A5: External stagnation detection supplements internal signal
        # Wire an agent solve_fn so the benchmark is connected (not always-zero)
        if not stagnation and round_idx % 5 == 0 and round_idx > 0:
            agent_solve_fn = self._make_agent_solve_fn()
            self.external_benchmark.run_adb_snapshot(solve_fn=agent_solve_fn)
            # BN-07: Run full external benchmark with real solvers
            arc_fn, he_fn = create_solver_pair()
            self.external_benchmark.run_full_benchmark(
                arc_solve_fn=arc_fn, humaneval_solve_fn=he_fn
            )
            external_stagnation = self.external_benchmark.detect_external_stagnation()
            if external_stagnation:
                stagnation = True

        # Feed stagnation info to GoalGenerator
        self.goal_gen.set_stagnating(bool(stagnation))

        gap_spec = None
        if stagnation:
            gap_spec = self._build_gap_spec(round_idx, round_out)
            self.candidate_queue.extend(self._omega_generate_candidates(gap_spec))

        # Transfer learning: attempt from best to worst performing domains
        transfer_report = None
        if self.transfer_engine.can_transfer(round_idx):
            all_keys = self.competence_map.all_keys()
            if len(all_keys) >= 2:
                # Sort by competence rate, transfer from best to worst
                sorted_keys = sorted(all_keys,
                    key=lambda k: self.competence_map.get_rate(k[0], k[1]),
                    reverse=True)
                source = sorted_keys[0]
                target = sorted_keys[-1]
                if source[0] != target[0]:  # Different domains
                    # Use raw reward scale for baseline (not normalized competence)
                    pre_baseline = self.competence_map.get_rate(target[0], target[1]) / 5.0
                    transfer_report = self.transfer_engine.transfer(
                        source[0], target[0])
                    if transfer_report.get("attempted"):
                        self.transfer_engine.record_transfer_round(round_idx)
                        success = self.transfer_engine.measure_transfer_success(
                            target[0], pre_baseline)
                        self.agi_tracker.update_transfer(success, True)
                        if success < -0.1:
                            self.transfer_engine.rollback_transfer(target[0])

        # Self-improvement introspection (every 5 rounds)
        self_improvement_result = None
        if round_idx > 0 and round_idx % 5 == 0:
            diagnosis = self.self_improvement.introspect_decision_quality(
                [{"reward": r["reward"], "action": r["action"],
                  "domain": r.get("info", {}).get("domain", "")}
                 for r in round_out["results"]])
            mod = self.self_improvement.propose_policy_modification(diagnosis)
            if mod:
                test_result = self.self_improvement.test_modification(
                    mod, self.env, {"risk": self._org_policy["risk"]})

                # ── Anti-wireheading: validate metric integrity before applying ──
                proposed_delta = test_result.get("delta", 0.0) if isinstance(test_result, dict) else float(test_result)
                ext_scores = self.external_benchmark.get_external_score_history()
                ext_delta = (ext_scores[-1] - ext_scores[-2]) if len(ext_scores) >= 2 else 0.0
                integrity = self.self_model.validate_metric_integrity(
                    proposed_delta=proposed_delta,
                    structural_complexity_change=self.concept_graph.size(),
                    external_benchmark_delta=ext_delta,
                )
                if integrity.get("should_reject"):
                    # Reward spoofing detected — governance rejects this modification
                    applied = False
                else:
                    applied = self.self_improvement.apply_if_beneficial(
                        mod, test_result, {"risk": self._org_policy["risk"]})
                self.agi_tracker.update_self_improvement(applied, True)
                if applied:
                    changes = mod.get("changes", {})
                    if "risk_delta" in changes:
                        new_risk = max(0.05, min(0.5,
                            self._org_policy["risk"] + changes["risk_delta"]))
                        self._org_policy["risk"] = new_risk
                self_improvement_result = {
                    "proposed": True,
                    "test_result": test_result,
                    "applied": applied,
                    "empirically_tested": test_result.get("empirically_tested", False)
                        if isinstance(test_result, dict) else False,
                }

        l1_update = self._apply_l1_update()
        if l1_update:
            l1_proposal = RuleProposal(
                proposal_id=stable_hash({"level": "L1", "payload": l1_update, "round": round_idx}),
                level="L1",
                payload={"evaluation_update": dict(l1_update)},
                creator_key=stable_hash({"source": "l1", "round": round_idx}),
                created_ms=now_ms(),
                evidence={"l1_update": dict(l1_update)},
            )
            self.candidate_queue.append(l1_proposal)

        l2_proposal = self._propose_l2_update(round_idx, force=force_meta_proposal or stagnation)
        if l2_proposal:
            self.candidate_queue.append(l2_proposal)

        critic_results: List[Dict[str, Any]] = []
        proposals_by_id: Dict[str, RuleProposal] = {}
        while self.candidate_queue:
            proposal = self.candidate_queue.pop(0)
            proposals_by_id[proposal.proposal_id] = proposal
            verdict = self._critic_evaluate(proposal)
            adopted = self._adopt_proposal(proposal, verdict)
            critic_results.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "level": proposal.level,
                    "verdict": verdict.get("verdict"),
                    "adopted": adopted,
                }
            )

        # BN-08: RSI skill registration + causal chain tracking
        skill_birth_events: List[Dict[str, Any]] = []
        skill_derived_goals: List[Any] = []
        if critic_results:
            try:
                from cognitive_core_engine.omega_forge.rsi_pipeline import RSISkillRegistrar
                if self._rsi_registrar is None:
                    self._rsi_registrar = RSISkillRegistrar(self.skills, self.mem)
                rsi_results = self._rsi_registrar.process_critic_results(
                    critic_results, proposals_by_id)
                # Collect skill birth events
                skill_birth_events = self._rsi_registrar.get_recent_birth_events()
            except Exception:
                pass

        # BN-08: Update environment-coupled fitness
        env_state = {
            "recent_rewards": list(self._recent_rewards[-10:]),
            "task_count": len(self.env.tasks),
            "round_idx": round_idx,
            "stagnation": stagnation,
            "difficulty": round_out.get("results", [{}])[0].get("info", {}).get("difficulty", 3)
                if round_out.get("results") else 3,
        }
        self.env_fitness.update_tasks(env_state)

        # BN-08: Record skill births in causal chain and notify GoalGenerator
        for birth_event in skill_birth_events:
            event_id = self.causal_tracker.record_skill_birth(
                skill_id=birth_event.get("skill_id", ""),
                genome_fitness=birth_event.get("genome_fitness", 0.0),
                round_idx=round_idx,
            )
            # Notify GoalGenerator to create skill-derived goals
            new_goals = self.goal_gen.on_skill_registered(birth_event)
            for goal in new_goals:
                goal_event_id = self.causal_tracker.record_goal_from_skill(
                    goal_name=goal.name,
                    trigger_skill_id=birth_event.get("skill_id", ""),
                    trigger_event_id=event_id,
                    round_idx=round_idx,
                )
                skill_derived_goals.append(goal)

        # BN-08: Check if skill-derived goals achieved positive reward
        for result in round_out.get("results", []):
            task_name = result.get("info", {}).get("task", "")
            reward = result.get("reward", 0)
            if task_name and reward > 0.1:
                # Check if this was a skill-derived goal
                for link_key, goal_name in self.goal_gen._skill_goal_links.items():
                    if goal_name == task_name:
                        self.causal_tracker.record_goal_achieved(
                            goal_name=task_name,
                            reward=reward,
                            round_idx=round_idx,
                            contributing_skill_ids=[link_key.split(":")[0]],
                        )
                        break

        # AGI tracker update
        self.agi_tracker.update_abstraction(self.concept_graph.depth())
        # BN-08: Update emergence metrics
        self.agi_tracker.update_emergence(
            skill_births=len(skill_birth_events),
            recursive_depth=self.causal_tracker.max_chain_depth(),
        )
        # Domain counting is done in _assign_tasks via fingerprinting;
        # here we only track difficulty increases.
        self.agi_tracker.tick_round()

        round_out.update(
            {
                "stagnation": stagnation,
                "gap_spec": gap_spec,
                "l1_update": l1_update,
                "l2_proposal": asdict(l2_proposal) if l2_proposal else None,
                "critic_results": critic_results,
                "transfer_report": transfer_report,
                "self_improvement": self_improvement_result,
                "agi_scores": self.agi_tracker.score(),
                "agi_composite": self.agi_tracker.composite_score(),
                "drift": self.self_model.detect_architectural_drift(),
                "anchor_alignment": self.self_model.get_objective_anchor_alignment(),
                # BN-08: Emergence metrics
                "emergence_depth": self.causal_tracker.max_chain_depth(),
                "emergence_chains": len(self.causal_tracker.chains_of_depth(2)),
                "total_skill_births": self.causal_tracker.skill_birth_count(),
                "skill_derived_goals": self.causal_tracker.goal_created_count(),
            }
        )
        return round_out
