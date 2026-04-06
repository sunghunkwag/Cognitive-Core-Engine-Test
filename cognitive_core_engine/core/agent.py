from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cognitive_core_engine.core.world_model import WorldModel
from cognitive_core_engine.core.planner import Planner
from cognitive_core_engine.core.skills import Skill, SkillStep, SkillLibrary
from cognitive_core_engine.core.tools import ToolRegistry
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.project_graph import ProjectNode

from agi_modules.intrinsic_motivation import IntrinsicMotivationModule
from agi_modules.self_model import SelfModel
from agi_modules.concept_graph import ConceptGraph
from agi_modules.hierarchical_planner import HierarchicalPlanner


@dataclass
class AgentConfig:
    name: str
    role: str = "general"     # "theorist" | "builder" | "experimenter" | "verifier" | "strategist"
    planner_depth: int = 3
    planner_width: int = 6
    risk: float = 0.2
    extrinsic_weight: float = 0.6   # Blend ratio for extrinsic reward
    intrinsic_weight: float = 0.4   # Blend ratio for intrinsic reward


class Agent:
    """
    B-type core:
    - WorldModel + Planner
    - SharedMemory + SkillLibrary + ToolRegistry
    - No self-modifying code; only state/memory/skills evolve.
    """

    def __init__(self, cfg: AgentConfig, tools: ToolRegistry,
                 shared_mem: SharedMemory, skills: SkillLibrary,
                 intrinsic_motivation: Optional[IntrinsicMotivationModule] = None,
                 self_model: Optional[SelfModel] = None,
                 concept_graph: Optional[ConceptGraph] = None) -> None:
        self.cfg = cfg
        self.tools = tools
        self.mem = shared_mem
        self.skills = skills

        self.wm = WorldModel()
        # v5: Adaptive planning - planner will be recreated dynamically
        self.planner = Planner(self.wm, depth=cfg.planner_depth,
                               width=cfg.planner_width)
        # v5: Agent specialization tracking
        self.domain_expertise: Dict[str, float] = {}
        # AGI modules
        self.intrinsic_motivation = intrinsic_motivation
        self.self_model = self_model
        self.concept_graph = concept_graph
        self.hierarchical_planner: Optional[HierarchicalPlanner] = None
        if concept_graph is not None:
            self.hierarchical_planner = HierarchicalPlanner(self.wm, concept_graph)
        self._skip_requested = False

    def action_space(self) -> List[str]:
        base = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]
        r = self.cfg.role
        if r == "verifier":
            return ["write_verified_note", "build_tool", "tune_orchestration", "attempt_breakthrough"]
        if r == "builder":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "theorist":
            return ["attempt_breakthrough", "write_verified_note", "build_tool", "tune_orchestration"]
        if r == "experimenter":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "strategist":
            return ["tune_orchestration", "attempt_breakthrough", "build_tool", "write_verified_note"]
        return base

    def choose_action(self, obs: Dict[str, Any]) -> str:
        # v5: Adaptive planning depth based on task difficulty
        difficulty = int(obs.get('difficulty', 3))

        # Robustness: Bound adaptive parameters
        adaptive_depth = min(10, max(2, difficulty))
        adaptive_width = min(12, max(4, 4 + difficulty // 2))

        # Recreate planner with adaptive parameters
        self.planner = Planner(self.wm, depth=adaptive_depth, width=adaptive_width)

        # System 1: Fast heuristic planning
        try:
            candidates = self.planner.propose(obs, self.action_space(), self.cfg.risk)
        except Exception:
            candidates = []

        if not candidates:
            return random.choice(self.action_space())

        draft_action = candidates[0].actions[0]
        task = obs.get('task', '')

        # v4: Improved System 2 with balanced success/failure analysis
        # Look at BOTH successes and failures
        past_episodes = self.mem.search(
            query=f"{task} {draft_action}",
            k=8,
            kinds=["episode"],
            tags=[task, draft_action]
        )

        success_count = 0
        failure_count = 0
        total_reward = 0.0

        for mem in past_episodes:
            if mem.content.get("action") == draft_action:
                reward = float(mem.content.get("reward", 0.0))
                total_reward += reward
                if reward >= 0.25:  # Success threshold
                    success_count += 1
                elif reward < 0.10:  # Failure threshold
                    failure_count += 1

        # v4: Probabilistic risk assessment instead of hard override
        if success_count + failure_count > 0:
            success_rate = success_count / (success_count + failure_count)
            avg_reward = total_reward / len(past_episodes) if past_episodes else 0.0

            # If this action has consistently failed, penalize it in exploration
            if success_rate < 0.3 and failure_count >= 3:
                # Don't completely avoid, but reduce probability
                if random.random() < 0.6 and len(candidates) > 1:
                    # Try second-best candidate
                    return candidates[1].actions[0]

        # Curiosity-boosted exploration (intrinsic motivation)
        # Uses deterministic hash to avoid perturbing global random state
        if self.intrinsic_motivation is not None:
            best_curiosity = 0.0
            best_curious_action = None
            for action in self.action_space():
                curiosity = self.intrinsic_motivation.curiosity_for_action(obs, action)
                if curiosity > 0.7 and curiosity > best_curiosity and action != draft_action:
                    best_curiosity = curiosity
                    best_curious_action = action
            if best_curious_action is not None:
                # Deterministic boost: use hash of obs to decide
                h = hash(str(obs.get("task", "")) + str(obs.get("difficulty", 0)))
                if h % 5 < 2:  # ~40% probability, 1.5x effective boost
                    return best_curious_action

        # Standard exploration vs exploitation
        if random.random() > self.cfg.risk:
            return draft_action
        return random.choice(self.action_space())

    def maybe_synthesize_skill(self, obs: Dict[str, Any]) -> Optional[str]:
        task = obs.get("task", "")
        if task == "verification_pipeline" and random.random() < 0.30:
            sk = Skill(
                name=f"{self.cfg.name}_verify_pipeline",
                purpose="Evaluate candidate and write verified note if passing.",
                steps=[
                    SkillStep(
                        kind="call",
                        tool="evaluate_candidate",
                        args_template={"task": "${task}", "candidate": "${candidate}"},
                    ),
                    SkillStep(
                        kind="if",
                        condition={"key": "last_verdict", "op": "eq", "value": "pass"},
                        steps=[
                            SkillStep(
                                kind="call",
                                tool="write_note",
                                args_template={"title": "verified_result", "payload": "${step_0}"},
                            )
                        ],
                        else_steps=[
                            SkillStep(
                                kind="call",
                                tool="write_note",
                                args_template={"title": "needs_revision", "payload": "${step_0}"},
                            )
                        ],
                    ),
                ],
                tags=["verification", "meta"],
            )
            return self.skills.add(sk)
        if task == "toolchain_speedup" and random.random() < 0.30:
            sk = Skill(
                name=f"{self.cfg.name}_toolchain_upgrade",
                purpose="Propose toolchain improvement artifact for each hint.",
                steps=[
                    SkillStep(
                        kind="foreach",
                        list_key="hint_titles",
                        item_key="hint",
                        steps=[
                            SkillStep(
                                kind="call",
                                tool="tool_build_report",
                                args_template={"task": "${task}", "idea": {"hint": "${hint}"}},
                            ),
                            SkillStep(
                                kind="call",
                                tool="write_artifact",
                                args_template={"title": "tool_artifact", "payload": "${last}"},
                            ),
                        ],
                    )
                ],
                tags=["toolchain", "artifact"],
            )
            return self.skills.add(sk)
        return None

    def act_on_project(self, env: ResearchEnvironment,
                       proj_node: ProjectNode,
                       obs: Dict[str, Any]) -> Dict[str, Any]:
        # Self-model: check if we should attempt this task
        self._skip_requested = False
        if self.self_model is not None:
            task_proxy = type('T', (), {
                'domain': obs.get('domain', ''),
                'difficulty': obs.get('difficulty', 3),
                'baseline': obs.get('baseline', 0.3),
            })()
            should, reason = self.self_model.should_attempt(task_proxy)
            if not should:
                self._skip_requested = True
                return {
                    "agent": self.cfg.name,
                    "role": self.cfg.role,
                    "project_id": proj_node.id,
                    "project_name": proj_node.name,
                    "action": "skip",
                    "reward": 0.0,
                    "mem_id": "",
                    "info": {"task": obs.get("task", ""), "domain": obs.get("domain", ""),
                             "skip_reason": reason, "skipped": True,
                             "tq": env.global_tool_quality,
                             "kq": env.global_kb_quality,
                             "oq": env.global_org_quality},
                }

        hints = self.mem.search(
            f"{obs.get('task','')} difficulty {obs.get('difficulty',0)}",
            k=6,
            kinds=["principle", "artifact", "note"],
        )

        context = {
            "task": obs.get("task"),
            "domain": obs.get("domain"),
            "difficulty": obs.get("difficulty"),
            "budget": obs.get("budget"),
            "project": {"id": proj_node.id, "name": proj_node.name},
            "candidate": {
                "type": "proposal",
                "from": self.cfg.name,
                "role": self.cfg.role,
                "hints": [h.title for h in hints],
            },
            "idea": {
                "from": self.cfg.name,
                "summary": "incremental improvement on project using accumulated tools/kb/org.",
            },
            "hint_titles": [h.title for h in hints],
        }

        sid = self.maybe_synthesize_skill(obs)
        if sid:
            self.mem.add(
                "artifact",
                f"skill_added:{sid}",
                {"agent": self.cfg.name, "skill_id": sid},
                tags=["skill"],
            )

        action = self.choose_action(obs)
        invest = max(1.0, float(obs.get("budget", 1)) / 10.0)
        payload = {
            "invest": invest,
            "agent": self.cfg.name,
            "role": self.cfg.role,
            "task": obs.get("task"),
            "project_id": proj_node.id,
        }

        next_obs, reward, info = env.step(obs, action, payload)

        # Intrinsic motivation: compute and blend with extrinsic reward
        combined_reward = reward
        intrinsic_val = 0.0
        if self.intrinsic_motivation is not None:
            outcome = {"reward": reward, "action": action, "info": info}
            intrinsic_val = self.intrinsic_motivation.total_intrinsic_reward(
                obs, action, outcome)
            combined_reward = (self.cfg.extrinsic_weight * reward +
                               self.cfg.intrinsic_weight * intrinsic_val)

        self.wm.update(obs, action, combined_reward, next_obs, self.action_space())

        # Self-model: update and diagnose failures
        if self.self_model is not None:
            result_entry = {
                "domain": obs.get("domain", ""),
                "reward": reward,
                "action": action,
                "info": info,
            }
            self.self_model.update(result_entry)
            self.self_model.record_actual(reward)
            if reward < 0.3:
                task_proxy = type('T', (), {
                    'domain': obs.get('domain', ''),
                    'difficulty': obs.get('difficulty', 3),
                    'baseline': obs.get('baseline', 0.3),
                })()
                self.self_model.diagnose_failure(task_proxy, result_entry)

        # Concept formation: record successful action patterns
        # Threshold 0.05: environment rewards are typically 0.02-0.15
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))
        if self.concept_graph is not None and reward > 0.05:
            concept_ctx = {"domain": domain, "difficulty": difficulty,
                           "action": action, "reward": reward}
            cid = self.concept_graph.add_concept(
                name=f"{action}@{domain}",
                level=0, children=[], context=concept_ctx,
                creation_round=0)
            self.concept_graph.record_usage(cid, reward, concept_ctx, success=True)
            self.mem.add("note", f"concept_formed:{cid}",
                         {"concept_id": cid, "action": action, "domain": domain},
                         tags=["concept"])

        mem_id = self.mem.add(
            "episode",
            f"{self.cfg.name}:{action}:{obs.get('task')}:{proj_node.name}",
            {
                "obs": obs,
                "action": action,
                "payload": payload,
                "reward": reward,
                "intrinsic_reward": intrinsic_val,
                "combined_reward": combined_reward,
                "info": info,
                "project_id": proj_node.id,
                "hints_used": [h.id for h in hints],
            },
            tags=["episode", self.cfg.role, obs.get("task", "task")],
        )

        if random.random() < 0.35:
            tag = "verification" if action == "write_verified_note" else "toolchain"
            candidates = self.skills.list(tag=tag)
            if candidates:
                sk = random.choice(candidates)
                out = sk.run(self.tools, context)
                self.mem.add(
                    "note",
                    f"{self.cfg.name}:skill_run:{sk.name}",
                    {"skill_id": sk.id, "out": out},
                    tags=["skill_run", tag],
                )

        return {
            "agent": self.cfg.name,
            "role": self.cfg.role,
            "project_id": proj_node.id,
            "project_name": proj_node.name,
            "action": action,
            "reward": reward,
            "mem_id": mem_id,
            "info": info,
        }
