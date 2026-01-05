"""
NON_RSI_AGI_CORE_v5.py
======================

Architecture goal:
- Fixed source code (no code-level RSI).
- AGI-oriented B×C structure:
  - B: world-model + planner + memory + skill-DSL interpreter (per agent)
  - C: multi-agent orchestrator + project/goal graph + evaluation/selection
- Self-improvement happens only via:
  - parameter updates (world model)
  - knowledge/memory accumulation
  - data-level skill programs
  - project graph + org policy adaptation
  NOT via modifying this file.

v5 Upgrade:
- "Real" Neuro-Symbolic Core using Hyperdimensional Computing (HDC).
- Strict Majority Rule for bundling (no OR hacks).
- Adaptive Planning robustness.
- Enhanced associative memory.

Run:
  python NON_RSI_AGI_CORE_v5.py --rounds 40 --agents 8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Utility
# ----------------------------

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    buf: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf


def load_unified_critic_module() -> Any:
    module_path = Path(__file__).with_name("unified_rsi_extended .py")
    spec = importlib.util.spec_from_file_location("unified_rsi_extended", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load critic module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------
# Hyperdimensional Computing (HDC) Core
# ----------------------------

class HyperVector:
    """
    Pure Python Hyperdimensional Vector implementation (10,000 bits).
    Uses strict Majority Rule for bundling.
    """
    DIM = 10000

    def __init__(self, val: Optional[int] = None) -> None:
        if val is None:
            self.val = random.getrandbits(self.DIM)
        else:
            self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperVector):
            return NotImplemented
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

    @classmethod
    def from_seed(cls, seed_obj: Any) -> HyperVector:
        """Deterministic generation from a seed object."""
        # Create a deterministic seed from the object string
        s = str(seed_obj)
        h_hex = hashlib.sha256(s.encode("utf-8")).hexdigest()
        h_int = int(h_hex, 16)
        rng = random.Random(h_int)
        return cls(rng.getrandbits(cls.DIM))

    @classmethod
    def zero(cls) -> HyperVector:
        return cls(0)

    def bind(self, other: HyperVector) -> HyperVector:
        """XOR binding operation."""
        return HyperVector(self.val ^ other.val)

    def permute(self, shifts: int = 1) -> HyperVector:
        """Cyclic shift."""
        shifts %= self.DIM
        if shifts == 0:
            return self
        mask = (1 << self.DIM) - 1
        new_val = ((self.val << shifts) & mask) | (self.val >> (self.DIM - shifts))
        return HyperVector(new_val)

    def similarity(self, other: HyperVector) -> float:
        """Hamming similarity (normalized 0.0 to 1.0)."""
        diff = self.val ^ other.val
        dist = diff.bit_count()
        return 1.0 - (dist / self.DIM)

    @staticmethod
    def bundle(vectors: List[HyperVector]) -> HyperVector:
        """
        Majority Rule bundling.
        Sum bits column-wise. Threshold at N/2.
        Optimized for pure Python using string manipulation.
        """
        if not vectors:
            return HyperVector.zero()

        n = len(vectors)
        if n == 1:
            return vectors[0]

        threshold = n / 2.0
        counts = [0] * HyperVector.DIM

        # Optimization: String iteration is faster than bitwise loops in Python
        for vec in vectors:
            # bin(val) -> '0b101...', slice [2:], zfill to DIM
            # Reverse so index 0 corresponds to LSB
            s = bin(vec.val)[2:].zfill(HyperVector.DIM)[::-1]
            for i, char in enumerate(s):
                if char == '1':
                    counts[i] += 1

        result_val = 0
        for i in range(HyperVector.DIM):
            c = counts[i]
            if c > threshold:
                result_val |= (1 << i)
            elif c == threshold:
                # Random tie-breaking
                if random.random() < 0.5:
                    result_val |= (1 << i)

        return HyperVector(result_val)


# ----------------------------
# Shared Memory / Knowledge Base (Neuro-Symbolic)
# ----------------------------

@dataclass
class MemoryItem:
    ts_ms: int
    kind: str               # "episode" | "note" | "artifact" | "principle"
    title: str
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {"ts": self.ts_ms, "k": self.kind, "t": self.title, "c": self.content, "tags": self.tags}
        )


class SharedMemory:
    """
    Shared KB using HDC for associative retrieval.
    """

    def __init__(self, max_items: int = 8000) -> None:
        self.max_items = max_items
        self._items: List[MemoryItem] = []
        # HDC Memory Index
        self._item_vectors: Dict[str, HyperVector] = {}
        # Cache common vectors to speed up encoding
        self._token_cache: Dict[str, HyperVector] = {}

    def _get_token_hv(self, token: str) -> HyperVector:
        if token not in self._token_cache:
            self._token_cache[token] = HyperVector.from_seed(f"token:{token}")
        return self._token_cache[token]

    def _encode_text_bag(self, text: str) -> HyperVector:
        tokens = tokenize(text)
        if not tokens:
            return HyperVector.zero()
        vecs = [self._get_token_hv(t) for t in tokens]
        return HyperVector.bundle(vecs)

    def _encode_item(self, item: MemoryItem) -> HyperVector:
        # Bundle: Title, Kind, Tags
        # Structure: Bind(Role, Value)

        # 1. Kind
        kind_hv = self._get_token_hv(f"kind:{item.kind}")

        # 2. Title
        title_hv = self._encode_text_bag(item.title)

        # 3. Tags
        if item.tags:
            tag_vecs = [self._get_token_hv(f"tag:{t}") for t in item.tags]
            tags_hv = HyperVector.bundle(tag_vecs)
        else:
            tags_hv = HyperVector.zero()

        # Bundle all components
        # Note: We don't bind to roles here to allow freer association,
        # or we could bind. Let's keep it simple: bundle of properties.
        return HyperVector.bundle([kind_hv, title_hv, tags_hv])

    def add(self, kind: str, title: str, content: Dict[str, Any],
            tags: Optional[List[str]] = None) -> str:
        tags = tags or []
        item = MemoryItem(ts_ms=now_ms(), kind=kind, title=title,
                          content=content, tags=tags)
        self._items.append(item)
        
        # Generate and store HV
        item_hv = self._encode_item(item)
        self._item_vectors[item.id] = item_hv
        
        if len(self._items) > self.max_items:
            removed = self._items.pop(0)
            self._item_vectors.pop(removed.id, None)

        return item.id

    def search(self, query: str, k: int = 10,
               kinds: Optional[List[str]] = None,
               tags: Optional[List[str]] = None) -> List[MemoryItem]:
        
        # 1. Encode Query
        query_parts = []
        
        # Text query
        if query:
            query_parts.append(self._encode_text_bag(query))

        # Tags query
        if tags:
            tag_vecs = [self._get_token_hv(f"tag:{t}") for t in tags]
            query_parts.append(HyperVector.bundle(tag_vecs))

        # Kinds (act as filter, but also can be part of query vector)
        if kinds:
             # We typically don't bundle all kinds, we use kinds as a hard filter.
             pass

        if not query_parts:
            return self._items[-k:]

        query_hv = HyperVector.bundle(query_parts)

        # 2. Score all items
        t_now = now_ms()
        scored: List[Tuple[float, MemoryItem]] = []

        # Optimization: Pre-filter by kind to reduce HDC checks?
        # Or just check all. 8000 checks is fine.

        for it in self._items:
            if kinds is not None and it.kind not in kinds:
                continue

            # HDC Similarity
            it_vec = self._item_vectors.get(it.id)
            if not it_vec:
                continue

            sim = query_hv.similarity(it_vec)

            # Recency & Reward boost
            recency = 1.0 / (1.0 + (t_now - it.ts_ms) / (1000.0 * 60.0 * 30.0))
            reward = float(it.content.get("reward", 0.0)) if isinstance(it.content, dict) else 0.0
            reward_boost = max(0.0, min(0.5, reward))

            # Composite Score
            # HDC similarity for random vectors is ~0.5.
            # We are interested in deviations above 0.5.
            if sim < 0.48:
                continue # Irrelevant

            # Normalize sim to 0..1 range roughly (0.5 -> 0, 1.0 -> 1)
            norm_sim = max(0.0, (sim - 0.5) * 2.0)

            final_score = norm_sim + 0.35 * recency + reward_boost

            if final_score > 0.1:
                scored.append((final_score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def extract_principles(self, k: int = 6) -> List[str]:
        episodes = [it for it in self._items if it.kind == "episode"]
        if not episodes:
            return []
        episodes.sort(key=lambda it: float(it.content.get("reward", 0.0)), reverse=True)
        selected = episodes[:k]
        created: List[str] = []
        for it in selected:
            obs = it.content.get("obs", {})
            action = it.content.get("action", "")
            reward = float(it.content.get("reward", 0.0))
            conditions = {
                "task": obs.get("task"),
                "domain": obs.get("domain"),
                "difficulty": obs.get("difficulty"),
                "phase": obs.get("phase"),
                "action": action,
            }
            pid = self.add(
                "principle",
                f"pattern:{obs.get('task','task')}:{action}",
                {
                    "conditions": conditions,
                    "reward": reward,
                    "source_episode": it.id,
                },
                tags=["principle", "derived"],
            )
            created.append(pid)
        return created

    def dump_summary(self, k: int = 15) -> List[Dict[str, Any]]:
        tail = self._items[-k:]
        return [
            {
                "id": it.id,
                "ts_ms": it.ts_ms,
                "kind": it.kind,
                "title": it.title,
                "tags": it.tags,
            }
            for it in tail
        ]


# ----------------------------
# Tool interface (external world hook)
# ----------------------------

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown_tool:{name}", "tool": name}
        try:
            out = fn(args)
            out = dict(out)
            out.setdefault("ok", True)
            out.setdefault("tool", name)
            return out
        except Exception as e:
            return {"ok": False, "error": repr(e), "tool": name}


# ----------------------------
# Skill DSL (data-level programs)
# ----------------------------

@dataclass
class SkillStep:
    kind: str
    tool: Optional[str] = None
    args_template: Optional[Dict[str, Any]] = None
    condition: Optional[Dict[str, Any]] = None
    steps: Optional[List["SkillStep"]] = None
    else_steps: Optional[List["SkillStep"]] = None
    list_key: Optional[str] = None
    item_key: Optional[str] = None


@dataclass
class Skill:
    """
    Interpreted skill program:
    - steps are data structures with explicit control-flow
    - supports: call, if, foreach
    - arguments can reference context via ${key}
    """
    name: str
    purpose: str
    steps: List[SkillStep]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {
                "name": self.name,
                "purpose": self.purpose,
                "steps": [self._serialize_step(s) for s in self.steps],
            }
        )

    def _serialize_step(self, step: SkillStep) -> Dict[str, Any]:
        return {
            "kind": step.kind,
            "tool": step.tool,
            "args_template": step.args_template,
            "condition": step.condition,
            "list_key": step.list_key,
            "item_key": step.item_key,
            "steps": [self._serialize_step(s) for s in step.steps] if step.steps else None,
            "else_steps": [self._serialize_step(s) for s in step.else_steps] if step.else_steps else None,
        }

    def run(self, tools: ToolRegistry, context: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        ctx = dict(context)

        def subst(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                key = value[2:-1]
                return ctx.get(key)
            if isinstance(value, dict):
                return {k: subst(v) for k, v in value.items()}
            if isinstance(value, list):
                return [subst(v) for v in value]
            return value

        def eval_condition(cond: Dict[str, Any]) -> bool:
            key = cond.get("key")
            op = cond.get("op", "truthy")
            val = cond.get("value")
            cur = ctx.get(key)
            if op == "eq":
                return cur == val
            if op == "neq":
                return cur != val
            if op == "contains":
                return isinstance(cur, (list, str)) and val in cur
            if op == "gt":
                return isinstance(cur, (int, float)) and cur > val
            if op == "lt":
                return isinstance(cur, (int, float)) and cur < val
            if op == "gte":
                return isinstance(cur, (int, float)) and cur >= val
            if op == "lte":
                return isinstance(cur, (int, float)) and cur <= val
            return bool(cur)

        def run_steps(steps: Iterable[SkillStep], depth: int = 0) -> bool:
            if depth > 12:
                return False
            for i, st in enumerate(steps):
                if st.kind == "call" and st.tool:
                    args = subst(st.args_template or {})
                    if not isinstance(args, dict):
                        args = {"value": args}
                    res = tools.call(st.tool, args)
                    trace.append({"i": len(trace), "tool": st.tool, "args": args, "res": res})
                    ctx["last"] = res
                    if isinstance(res, dict):
                        ctx["last_verdict"] = res.get("verdict")
                    ctx[f"step_{len(trace) - 1}"] = res
                    if not res.get("ok", False):
                        return False
                elif st.kind == "if" and st.condition:
                    branch = st.steps if eval_condition(st.condition) else st.else_steps
                    if branch:
                        if not run_steps(branch, depth + 1):
                            return False
                elif st.kind == "foreach" and st.list_key:
                    items = ctx.get(st.list_key, [])
                    if isinstance(items, list) and st.steps:
                        for idx, item in enumerate(items):
                            ctx[st.item_key or "item"] = item
                            ctx["index"] = idx
                            if not run_steps(st.steps, depth + 1):
                                return False
                else:
                    return False
            return True

        ok = run_steps(self.steps)
        return {
            "ok": ok,
            "trace": trace,
            "final": ctx.get("last"),
        }


class SkillLibrary:
    def __init__(self, max_skills: int = 3000) -> None:
        self.max_skills = max_skills
        self._skills: Dict[str, Skill] = {}

    def add(self, sk: Skill) -> str:
        self._skills[sk.id] = sk
        if len(self._skills) > self.max_skills:
            for sid in list(self._skills.keys())[: len(self._skills) - self.max_skills]:
                self._skills.pop(sid, None)
        return sk.id

    def list(self, tag: Optional[str] = None) -> List[Skill]:
        vals = list(self._skills.values())
        if tag is None:
            return vals
        return [s for s in vals if tag in s.tags]

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)


# ----------------------------
# World Model (feature-based value model)
# ----------------------------

@dataclass
class TransitionSummary:
    count: int = 0


class WorldModel:
    """
    Feature-based Q-value model with v5 enhancements:
    - Non-linear feature combinations
    - Experience replay buffer
    - online TD updates
    - separate state-action counts for uncertainty estimates
    """

    def __init__(self, gamma: float = 0.9, lr: float = 0.08) -> None:
        self.gamma = gamma
        self.lr = lr
        self._weights: Dict[str, float] = {}
        self._sa_counts: Dict[Tuple[str, str], TransitionSummary] = {}
        # v5: Experience replay buffer
        self.replay_buffer: List[Tuple[Dict[str, Any], str, float, Dict[str, Any], List[str]]] = []
        self.max_buffer_size = 200

    def _feature_bucket(self, budget: int) -> int:
        return min(5, max(0, budget // 10))

    def encode_state(self, obs: Dict[str, Any]) -> str:
        key = {
            "task": obs.get("task", ""),
            "domain": obs.get("domain", ""),
            "difficulty": int(obs.get("difficulty", 0)),
            "budget": int(obs.get("budget", 0)),
            "phase": obs.get("phase", ""),
        }
        return stable_hash(key)

    def features(self, obs: Dict[str, Any], action: str) -> Dict[str, float]:
        task = str(obs.get("task", ""))
        domain = str(obs.get("domain", ""))
        diff = int(obs.get("difficulty", 0))
        phase = str(obs.get("phase", ""))
        budget = int(obs.get("budget", 0))
        bucket = self._feature_bucket(budget)
        
        # v5: Non-linear feature combinations
        feats = {
            "bias": 1.0,
            f"task:{task}": 1.0,
            f"domain:{domain}": 1.0,
            f"diff:{diff}": 1.0,
            f"phase:{phase}": 1.0,
            f"action:{action}": 1.0,
            f"task_action:{task}|{action}": 1.0,
            f"budget_bucket:{bucket}": 1.0,
            # Non-linear combinations
            f"diff_action:{diff}|{action}": 1.0,
            f"domain_diff:{domain}|{diff}": float(diff) / 5.0,
            f"task_phase:{task}|{phase}": 1.0,
        }
        return feats

    def q_value(self, obs: Dict[str, Any], action: str) -> float:
        feats = self.features(obs, action)
        return sum(self._weights.get(k, 0.0) * v for k, v in feats.items())

    def confidence(self, obs: Dict[str, Any], action: str) -> float:
        s = self.encode_state(obs)
        count = self._sa_counts.get((s, action), TransitionSummary()).count
        return 1.0 - (1.0 / math.sqrt(count + 1.0))

    def update(self, obs: Dict[str, Any], action: str, reward: float,
               next_obs: Dict[str, Any], action_space: List[str]) -> None:
        # v5: Add to replay buffer
        self.replay_buffer.append((obs, action, reward, next_obs, action_space))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
        
        # Current experience update
        feats = self.features(obs, action)
        current = self.q_value(obs, action)
        next_best = max(self.q_value(next_obs, a) for a in action_space)
        target = reward + self.gamma * next_best
        td_error = target - current
        for k, v in feats.items():
            self._weights[k] = self._weights.get(k, 0.0) + self.lr * td_error * v
        
        # v5: Experience replay (sample mini-batch)
        if len(self.replay_buffer) >= 10:
            samples = random.sample(self.replay_buffer, min(5, len(self.replay_buffer)))
            for s_obs, s_action, s_reward, s_next_obs, s_action_space in samples:
                s_feats = self.features(s_obs, s_action)
                s_current = self.q_value(s_obs, s_action)
                s_next_best = max(self.q_value(s_next_obs, a) for a in s_action_space)
                s_target = s_reward + self.gamma * s_next_best
                s_td_error = s_target - s_current
                for k, v in s_feats.items():
                    self._weights[k] = self._weights.get(k, 0.0) + (self.lr * 0.5) * s_td_error * v
        
        # Update visitation counts
        s = self.encode_state(obs)
        entry = self._sa_counts.get((s, action))
        if entry is None:
            entry = TransitionSummary()
            self._sa_counts[(s, action)] = entry
        entry.count += 1


# ----------------------------
# Planner (lookahead over world model)
# ----------------------------

@dataclass
class PlanCandidate:
    actions: List[str]
    score: float


class Planner:
    def __init__(self, wm: WorldModel, depth: int = 3,
                 width: int = 6, gamma: float = 0.9) -> None:
        self.wm = wm
        self.depth = depth
        self.width = width
        self.gamma = gamma

    def propose(self, obs: Dict[str, Any], action_space: List[str],
                risk_pref: float) -> List[PlanCandidate]:
        # Robustness: Safety check
        if not action_space:
            return []

        beam: List[PlanCandidate] = [PlanCandidate(actions=[], score=0.0)]

        for d in range(self.depth):
            new_beam: List[PlanCandidate] = []
            for cand in beam:
                for a in action_space:
                    q = self.wm.q_value(obs, a)
                    uncertainty = 1.0 - self.wm.confidence(obs, a)
                    adjusted = q - (1.0 - risk_pref) * uncertainty
                    sc = cand.score + (self.gamma ** d) * adjusted
                    new_beam.append(PlanCandidate(actions=cand.actions + [a], score=sc))

            # Robustness: Sort and Prune
            if not new_beam:
                break

            new_beam.sort(key=lambda c: c.score, reverse=True)
            beam = new_beam[: self.width]

        return beam


# ----------------------------
# Project / Goal Graph (C-layer long-horizon structure)
# ----------------------------

@dataclass
class ProjectNode:
    id: str
    name: str
    task: str
    status: str = "open"      # "open" | "active" | "done"
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    value_estimate: float = 0.0
    history: List[str] = field(default_factory=list)  # memory ids
    value_history: List[float] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)


class ProjectGraph:
    """
    Long-horizon project DAG:
    - orchestrator attaches agent runs to nodes
    - nodes accumulate evidence and value estimates
    - spawn subprojects based on value thresholds
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProjectNode] = {}

    def create_root(self, name: str, task: str) -> str:
        nid = stable_hash({"name": name, "task": task, "root": True})
        self._nodes[nid] = ProjectNode(id=nid, name=name, task=task, status="open")
        return nid

    def add_child(self, parent_id: str, name: str,
                  task: Optional[str] = None) -> str:
        parent = self._nodes[parent_id]
        nid = stable_hash({"name": name, "task": task or parent.task, "parent": parent_id})
        node = ProjectNode(id=nid, name=name, task=task or parent.task,
                           status="open", parent_id=parent_id)
        self._nodes[nid] = node
        parent.children.append(nid)
        return nid

    def nodes_for_task(self, task: str) -> List[ProjectNode]:
        return [n for n in self._nodes.values() if n.task == task]

    def pick_node_for_round(self, task: str) -> ProjectNode:
        candidates = [n for n in self._nodes.values()
                      if n.task == task and n.status != "done"]
        if not candidates:
            nid = self.create_root(name=f"{task}_root", task=task)
            return self._nodes[nid]
        candidates.sort(key=lambda n: n.value_estimate, reverse=True)
        return candidates[0]

    def update_node(self, nid: str, reward: float,
                    memory_id: Optional[str]) -> None:
        node = self._nodes[nid]
        alpha = 0.25
        node.value_estimate = (1 - alpha) * node.value_estimate + alpha * reward
        node.value_history.append(node.value_estimate)
        if memory_id:
            node.history.append(memory_id)
            node.evidence_refs.append(memory_id)
        if node.value_estimate > 0.18 and len(node.children) < 3:
            self.add_child(parent_id=nid, name=f"{node.name}_infra_focus")
            self.add_child(parent_id=nid, name=f"{node.name}_breakthrough_focus")
        if node.value_estimate > 0.35:
            node.status = "active"


@dataclass
class RuleProposal:
    proposal_id: str
    level: str  # "L0" | "L1" | "L2"
    payload: Dict[str, Any]
    creator_key: str
    created_ms: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None


# ----------------------------
# Environment (research/engineering playground)
# ----------------------------

@dataclass
class TaskSpec:
    name: str
    difficulty: int
    baseline: float
    domain: str   # "algorithm" | "systems" | "theory" | "strategy" ...


class ResearchEnvironment:
    """
    Abstract multi-domain environment.
    - Each step is "run one agent on one project node for a given task/budget"
    - Reward ~ improvement over task baseline + infra gain
    - Global qualities (tool/kb/org) mediate acceleration
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tasks: List[TaskSpec] = [
            TaskSpec("algorithm_design", difficulty=3, baseline=0.35, domain="algorithm"),
            TaskSpec("systems_optimization", difficulty=4, baseline=0.30, domain="systems"),
            TaskSpec("verification_pipeline", difficulty=2, baseline=0.40, domain="verification"),
            TaskSpec("toolchain_speedup", difficulty=5, baseline=0.25, domain="engineering"),
            TaskSpec("theory_discovery", difficulty=5, baseline=0.28, domain="theory"),
            TaskSpec("strategy_optimization", difficulty=3, baseline=0.32, domain="strategy"),
        ]
        self.global_tool_quality = 0.10
        self.global_kb_quality = 0.10
        self.global_org_quality = 0.10

    def sample_task(self) -> TaskSpec:
        return self.rng.choice(self.tasks)

    def make_observation(self, task: TaskSpec, budget: int,
                         phase: str = "research") -> Dict[str, Any]:
        return {
            "task": task.name,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "baseline": task.baseline,
            "budget": budget,
            "phase": phase,
        }

    def step(self, obs: Dict[str, Any], action: str,
             payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        diff = int(obs["difficulty"])
        base = float(obs["baseline"])
        budget = int(obs["budget"])
        domain = str(obs.get("domain", ""))

        tq = self.global_tool_quality
        kq = self.global_kb_quality
        oq = self.global_org_quality

        infra_scale = 1.0 / (1.0 + 0.4 * diff)
        leverage = 0.30 * tq + 0.30 * kq + 0.30 * oq
        diminishing = 1.0 / (1.0 + 2.0 * leverage)

        domain_bonus = {
            "algorithm": 0.04 if action == "attempt_breakthrough" else 0.01,
            "theory": 0.05 if action == "attempt_breakthrough" else 0.01,
            "systems": 0.04 if action in ("build_tool", "tune_orchestration") else 0.01,
            "engineering": 0.05 if action == "build_tool" else 0.01,
            "verification": 0.05 if action == "write_verified_note" else 0.01,
            "strategy": 0.04 if action == "tune_orchestration" else 0.01,
        }.get(domain, 0.01)

        if action == "build_tool":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.12 * tq) * invest * infra_scale * diminishing
            self.global_tool_quality = min(1.0, self.global_tool_quality + gain)
            raw = 0.02 * invest + domain_bonus
        elif action == "write_verified_note":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.10 * kq) * invest * infra_scale * diminishing
            self.global_kb_quality = min(1.0, self.global_kb_quality + gain)
            raw = 0.018 * invest + domain_bonus
        elif action == "tune_orchestration":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.10 * oq) * invest * infra_scale * diminishing
            self.global_org_quality = min(1.0, self.global_org_quality + gain)
            raw = 0.016 * invest + domain_bonus
        elif action == "attempt_breakthrough":
            effort = (1.0 + math.log(1 + budget) / 4.0)
            raw = (0.04 + 0.32 * leverage) * effort * (1.0 / (1.0 + 0.30 * diff)) + domain_bonus
        else:
            raw = 0.0

        noise = self.rng.uniform(-0.02, 0.02)
        performance = max(0.0, min(1.0, base + raw + noise))
        delta = performance - base
        infra_bonus = 0.025 * (tq + kq + oq) / 3.0
        reward = delta + infra_bonus

        next_obs = dict(obs)
        next_obs["phase"] = "integrate"
        info = {
            "task": obs.get("task"),
            "performance": performance,
            "delta": delta,
            "tq": self.global_tool_quality,
            "kq": self.global_kb_quality,
            "oq": self.global_org_quality,
        }
        return next_obs, reward, info


# ----------------------------
# Agent (B-type architecture)
# ----------------------------

@dataclass
class AgentConfig:
    name: str
    role: str = "general"     # "theorist" | "builder" | "experimenter" | "verifier" | "strategist"
    planner_depth: int = 3
    planner_width: int = 6
    risk: float = 0.2


class Agent:
    """
    B-type core:
    - WorldModel + Planner
    - SharedMemory + SkillLibrary + ToolRegistry
    - No self-modifying code; only state/memory/skills evolve.
    """

    def __init__(self, cfg: AgentConfig, tools: ToolRegistry,
                 shared_mem: SharedMemory, skills: SkillLibrary) -> None:
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
        self.wm.update(obs, action, reward, next_obs, self.action_space())

        mem_id = self.mem.add(
            "episode",
            f"{self.cfg.name}:{action}:{obs.get('task')}:{proj_node.name}",
            {
                "obs": obs,
                "action": action,
                "payload": payload,
                "reward": reward,
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


# ----------------------------
# Orchestrator (C-layer: multi-agent + project graph)
# ----------------------------

@dataclass
class OrchestratorConfig:
    agents: int = 8
    base_budget: int = 20
    selection_top_k: int = 4
    budget_growth: float = 1.06


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
            "min_score": 0.42,
            "l1_update_rate": 0.08,
            "min_transfer": 0.05,
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
            self._agents.append(Agent(cfg, self.tools, self.mem, self.skills))

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
        }

    def _omega_generate_candidates(self, gap_spec: Dict[str, Any]) -> List[RuleProposal]:
        import omega_forge_two_stage_feedback as omega

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
                    "metrics": {"fallback": True},
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
        return tasks

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
        for idx, ag in enumerate(self._agents):
            task = tasks[idx % len(tasks)]
            proj_node = self.projects.pick_node_for_round(task.name)
            agent_budget = self._budget_for_agent(budget, ag.cfg.role)
            obs = self.env.make_observation(task, agent_budget)
            res = ag.act_on_project(self.env, proj_node, obs)
            results.append(res)
            self.projects.update_node(proj_node.id, res["reward"], res["mem_id"])

        self._distill_principles(round_idx, results)

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
        }

    def run_recursive_cycle(
        self,
        round_idx: int,
        stagnation_override: Optional[bool] = None,
        force_meta_proposal: bool = False,
    ) -> Dict[str, Any]:
        round_out = self.run_round(round_idx)
        self._record_round_rewards(round_out["results"])

        stagnation = stagnation_override if stagnation_override is not None else self._detect_stagnation()
        gap_spec = None
        if stagnation:
            gap_spec = self._build_gap_spec(round_idx, round_out)
            self.candidate_queue.extend(self._omega_generate_candidates(gap_spec))

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
        while self.candidate_queue:
            proposal = self.candidate_queue.pop(0)
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

        round_out.update(
            {
                "stagnation": stagnation,
                "gap_spec": gap_spec,
                "l1_update": l1_update,
                "l2_proposal": asdict(l2_proposal) if l2_proposal else None,
                "critic_results": critic_results,
            }
        )
        return round_out


# ----------------------------
# Minimal tools (replace with real-world hooks)
# ----------------------------

def tool_write_note_factory(shared_mem: SharedMemory) -> ToolFn:
    def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title", "note"))
        payload = args.get("payload", {})
        mid = shared_mem.add("note", title, {"payload": payload},
                             tags=["tool_note"])
        return {"ok": True, "memory_id": mid, "title": title}
    return _fn


def tool_write_artifact_factory(shared_mem: SharedMemory) -> ToolFn:
    def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title", "artifact"))
        payload = args.get("payload", {})
        mid = shared_mem.add("artifact", title, {"payload": payload},
                             tags=["tool_artifact"])
        return {"ok": True, "memory_id": mid, "title": title}
    return _fn


def tool_evaluate_candidate(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task", "unknown"))
    cand = args.get("candidate", {})
    size = len(json.dumps(cand, default=str))
    score = (size % 97) / 100.0
    if "hints" in cand and isinstance(cand["hints"], list) and len(cand["hints"]) > 4:
        score *= 0.93
    verdict = "pass" if score > 0.4 else "revise"
    return {"ok": True, "task": task, "score": score, "verdict": verdict}


def tool_tool_build_report(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task", "unknown"))
    idea = args.get("idea", {})
    return {
        "ok": True,
        "task": task,
        "artifact": {
            "type": "tool_proposal",
            "idea": idea,
            "expected_effect": "increase evaluation throughput & reliability",
        },
    }


# ----------------------------
# Main entry
# ----------------------------

def run_full_system_selftest() -> None:
    random.seed(0)
    env = ResearchEnvironment(seed=0)
    tools = ToolRegistry()
    orch_cfg = OrchestratorConfig(
        agents=4,
        base_budget=12,
        selection_top_k=2,
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    orch.run_recursive_cycle(0, stagnation_override=True, force_meta_proposal=True)
    assert (round_out := orch.run_recursive_cycle(1, stagnation_override=True, force_meta_proposal=True))
    assert round_out["stagnation"] is True
    assert "gap_spec" in round_out and isinstance(round_out["gap_spec"], dict)
    assert "constraints" in round_out["gap_spec"] and isinstance(round_out["gap_spec"]["constraints"], dict)
    assert "quarantine_only" in round_out["gap_spec"]["constraints"]
    assert round_out["gap_spec"]["constraints"]["quarantine_only"] is True
    assert "no_self_adoption" in round_out["gap_spec"]["constraints"]
    assert round_out["gap_spec"]["constraints"]["no_self_adoption"] is True
    assert round_out["critic_results"]
    assert all("verdict" in item for item in round_out["critic_results"])
    assert all("proposal_id" in item for item in round_out["critic_results"])
    assert any(item["level"] == "L0" for item in round_out["critic_results"])
    assert any(item["level"] == "L1" for item in round_out["critic_results"])
    assert any(item["level"] == "L2" for item in round_out["critic_results"])
    assert all(
        (not item.get("adopted", False)) or item.get("verdict") == "approve"
        for item in round_out["critic_results"]
    )
    print("recursive rule loop executed")
    print("critic decision received")

    x = [[1.0, 2.0], [3.0, 4.0]]
    w = [[1.0], [1.0]]
    y = [
        [x[0][0] * w[0][0] + x[0][1] * w[1][0]],
        [x[1][0] * w[0][0] + x[1][1] * w[1][0]],
    ]
    assert len(y) == 2 and len(y[0]) == 1
    print("tensor execution verified (torch-free)")


def run_torch_smoke_test() -> None:
    import torch

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x @ torch.tensor([[1.0], [1.0]])
    assert y.shape == (2, 1)
    print("pytorch execution verified")


def run_contract_negative_tests() -> None:
    random.seed(1)
    env = ResearchEnvironment(seed=1)
    tools = ToolRegistry()
    orch_cfg = OrchestratorConfig(
        agents=4,
        base_budget=12,
        selection_top_k=2,
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    round_out = orch.run_recursive_cycle(0, stagnation_override=True, force_meta_proposal=True)

    def expect_failure(fn: Callable[[], None], exc_types: Tuple[type, ...], msg_substr: str) -> None:
        try:
            fn()
        except exc_types as exc:
            assert msg_substr in str(exc), f"message mismatch: {exc}"
            return
        except Exception:
            raise
        raise AssertionError("expected failure was not raised")

    assert "gap_spec" in round_out and isinstance(round_out["gap_spec"], dict)
    assert "critic_results" in round_out and isinstance(round_out["critic_results"], list)
    assert any(item.get("level") == "L0" for item in round_out["critic_results"])

    proposals = orch._omega_generate_candidates(round_out["gap_spec"])
    assert proposals
    proposal = proposals[0]
    verdict = orch._critic_evaluate(proposal)
    l1_proposal = RuleProposal(
        proposal_id="l1_negative",
        level="L1",
        payload={"evaluation_update": {"min_score": 0.5}},
        creator_key="creator",
        created_ms=now_ms(),
    )
    l2_proposal = RuleProposal(
        proposal_id="l2_negative",
        level="L2",
        payload={"meta_update": {"l1_update_rate": 0.1}},
        creator_key="creator",
        created_ms=now_ms(),
    )
    l1_verdict = orch._critic_evaluate(l1_proposal)
    l2_verdict = orch._critic_evaluate(l2_proposal)

    def validate_critic_verdict(result: Dict[str, Any]) -> None:
        assert "verdict" in result, "verdict missing"
        assert "approval_key" in result, "approval_key missing"

    def adopt_with_contract(test_proposal: RuleProposal, result: Dict[str, Any]) -> None:
        if result.get("verdict") != "approve":
            raise ValueError("verdict not approved")
        if "approval_key" not in result:
            raise ValueError("approval_key missing")
        orch._adopt_proposal(test_proposal, result)

    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="missing_candidate",
                level="L0",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="non_dict_candidate",
                level="L0",
                payload={"candidate": "not-a-dict"},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="gid_none",
                level="L0",
                payload={"candidate": {**proposal.payload["candidate"], "gid": None}},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate gid missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="gid_empty",
                level="L0",
                payload={"candidate": {**proposal.payload["candidate"], "gid": ""}},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate gid missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="l1_missing_update",
                level="L1",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "evaluation_update missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="l2_missing_update",
                level="L2",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "meta_update missing",
    )
    expect_failure(
        lambda: adopt_with_contract(l1_proposal, {**l1_verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(l1_proposal, {"verdict": "approve"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: adopt_with_contract(l2_proposal, {**l2_verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(l2_proposal, {"verdict": "approve"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: adopt_with_contract(proposal, {**verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(proposal, {k: v for k, v in verdict.items() if k != "approval_key"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: validate_critic_verdict({k: v for k, v in verdict.items() if k != "approval_key"}),
        (AssertionError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: validate_critic_verdict({k: v for k, v in verdict.items() if k != "verdict"}),
        (AssertionError,),
        "verdict missing",
    )
    print("negative contract tests passed")


def _make_benchmark_stack(seed: int) -> Tuple[ResearchEnvironment, Agent, ProjectGraph]:
    random.seed(seed)
    env = ResearchEnvironment(seed=seed)
    tools = ToolRegistry()
    mem = SharedMemory()
    skills = SkillLibrary()
    agent = Agent(AgentConfig(name="bench_agent", role="general"), tools, mem, skills)

    tools.register("write_note", tool_write_note_factory(mem))
    tools.register("write_artifact", tool_write_artifact_factory(mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    projects = ProjectGraph()
    return env, agent, projects


def _run_benchmark_step(
    env: ResearchEnvironment,
    agent: Agent,
    projects: ProjectGraph,
    task: TaskSpec,
    budget: int,
) -> Dict[str, Any]:
    obs = env.make_observation(task, budget)
    proj_node = projects.pick_node_for_round(task.name)
    return agent.act_on_project(env, proj_node, obs)


def _adb_apply_rule(rule: str, params: Dict[str, int], seq: List[int]) -> List[int]:
    if rule == "reverse":
        return list(reversed(seq))
    if rule == "sort_unique":
        out: List[int] = []
        seen = set()
        for val in sorted(seq):
            if val not in seen:
                seen.add(val)
                out.append(val)
        return out
    if rule == "add_then_filter":
        delta = params.get("delta", 0)
        threshold = params.get("threshold", 0)
        return [val + delta for val in seq if val + delta >= threshold]
    if rule == "window_sum":
        width = max(1, params.get("width", 2))
        return [sum(seq[i:i + width]) for i in range(0, len(seq), width)]
    return seq


def _generate_adb_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["reverse", "sort_unique", "add_then_filter", "window_sum"])
    params = {}
    if rule == "add_then_filter":
        params = {"delta": rng.randint(-3, 3), "threshold": rng.randint(0, 6)}
    if rule == "window_sum":
        params = {"width": rng.randint(2, 3)}

    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 6)
        inp = [rng.randint(-4, 9) for _ in range(length)]
        out = _adb_apply_rule(rule, params, inp)
        train_pairs.append({"input": inp, "output": out})

    test_length = rng.randint(6, 9)
    test_input = [rng.randint(-6, 12) for _ in range(test_length)]
    adversarial = test_input[:]
    rng.shuffle(adversarial)
    if rule == "add_then_filter":
        adversarial = [val - params.get("delta", 0) for val in adversarial]
    test_output = _adb_apply_rule(rule, params, test_input)
    adversarial_output = _adb_apply_rule(rule, params, adversarial)
    return {
        "train": train_pairs,
        "test": {"input": test_input, "output": test_output},
        "adversarial": {"input": adversarial, "output": adversarial_output},
    }


def _solve_adb(task: Dict[str, Any], test_input: List[int]) -> Tuple[Any, int]:
    attempts = 0
    train_pairs = task.get("train", [])
    if train_pairs and all(
        pair.get("output") == list(reversed(pair.get("input", []))) for pair in train_pairs
    ):
        attempts += 1
        return list(reversed(test_input)), attempts
    if train_pairs and all(
        pair.get("output") == sorted(pair.get("input", [])) for pair in train_pairs
    ):
        attempts += 1
        return sorted(test_input), attempts
    attempts += 1
    return [], attempts


def _run_adb_suite_split(seed: int, trials: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    passes = 0
    robust_passes = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for _ in range(trials):
        task = _generate_adb_task(rng)
        start = now_ms()
        base_input = task["test"]["input"]
        prediction, attempts = _solve_adb(task, base_input)
        commit_hash = stable_hash({"pred": prediction})
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        base_ok = prediction == task["test"]["output"]

        robust_ok = False
        if base_ok:
            adv_input = task["adversarial"]["input"]
            adv_prediction, _ = _solve_adb(task, adv_input)
            robust_ok = adv_prediction == task["adversarial"]["output"]
            _ = commit_hash
        if base_ok:
            passes += 1
        if base_ok and robust_ok:
            robust_passes += 1

    trials_count = max(1, trials)
    return {
        "pass_rate": passes / trials_count,
        "robust_pass_rate": robust_passes / trials_count,
        "discovery_cost": total_attempts / max(1, passes),
        "avg_runtime_ms_per_trial": sum(runtimes_ms) / max(1, len(runtimes_ms)),
    }


def run_adb_benchmark_suite(seed: int, trials: int) -> Dict[str, Any]:
    train_result = _run_adb_suite_split(seed, trials)
    holdout_seed = _derive_holdout_seed(seed)
    holdout_result = _run_adb_suite_split(holdout_seed, trials)
    return {
        "suite": "ADB_v1",
        "seed": seed,
        "trials": trials,
        "train_pass_rate": train_result["pass_rate"],
        "holdout_pass_rate": holdout_result["pass_rate"],
        "discovery_cost": {
            "train": train_result["discovery_cost"],
            "holdout": holdout_result["discovery_cost"],
        },
        "robust_pass_rate": {
            "train": train_result["robust_pass_rate"],
            "holdout": holdout_result["robust_pass_rate"],
        },
        "avg_runtime_ms_per_trial": {
            "train": train_result["avg_runtime_ms_per_trial"],
            "holdout": holdout_result["avg_runtime_ms_per_trial"],
        },
    }


def _derive_holdout_seed(base_seed: int) -> int:
    nonce = "holdout-seed-v1"
    file_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    mix = f"{base_seed}:{file_hash}:{nonce}".encode("utf-8")
    return int(hashlib.sha256(mix).hexdigest()[:8], 16)


def _generate_program_synthesis_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["reverse", "sort", "dedup"])
    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 6)
        inp = [rng.randint(-3, 9) for _ in range(length)]
        if rule == "reverse":
            out = list(reversed(inp))
        elif rule == "sort":
            out = sorted(inp)
        else:
            out = []
            seen = set()
            for val in inp:
                if val not in seen:
                    seen.add(val)
                    out.append(val)
        train_pairs.append({"input": inp, "output": out})
    test_length = rng.randint(3, 6)
    test_input = [rng.randint(-3, 9) for _ in range(test_length)]
    if rule == "reverse":
        test_output = list(reversed(test_input))
    elif rule == "sort":
        test_output = sorted(test_input)
    else:
        test_output = []
        seen = set()
        for val in test_input:
            if val not in seen:
                seen.add(val)
                test_output.append(val)
    return {"train": train_pairs, "test": {"input": test_input, "output": test_output}}


def _generate_algo_micro_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["sum", "max", "count_even"])
    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 7)
        inp = [rng.randint(-5, 12) for _ in range(length)]
        if rule == "sum":
            out = sum(inp)
        elif rule == "max":
            out = max(inp)
        else:
            out = sum(1 for v in inp if v % 2 == 0)
        train_pairs.append({"input": inp, "output": out})
    test_length = rng.randint(3, 7)
    test_input = [rng.randint(-5, 12) for _ in range(test_length)]
    if rule == "sum":
        test_output = sum(test_input)
    elif rule == "max":
        test_output = max(test_input)
    else:
        test_output = sum(1 for v in test_input if v % 2 == 0)
    return {"train": train_pairs, "test": {"input": test_input, "output": test_output}}


def _generate_robustness_task(rng: random.Random) -> Dict[str, Any]:
    length = rng.randint(4, 8)
    base_input = [rng.randint(-4, 9) for _ in range(length)]
    base_output = sum(base_input)
    return {"base_input": base_input, "base_output": base_output}


def _solve_program_synthesis(task: Dict[str, Any]) -> Tuple[Any, int]:
    train_pairs = task.get("train", [])
    attempts = 0
    if train_pairs and all(pair.get("input") == pair.get("output") for pair in train_pairs):
        attempts += 1
        return task["test"]["input"], attempts
    if train_pairs and all(
        pair.get("output") == list(reversed(pair.get("input", []))) for pair in train_pairs
    ):
        attempts += 1
        return list(reversed(task["test"]["input"])), attempts
    attempts += 1
    return [], attempts


def _solve_algo_micro(task: Dict[str, Any]) -> Tuple[Any, int]:
    attempts = 1
    return 0, attempts


def _run_hard_suite_split(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    passes = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for _ in range(trials):
        start = now_ms()
        if suite == "program_synthesis_hard_v1":
            task = _generate_program_synthesis_task(rng)
            prediction, attempts = _solve_program_synthesis(task)
            expected = task["test"]["output"]
            solved = prediction == expected
        elif suite == "algo_micro_hard_v1":
            task = _generate_algo_micro_task(rng)
            prediction, attempts = _solve_algo_micro(task)
            expected = task["test"]["output"]
            solved = prediction == expected
        elif suite == "robustness_hard_v1":
            attempts = 1
            base_task = _generate_robustness_task(rng)
            base_input = base_task["base_input"]
            expected = base_task["base_output"]
            prediction, _ = _solve_algo_micro({"input": base_input})
            solved = prediction == expected
            if solved:
                shuffled = base_input[:]
                rng.shuffle(shuffled)
                prediction, _ = _solve_algo_micro({"input": shuffled})
                solved = prediction == expected
            if solved:
                noisy = base_input[:] + [0, 0]
                rng.shuffle(noisy)
                prediction, _ = _solve_algo_micro({"input": noisy})
                solved = prediction == expected
        else:
            raise ValueError(f"unknown suite: {suite}")
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        if solved:
            passes += 1

    pass_rate = passes / max(1, trials)
    return {
        "pass_rate": pass_rate,
        "proposals_evaluated_per_solve": total_attempts / max(1, passes),
        "avg_runtime_ms_per_trial": sum(runtimes_ms) / max(1, len(runtimes_ms)),
    }


def run_hard_benchmark_suite(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    train_result = _run_hard_suite_split(suite, seed, trials)
    holdout_seed = _derive_holdout_seed(seed)
    holdout_result = _run_hard_suite_split(suite, holdout_seed, trials)
    result = {
        "suite": suite,
        "seed": seed,
        "trials": trials,
        "train_pass_rate": train_result["pass_rate"],
        "holdout_pass_rate": holdout_result["pass_rate"],
        "avg_runtime_ms_per_trial": {
            "train": train_result["avg_runtime_ms_per_trial"],
            "holdout": holdout_result["avg_runtime_ms_per_trial"],
        },
    }
    if suite == "program_synthesis_hard_v1":
        result["proposals_evaluated_per_solve"] = {
            "train": train_result["proposals_evaluated_per_solve"],
            "holdout": holdout_result["proposals_evaluated_per_solve"],
        }
    return result


def run_benchmark_suite(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    if suite == "ADB_v1":
        return run_adb_benchmark_suite(seed, trials)
    if suite in {"program_synthesis_hard_v1", "algo_micro_hard_v1", "robustness_hard_v1"}:
        return run_hard_benchmark_suite(suite, seed, trials)
    passes = 0
    total_rewards: List[float] = []
    skill_successes = 0
    attempts = 0

    for idx in range(trials):
        env, agent, projects = _make_benchmark_stack(seed + idx)

        if suite == "algo_micro_v1":
            task = next(t for t in env.tasks if t.domain == "algorithm")
            res = _run_benchmark_step(env, agent, projects, task, budget=12)
            reward = float(res.get("reward", 0.0))
            total_rewards.append(reward)
            if reward >= 0.02:
                passes += 1
        elif suite == "robustness_v1":
            rewards: List[float] = []
            for budget in (8, 12, 16):
                task = env.sample_task()
                res = _run_benchmark_step(env, agent, projects, task, budget=budget)
                rewards.append(float(res.get("reward", 0.0)))
            total_rewards.extend(rewards)
            if min(rewards) >= -0.01:
                passes += 1
        elif suite == "program_synthesis_v1":
            for _ in range(5):
                task = next(
                    t for t in env.tasks if t.name in ("verification_pipeline", "toolchain_speedup")
                )
                res = _run_benchmark_step(env, agent, projects, task, budget=12)
                total_rewards.append(float(res.get("reward", 0.0)))
                attempts += 1
            if agent.skills.list():
                passes += 1
                skill_successes += 1
        else:
            raise ValueError(f"unknown suite: {suite}")

    pass_rate = passes / max(1, trials)
    result = {
        "suite": suite,
        "seed": seed,
        "trials": trials,
        "pass_rate": pass_rate,
        "avg_reward": sum(total_rewards) / max(1, len(total_rewards)),
    }
    if suite == "program_synthesis_v1":
        proposals_per_solve = attempts / max(1, skill_successes)
        result["proposals_evaluated_per_solve"] = proposals_per_solve
    return result


def _load_arc_tasks(data_root: Path, suite: str) -> List[Dict[str, Any]]:
    if suite != "arc_agi2_public_eval":
        raise ValueError(f"unknown suite: {suite}")
    candidates = [
        data_root / "public_eval",
        data_root / "public",
        data_root / "evaluation",
        data_root / "eval",
        data_root / "public_eval_tasks",
    ]
    task_dir = next((p for p in candidates if p.exists()), None)
    if task_dir is None:
        raise FileNotFoundError(f"ARC public eval dataset not found under {data_root}")
    tasks = []
    for path in sorted(task_dir.glob("*.json")):
        tasks.append(json.loads(path.read_text(encoding="utf-8")))
    if not tasks:
        raise ValueError(f"no ARC tasks found in {task_dir}")
    return tasks


def _arc_constant_output(train_pairs: List[Dict[str, Any]]) -> Optional[List[List[int]]]:
    if not train_pairs:
        return None
    first = train_pairs[0].get("output")
    if first is None:
        return None
    for pair in train_pairs[1:]:
        if pair.get("output") != first:
            return None
    return first


def _arc_color_map(train_pairs: List[Dict[str, Any]]) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for pair in train_pairs:
        inp = pair.get("input")
        out = pair.get("output")
        if inp is None or out is None or len(inp) != len(out):
            return None
        if any(len(inp[r]) != len(out[r]) for r in range(len(inp))):
            return None
        for r in range(len(inp)):
            for c in range(len(inp[r])):
                src = int(inp[r][c])
                dst = int(out[r][c])
                if src in mapping and mapping[src] != dst:
                    return None
                mapping[src] = dst
    return mapping if mapping else None


def _arc_apply_color_map(grid: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
    return [[mapping.get(int(cell), int(cell)) for cell in row] for row in grid]


def solve_arc_task(task: Dict[str, Any]) -> Tuple[List[List[int]], int]:
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    test_input = test_pairs[0].get("input") if test_pairs else None
    attempts = 0

    constant_output = _arc_constant_output(train_pairs)
    if constant_output is not None:
        attempts += 1
        return constant_output, attempts

    color_map = _arc_color_map(train_pairs)
    if color_map is not None and test_input is not None:
        attempts += 1
        return _arc_apply_color_map(test_input, color_map), attempts

    attempts += 1
    return test_input if test_input is not None else [], attempts


def run_arc_benchmark(suite: str, seed: int) -> Dict[str, Any]:
    data_root = Path(os.environ.get("ARC_GYM_PATH", ""))
    if not str(data_root):
        raise EnvironmentError("ARC_GYM_PATH is not set")
    tasks = _load_arc_tasks(data_root, suite)
    random.seed(seed)

    tasks_solved = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for task in tasks:
        start = now_ms()
        prediction, attempts = solve_arc_task(task)
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        test_pairs = task.get("test", [])
        expected = test_pairs[0].get("output") if test_pairs else None
        if expected is not None and prediction == expected:
            tasks_solved += 1

    tasks_total = len(tasks)
    accuracy = tasks_solved / max(1, tasks_total)
    return {
        "suite": suite,
        "seed": seed,
        "tasks_total": tasks_total,
        "tasks_solved": tasks_solved,
        "accuracy": accuracy,
        "avg_attempts_per_task": total_attempts / max(1, tasks_total),
        "avg_runtime_ms_per_task": sum(runtimes_ms) / max(1, tasks_total),
    }
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        ap = argparse.ArgumentParser()
        ap.add_argument("benchmark")
        ap.add_argument("--suite", required=True)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--trials", type=int, default=20)
        args = ap.parse_args()
        result = run_benchmark_suite(args.suite, args.seed, args.trials)
        print(json.dumps(result, ensure_ascii=False))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "arc-benchmark":
        ap = argparse.ArgumentParser()
        ap.add_argument("arc-benchmark")
        ap.add_argument("--suite", required=True)
        ap.add_argument("--seed", type=int, default=0)
        args = ap.parse_args()
        result = run_arc_benchmark(args.suite, args.seed)
        print(json.dumps(result, ensure_ascii=False))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    env = ResearchEnvironment(seed=args.seed)
    tools = ToolRegistry()

    orch_cfg = OrchestratorConfig(
        agents=args.agents,
        base_budget=20,
        selection_top_k=max(3, args.agents // 2),
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    print("=== NON-RSI AGI CORE v5 (Neuro-Symbolic): RUN START ===")
    for r in range(args.rounds):
        out = orch.run_round(r)
        top = sorted(out["results"], key=lambda x: x["reward"], reverse=True)[:3]
        print(
            f"[Round {r:02d}] tasks={','.join(out['tasks']):<35} "
            f"tq={out['env']['tq']:.3f} kq={out['env']['kq']:.3f} oq={out['env']['oq']:.3f} "
            f"risk={out['policy']['risk']:.2f} infra={out['policy']['infra_focus']:.2f} "
            f"top_rewards={[round(x['reward'],4) for x in top]}"
        )

    print("=== RUN END ===")
    print("Recent memory summary:")
    for it in orch.mem.dump_summary(k=15):
        print(it)


if __name__ == "__main__":
    main()
