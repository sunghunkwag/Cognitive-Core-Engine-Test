"""World Model (feature-based value model) extracted from NON_RSI_AGI_CORE_v5."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from cognitive_core_engine.core.utils import stable_hash


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
