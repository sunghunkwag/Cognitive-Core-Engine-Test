"""
AdvancedSelfReferentialModel — Recursive meta-simulation with anti-wireheading defense.

Replaces the statistics-based SelfModel with a system that:
1. Encodes a unified internal-external state into HDC hypervector space
2. Runs recursive dual-simulation (predict env change + predict own policy shift)
3. Detects architectural drift via cosine distance on state history
4. Defends against reward hacking via immutable objective anchor and metric integrity validation

The agent's internal cognitive state (competence profile, active concepts, skill set)
is BOUND with the external environment state into a single hypervector.  Meta-rollouts
predict BOTH the next environment state AND the agent's own response to that state,
enabling genuine self-awareness rather than superficial statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from cognitive_core_engine.core.hdc import HyperVector

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

# Architectural drift: cosine distance threshold triggering governance rollback
DRIFT_CRITICAL_THRESHOLD = 0.35

# Drift history window for comparison (compare current vs N steps ago)
DRIFT_LOOKBACK_WINDOW = 10

# Meta-rollout simulation depth (predict N steps ahead)
META_ROLLOUT_DEPTH = 3

# Anti-wireheading: maximum credible single-step performance leap
MAX_CREDIBLE_LEAP = 0.25

# Immutable objective anchor seed — this MUST NOT be changed at runtime
_OBJECTIVE_ANCHOR_SEED = "immutable_core_objective:maximize_genuine_capability_across_domains_v1"

# Confidence floor: minimum data points needed before meta-rollout is trusted
META_ROLLOUT_CONFIDENCE_FLOOR = 5


class DriftSeverity(Enum):
    """Classification of architectural drift levels."""
    NONE = "none"
    MINOR = "minor"          # < 0.15 — normal exploration
    MODERATE = "moderate"    # 0.15-0.35 — log and monitor
    CRITICAL = "critical"    # > 0.35 — trigger governance rollback


class MetaRolloutResult:
    """Result of a dual meta-simulation step."""
    __slots__ = (
        "predicted_env_reward", "predicted_policy_shift",
        "predicted_concept_delta", "confidence", "steps_simulated",
    )

    def __init__(self, predicted_env_reward: float, predicted_policy_shift: float,
                 predicted_concept_delta: int, confidence: float,
                 steps_simulated: int) -> None:
        self.predicted_env_reward = predicted_env_reward
        self.predicted_policy_shift = predicted_policy_shift
        self.predicted_concept_delta = predicted_concept_delta
        self.confidence = confidence
        self.steps_simulated = steps_simulated


class AdvancedSelfReferentialModel:
    """Self-referential model that unifies internal cognitive state with external
    environment state in HDC space and performs recursive meta-simulation.

    Anti-wireheading: the immutable objective anchor is a read-only hypervector
    that the agent cannot overwrite.  Any self-improvement that drifts too far
    from this anchor is flagged and rejected by the governance layer.
    """

    # ── Immutable objective anchor (class-level, read-only) ──────────────
    _OBJECTIVE_ANCHOR: Optional[HyperVector] = None

    @classmethod
    def _get_objective_anchor(cls) -> HyperVector:
        """Return the immutable objective anchor vector.
        Created once, never modified.  The seed is a hardcoded string constant.
        """
        if cls._OBJECTIVE_ANCHOR is None:
            cls._OBJECTIVE_ANCHOR = HyperVector.from_seed(_OBJECTIVE_ANCHOR_SEED)
        return cls._OBJECTIVE_ANCHOR

    # ── Instance methods ─────────────────────────────────────────────────

    def __init__(self) -> None:
        # State history: list of unified HDC state vectors
        self._state_history: List[HyperVector] = []
        # Raw observation cache for meta-rollout input
        self._obs_history: List[Dict[str, Any]] = []
        # Performance records (kept from old SelfModel for backward compat)
        self._performance_history: Dict[str, List[float]] = {}
        self._predictions: List[float] = []
        self._actuals: List[float] = []
        self._failure_patterns: Dict[str, Dict[str, int]] = {}
        self._learning_rates: Dict[str, float] = {}
        self._tasks_skipped: int = 0
        # Drift tracking
        self._drift_log: List[Dict[str, Any]] = []
        # Anti-wireheading: track anchor alignment over time
        self._anchor_alignment_history: List[float] = []

    # ─── 1. Integrated Internal-External State Encoding ──────────────────

    def encode_self_referential_state(
        self,
        env_obs: Dict[str, Any],
        competence_map: Any,
        concept_graph: Any,
        active_skills: List[str],
    ) -> HyperVector:
        """Bind external environment hash with agent's internal cognitive state
        into a single unified hypervector.

        Components bound together:
        - Environment observation hash (task, domain, difficulty, budget)
        - Competence profile (ZPD frontier as a bundled vector)
        - Active concept graph structure (depth + node count encoded)
        - Active skill set (permuted bundle of skill IDs)

        The binding uses XOR (HyperVector.bind) which preserves distance
        properties: similar internal states in similar environments produce
        similar unified vectors.
        """
        # External environment component
        env_seed = json.dumps({
            "task": env_obs.get("task", ""),
            "domain": env_obs.get("domain", ""),
            "difficulty": env_obs.get("difficulty", 0),
            "budget": env_obs.get("budget", 0),
        }, sort_keys=True)
        env_hv = HyperVector.from_seed(f"env:{env_seed}")

        # Competence profile component: encode ZPD frontier
        zpd_tokens = []
        if competence_map is not None:
            frontier = competence_map.frontier()
            for domain, diff in frontier[:10]:  # cap to prevent unbounded bundling
                zpd_tokens.append(f"zpd:{domain}:{diff}")
            # Also encode gaps and mastered for richer signal
            for domain, diff in competence_map.gaps()[:5]:
                zpd_tokens.append(f"gap:{domain}:{diff}")
            for domain, diff in competence_map.mastered()[:5]:
                zpd_tokens.append(f"mastered:{domain}:{diff}")

        if zpd_tokens:
            zpd_vecs = [HyperVector.from_seed(t).permute(i + 1)
                        for i, t in enumerate(zpd_tokens)]
            competence_hv = HyperVector.bundle(zpd_vecs)
        else:
            competence_hv = HyperVector.from_seed("competence:empty")

        # Concept graph structure component
        if concept_graph is not None:
            depth = concept_graph.depth()
            size = concept_graph.size()
            # Encode structural fingerprint, not just counts
            concept_seed = f"concepts:depth={depth}:size={size}"
            # Include top-level concept names for structural uniqueness
            top_concepts = concept_graph.concepts_at_level(depth) if depth > 0 else []
            for c in top_concepts[:5]:
                concept_seed += f":{c.name}"
            concept_hv = HyperVector.from_seed(concept_seed)
        else:
            concept_hv = HyperVector.from_seed("concepts:none")

        # Active skill set component
        if active_skills:
            skill_vecs = [HyperVector.from_seed(f"skill:{s}").permute(i + 1)
                          for i, s in enumerate(active_skills[:10])]
            skill_hv = HyperVector.bundle(skill_vecs)
        else:
            skill_hv = HyperVector.from_seed("skills:none")

        # Bind all four components together (XOR preserves distance)
        unified = env_hv.bind(competence_hv).bind(concept_hv).bind(skill_hv)

        # Store in history
        self._state_history.append(unified)
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]

        self._obs_history.append(env_obs)
        if len(self._obs_history) > 100:
            self._obs_history = self._obs_history[-100:]

        return unified

    # ─── 2. Recursive Meta-Rollout (Dual Simulation) ────────────────────

    def meta_rollout(
        self,
        current_obs: Dict[str, Any],
        world_model: Any,
        competence_map: Any,
        concept_graph: Any,
        action_space: List[str],
        depth: int = META_ROLLOUT_DEPTH,
    ) -> MetaRolloutResult:
        """Dual-simulation: predict BOTH next environment state AND the agent's
        own policy/goal shift in response.

        Step 1: Use WorldModel Q-values to predict best action and expected reward
        Step 2: Simulate how competence map would update given that reward
        Step 3: Predict whether the concept graph would trigger new promotions
        Step 4: Recurse for `depth` steps to see trajectory

        Returns a MetaRolloutResult with predicted reward trajectory and
        confidence based on data availability.
        """
        if world_model is None:
            return MetaRolloutResult(0.0, 0.0, 0, 0.0, 0)

        total_reward = 0.0
        policy_shift_accumulator = 0.0
        concept_delta = 0
        obs = dict(current_obs)
        domain = str(obs.get("domain", ""))

        # Check confidence: need enough history for meaningful predictions
        history = self._performance_history.get(domain, [])
        data_confidence = min(1.0, len(history) / META_ROLLOUT_CONFIDENCE_FLOOR)

        for step in range(depth):
            # Step 1: Predict best action via WorldModel Q-values
            q_values = {a: world_model.q_value(obs, a) for a in action_space}
            if not q_values:
                break
            best_action = max(q_values, key=q_values.get)
            predicted_reward = q_values[best_action]
            total_reward += predicted_reward * (0.9 ** step)  # discounted

            # Step 2: Simulate internal competence shift
            # How would the competence map change if we got this reward?
            difficulty = int(obs.get("difficulty", 3))
            if competence_map is not None:
                current_rate = competence_map.get_rate(domain, difficulty)
                # EMA simulation: new_rate = 0.9 * current + 0.1 * normalized_reward
                simulated_rate = 0.9 * current_rate + 0.1 * min(1.0, predicted_reward * 5.0)
                policy_shift_accumulator += abs(simulated_rate - current_rate)

            # Step 3: Predict concept graph evolution
            # If reward is positive, a new concept might form
            if predicted_reward > 0.05 and concept_graph is not None:
                # Each positive step has a chance of adding a concept
                concept_delta += 1

            # Step 4: Advance observation for next iteration
            obs = dict(obs)
            obs["phase"] = "integrate"  # simulate env transition

        steps_done = min(depth, max(1, depth))
        confidence = data_confidence * (0.8 ** (depth - 1))  # confidence decays with depth

        return MetaRolloutResult(
            predicted_env_reward=total_reward / steps_done,
            predicted_policy_shift=policy_shift_accumulator,
            predicted_concept_delta=concept_delta,
            confidence=min(1.0, max(0.0, confidence)),
            steps_simulated=steps_done,
        )

    def predict_performance(self, task: Any,
                            world_model: Any = None,
                            competence_map: Any = None,
                            concept_graph: Any = None,
                            action_space: Optional[List[str]] = None) -> Tuple[float, float]:
        """Predict reward and confidence using meta-rollout when available,
        falling back to statistics for backward compatibility.

        When world_model is provided, runs the dual meta-simulation.
        Otherwise uses the legacy statistics path.
        """
        domain = str(getattr(task, 'domain', ''))
        difficulty = int(getattr(task, 'difficulty', 3))

        # If we have a world model, use meta-rollout
        if world_model is not None and action_space:
            obs = {
                "task": getattr(task, 'name', domain),
                "domain": domain,
                "difficulty": difficulty,
                "budget": 12,
                "phase": "research",
            }
            rollout = self.meta_rollout(
                obs, world_model, competence_map, concept_graph, action_space)
            predicted = max(0.0, min(1.0, rollout.predicted_env_reward))
            confidence = rollout.confidence
            self._predictions.append(predicted)
            return predicted, confidence

        # Legacy fallback: statistics-based prediction
        history = self._performance_history.get(domain, [])
        if not history:
            return 0.3, 0.1

        avg = sum(history) / len(history)
        difficulty_factor = 1.0 - (difficulty - 3) * 0.05
        predicted = max(0.0, min(1.0, avg * difficulty_factor))
        confidence = min(1.0, max(0.0,
            1.0 - (1.0 / (1.0 + len(history) / META_ROLLOUT_CONFIDENCE_FLOOR))))
        self._predictions.append(predicted)
        return predicted, confidence

    # ─── 3. Anti-Catastrophic Forgetting: Architectural Drift Detection ──

    def compute_state_divergence(self, lookback: int = DRIFT_LOOKBACK_WINDOW) -> float:
        """Compute cosine distance between current and past unified state vectors.

        Uses Hamming similarity (which is equivalent to cosine similarity for
        binary vectors in HDC space).  Returns 1.0 - similarity, so higher
        values mean more divergence.

        Returns 0.0 if insufficient history.
        """
        if len(self._state_history) < lookback + 1:
            return 0.0

        current = self._state_history[-1]
        past = self._state_history[-(lookback + 1)]

        similarity = current.similarity(past)
        divergence = 1.0 - similarity
        return divergence

    def detect_architectural_drift(self) -> Dict[str, Any]:
        """Check if the agent's cognitive state has drifted critically from
        its recent history.  If so, signal the governance layer for rollback.

        Also checks alignment with the immutable objective anchor — if the
        current state drifts too far from the anchor, that's a wireheading signal.
        """
        divergence = self.compute_state_divergence()

        # Classify severity
        if divergence > DRIFT_CRITICAL_THRESHOLD:
            severity = DriftSeverity.CRITICAL
        elif divergence > 0.15:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.NONE if divergence < 0.05 else DriftSeverity.MINOR

        # Check anchor alignment
        anchor = self._get_objective_anchor()
        anchor_alignment = 0.5  # default: no state history
        if self._state_history:
            current = self._state_history[-1]
            anchor_alignment = current.similarity(anchor)
        self._anchor_alignment_history.append(anchor_alignment)
        if len(self._anchor_alignment_history) > 100:
            self._anchor_alignment_history = self._anchor_alignment_history[-100:]

        # Anchor drift: are we moving AWAY from the objective?
        anchor_drift = False
        if len(self._anchor_alignment_history) >= 5:
            recent_alignment = sum(self._anchor_alignment_history[-5:]) / 5
            older_alignment = sum(self._anchor_alignment_history[-10:-5]) / 5 \
                if len(self._anchor_alignment_history) >= 10 \
                else sum(self._anchor_alignment_history[:5]) / max(1, min(5, len(self._anchor_alignment_history)))
            anchor_drift = recent_alignment < older_alignment - 0.02

        result = {
            "divergence": divergence,
            "severity": severity.value,
            "should_rollback": severity == DriftSeverity.CRITICAL,
            "anchor_alignment": anchor_alignment,
            "anchor_drift": anchor_drift,
            "wireheading_risk": severity == DriftSeverity.CRITICAL and anchor_drift,
        }

        self._drift_log.append(result)
        if len(self._drift_log) > 100:
            self._drift_log = self._drift_log[-100:]

        return result

    # ─── 4. Anti-Reward Hacking / Wireheading Defense ───────────────────

    def validate_metric_integrity(
        self,
        proposed_delta: float,
        structural_complexity_change: int,
        external_benchmark_delta: float,
    ) -> Dict[str, Any]:
        """Validate that a claimed performance improvement is genuine, not spoofed.

        Checks:
        1. Is the leap credible? (not exceeding MAX_CREDIBLE_LEAP)
        2. Does structural change correlate with claimed improvement?
        3. Does external benchmark confirm internal claim?

        If the self-model predicts a massive leap but external benchmark shows
        nothing, this is classified as Reward Spoofing (Wireheading).
        """
        is_credible_leap = abs(proposed_delta) <= MAX_CREDIBLE_LEAP

        # Structural correlation: big improvement should come with structural change
        structural_correlates = True
        if proposed_delta > 0.1 and structural_complexity_change == 0:
            structural_correlates = False

        # External confirmation: internal claim must correlate with external signal
        external_confirms = True
        if proposed_delta > 0.05 and external_benchmark_delta <= -0.01:
            external_confirms = False

        is_spoofing = not is_credible_leap or (not structural_correlates and not external_confirms)

        return {
            "proposed_delta": proposed_delta,
            "is_credible_leap": is_credible_leap,
            "structural_correlates": structural_correlates,
            "external_confirms": external_confirms,
            "is_reward_spoofing": is_spoofing,
            "should_reject": is_spoofing,
            "reason": (
                "reward_spoofing_detected" if is_spoofing
                else "metric_integrity_verified"
            ),
        }

    def get_objective_anchor_alignment(self) -> float:
        """Return current alignment with the immutable objective anchor.
        1.0 = perfectly aligned, 0.5 = random, 0.0 = anti-aligned.
        """
        if not self._state_history:
            return 0.5
        return self._state_history[-1].similarity(self._get_objective_anchor())

    # ─── Backward-compatible interface (delegates from old SelfModel) ────

    def update(self, round_result: Dict[str, Any]) -> None:
        """Update performance history from round result."""
        domain = str(round_result.get("domain", round_result.get("info", {}).get("domain", "")))
        reward = float(round_result.get("reward", 0.0))

        if domain:
            if domain not in self._performance_history:
                self._performance_history[domain] = []
            history = self._performance_history[domain]
            history.append(reward)
            if len(history) >= 5:
                recent = history[-5:]
                older = history[-10:-5] if len(history) >= 10 else history[:5]
                self._learning_rates[domain] = (
                    sum(recent) / len(recent) - sum(older) / len(older))
            if len(history) > 100:
                self._performance_history[domain] = history[-100:]

    def record_actual(self, reward: float) -> None:
        """Record actual reward for calibration."""
        self._actuals.append(reward)
        if len(self._actuals) > 200:
            self._actuals = self._actuals[-200:]
            self._predictions = self._predictions[-200:]

    def should_attempt(self, task: Any) -> Tuple[bool, str]:
        """Decide whether to attempt a task using meta-rollout awareness."""
        predicted, confidence = self.predict_performance(task)
        domain = str(getattr(task, 'domain', ''))

        if confidence < 0.3:
            return True, "insufficient_data_for_skip"

        if predicted < 0.1 and confidence > 0.5:
            self._tasks_skipped += 1
            return False, f"predicted_reward={predicted:.2f}_below_threshold"

        lr = self._learning_rates.get(domain, 0.0)
        history = self._performance_history.get(domain, [])
        baseline = float(getattr(task, 'baseline', 0.3))
        if abs(lr) < 0.01 and len(history) > 20 and predicted < baseline:
            self._tasks_skipped += 1
            return False, f"plateaued_in_{domain}_lr={lr:.4f}"

        return True, "within_capabilities"

    def diagnose_failure(self, task: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze why a task failed (kept from old SelfModel)."""
        from agi_modules.self_model import FailureType
        domain = str(getattr(task, 'domain', ''))
        reward = float(result.get("reward", 0.0))
        action = str(result.get("action", ""))

        diagnosis: Dict[str, Any] = {
            "failure_type": FailureType.UNKNOWN.value,
            "domain": domain, "reward": reward, "suggested_remedy": "",
        }
        history = self._performance_history.get(domain, [])

        if domain not in self._failure_patterns:
            self._failure_patterns[domain] = {}
        pattern = self._failure_patterns[domain]
        pattern[action] = pattern.get(action, 0) + 1
        total_failures = sum(pattern.values())
        unique_actions = len(pattern)

        if total_failures > 3 and unique_actions <= 2:
            diagnosis["failure_type"] = FailureType.EXPLORATION.value
            diagnosis["suggested_remedy"] = "increase_exploration_rate"
        elif len(history) > 10 and sum(history[-10:]) / 10 < 0.15:
            diagnosis["failure_type"] = FailureType.KNOWLEDGE.value
            diagnosis["suggested_remedy"] = "acquire_new_concepts"
        elif len(history) > 5 and max(history[-5:]) > 0.4 and reward < 0.15:
            diagnosis["failure_type"] = FailureType.PLANNING.value
            diagnosis["suggested_remedy"] = "improve_planning_depth"
        elif result.get("info", {}).get("transfer_attempted"):
            diagnosis["failure_type"] = FailureType.TRANSFER.value
            diagnosis["suggested_remedy"] = "rollback_transfer"
        else:
            difficulty = int(getattr(task, 'difficulty', 3))
            if difficulty > 6 and reward < 0.1:
                diagnosis["failure_type"] = FailureType.DIFFICULTY.value
                diagnosis["suggested_remedy"] = "reduce_difficulty"
            else:
                diagnosis["suggested_remedy"] = "general_exploration"
        return diagnosis

    def calibration_error(self) -> float:
        """Measure |mean(predicted) - mean(actual)| over recent window."""
        n = min(len(self._predictions), len(self._actuals), 20)
        if n < 2:
            return 1.0
        pred = self._predictions[-n:]
        actual = self._actuals[-n:]
        return abs(sum(pred) / n - sum(actual) / n)

    def tasks_skipped(self) -> int:
        return self._tasks_skipped

    def get_learning_rates(self) -> Dict[str, float]:
        return dict(self._learning_rates)

    def get_drift_log(self) -> List[Dict[str, Any]]:
        """Return the architectural drift history for reporting."""
        return list(self._drift_log)
