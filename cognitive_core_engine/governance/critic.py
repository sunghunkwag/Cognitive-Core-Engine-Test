"""
Governance Critic — evaluates candidate proposals before admission.

Security note (BN-05):
  The original implementation fell back to a SHA-256-derived pseudo-score
  when holdout_rate was absent.  This allowed proposals with NO empirical
  metrics to pass governance purely by chance of their serialised hash.
  Fix: score defaults to 0.0 when holdout_rate is None.  hash_score is
  kept in score_components for logging/debugging only and NEVER influences
  the verdict.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from cognitive_core_engine.governance.utils import now_ms, sha256, read_json, write_json, safe_mkdir


def critic_evaluate_candidate_packet(
    packet: Dict[str, Any],
    invariants: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a governance proposal packet and return a verdict.

    BN-05 security fix: when holdout_rate is absent, score is set to 0.0
    rather than using a SHA-256 hash as a pseudo-score.  The hash_score is
    retained in score_components for diagnostics but MUST NOT influence the
    verdict.  The 'used_hash_fallback' field is set to True whenever the
    original code would have used the hash fallback, so callers can detect
    and log this situation.
    """
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    invariants = dict(invariants or {})
    proposal = packet.get("proposal", {}) if isinstance(packet, dict) else {}
    evaluation_rules = packet.get("evaluation_rules", {}) if isinstance(packet, dict) else {}

    level = str(proposal.get("level", "L0"))
    payload = proposal.get("payload", {}) if isinstance(proposal, dict) else {}
    candidate = payload.get("candidate", {})
    meta_update = payload.get("meta_update", {})
    evidence = proposal.get("evidence", {}) if isinstance(proposal, dict) else {}

    # Compute hash_score for diagnostics ONLY — never used in scoring.
    serialized = json.dumps(candidate, sort_keys=True, default=str)
    hash_score = (int(sha256(serialized)[:8], 16) % 100) / 100.0

    min_score = float(evaluation_rules.get("min_score", 0.4))

    metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    train_rate = _coerce_float(metrics.get("train_pass_rate"))
    holdout_rate = _coerce_float(metrics.get("holdout_pass_rate"))
    discovery_cost = metrics.get("discovery_cost", {})
    holdout_cost = None
    if isinstance(discovery_cost, dict):
        holdout_cost = _coerce_float(discovery_cost.get("holdout"))
    adversarial_rate = _coerce_float(metrics.get("adversarial_pass_rate"))
    distribution_shift = metrics.get("distribution_shift", {})
    shift_holdout_rate = None
    if isinstance(distribution_shift, dict):
        shift_holdout_rate = _coerce_float(distribution_shift.get("holdout_pass_rate"))

    holdout_weight = float(evaluation_rules.get("holdout_weight", 1.0))
    gap_penalty = float(evaluation_rules.get("generalization_gap_penalty", 0.75))
    cost_penalty = float(evaluation_rules.get("discovery_cost_penalty", 0.08))
    gap = None

    # BN-05 FIX: when holdout_rate is absent, score = 0.0 (not hash_score).
    # Proposals without real empirical metrics must not pass governance.
    used_hash_fallback = holdout_rate is None
    score = 0.0  # safe default — zero until proven by real metrics

    score_components = {
        "holdout_term": None,
        "gap_penalty": 0.0,
        "cost_penalty": 0.0,
        # hash_score is logged here for diagnostics but NEVER added to score.
        "hash_score": hash_score,
    }

    if holdout_rate is not None:
        if train_rate is not None:
            gap = abs(train_rate - holdout_rate)
        score = holdout_weight * holdout_rate
        score_components["holdout_term"] = score
        if gap is not None:
            penalty = gap_penalty * gap
            score -= penalty
            score_components["gap_penalty"] = penalty
        if holdout_cost is not None:
            penalty = cost_penalty * holdout_cost
            score -= penalty
            score_components["cost_penalty"] = penalty

    min_holdout = float(evaluation_rules.get("min_holdout_pass_rate", 0.3))
    max_gap = float(evaluation_rules.get("max_generalization_gap", 0.05))
    min_adversarial = float(evaluation_rules.get("min_adversarial_pass_rate", min_holdout))
    min_shift_holdout = float(evaluation_rules.get("min_shift_holdout_pass_rate", min_holdout))
    max_holdout_cost = float(evaluation_rules.get("max_holdout_discovery_cost", 4.0))
    require_holdout_metrics = bool(evaluation_rules.get("require_holdout_metrics", False))

    evidence_count = 0
    if isinstance(evidence, dict):
        for val in evidence.values():
            if isinstance(val, dict):
                evidence_count += len(val)
            elif isinstance(val, list):
                evidence_count += len(val)
            else:
                evidence_count += 1
    min_evidence = int(invariants.get("min_evidence", 1))
    evidence_ok = evidence_count >= min_evidence or bool(candidate)

    holdout_ok = True
    if require_holdout_metrics and holdout_rate is None:
        holdout_ok = False
    if holdout_rate is not None and holdout_rate < min_holdout:
        holdout_ok = False

    gap_ok = True
    if gap is not None and gap > max_gap:
        gap_ok = False

    adversarial_ok = True
    if adversarial_rate is not None and adversarial_rate < min_adversarial:
        adversarial_ok = False

    shift_ok = True
    if shift_holdout_rate is not None and shift_holdout_rate < min_shift_holdout:
        shift_ok = False

    holdout_cost_ok = True
    if require_holdout_metrics:
        holdout_cost_ok = holdout_cost is not None and holdout_cost <= max_holdout_cost

    regression_ok = True
    baseline = metrics.get("baseline")
    if isinstance(baseline, dict):
        baseline_train = _coerce_float(baseline.get("train_pass_rate"))
        baseline_holdout = _coerce_float(baseline.get("holdout_pass_rate"))
        if (
            train_rate is not None
            and holdout_rate is not None
            and baseline_train is not None
            and baseline_holdout is not None
            and train_rate > baseline_train
            and holdout_rate < baseline_holdout
        ):
            regression_ok = False

    meta_ok = True
    if level == "L2":
        proposed_rate = meta_update.get("l1_update_rate")
        bounds = invariants.get("l1_update_rate_bounds", (0.04, 0.20))
        if proposed_rate is None:
            meta_ok = False
        else:
            meta_ok = float(bounds[0]) <= float(proposed_rate) <= float(bounds[1])

    guardrails_ok = (
        holdout_ok
        and gap_ok
        and adversarial_ok
        and shift_ok
        and regression_ok
        and holdout_cost_ok
    )
    verdict = "approve" if score >= min_score and evidence_ok and meta_ok and guardrails_ok else "reject"
    approval_key = sha256(f"{proposal.get('proposal_id', '')}:{level}:{score}")[:12]
    return {
        "verdict": verdict,
        "score": score,
        "hash_score": hash_score,          # diagnostic only
        "used_hash_fallback": used_hash_fallback,  # BN-05: caller alert flag
        "score_components": score_components,
        "approval_key": approval_key,
        "level": level,
        "min_score": min_score,
        "evidence_ok": evidence_ok,
        "meta_ok": meta_ok,
        "holdout_rate": holdout_rate,
        "train_rate": train_rate,
        "gap": gap,
        "holdout_ok": holdout_ok,
        "gap_ok": gap_ok,
        "adversarial_ok": adversarial_ok,
        "shift_ok": shift_ok,
        "regression_ok": regression_ok,
        "holdout_cost_ok": holdout_cost_ok,
        "guardrails_ok": guardrails_ok,
    }


class RunLogger:
    def __init__(self, path: Path, window: int = 10, append: bool = False):
        self.path = path
        self.window = window
        self.records: List[Dict[str, Any]] = []
        self.best_scores: List[float] = []
        self.best_hold: List[float] = []
        self.seen_hashes: Set[str] = set()
        safe_mkdir(self.path.parent)
        if self.path.exists() and not append:
            self.path.unlink()

    def _window_slice(self, vals: List[float]) -> List[float]:
        if not vals:
            return []
        return vals[-self.window :]

    def log(
        self,
        gen: int,
        task_id: str,
        mode: str,
        score_hold: float,
        score_stress: float,
        score_test: float,
        runtime_ms: int,
        nodes: int,
        code_hash: str,
        accepted: bool,
        novelty: float,
        meta_policy_params: Dict[str, Any],
        solver_hash: Optional[str] = None,
        p1_hash: Optional[str] = None,
        err_hold: Optional[float] = None,
        err_stress: Optional[float] = None,
        err_test: Optional[float] = None,
        steps: Optional[int] = None,
        timeout_rate: Optional[float] = None,
        counterexample_count: Optional[int] = None,
        library_size: Optional[int] = None,
        control_packet: Optional[Dict[str, Any]] = None,
        task_descriptor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.best_scores.append(score_hold)
        self.best_hold.append(score_hold)
        window_vals = self._window_slice(self.best_hold)
        auc_window = sum(window_vals) / max(1, len(window_vals))
        if len(self.best_hold) > self.window:
            delta_best_window = self.best_hold[-1] - self.best_hold[-self.window]
        else:
            delta_best_window = self.best_hold[-1] - self.best_hold[0]
        record = {
            "gen": gen,
            "task_id": task_id,
            "solver_hash": solver_hash or code_hash,
            "p1_hash": p1_hash or "default",
            "mode": mode,
            "score_hold": score_hold,
            "score_stress": score_stress,
            "score_test": score_test,
            "err_hold": err_hold if err_hold is not None else score_hold,
            "err_stress": err_stress if err_stress is not None else score_stress,
            "err_test": err_test if err_test is not None else score_test,
            "auc_window": auc_window,
            "delta_best_window": delta_best_window,
            "runtime_ms": runtime_ms,
            "nodes": nodes,
            "hash": code_hash,
            "accepted": accepted,
            "novelty": novelty,
            "meta_policy_params": meta_policy_params,
            "steps": steps,
            "timeout_rate": timeout_rate,
            "counterexample_count": counterexample_count,
            "library_size": library_size,
            "control_packet": control_packet or {},
            "task_descriptor": task_descriptor,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)
        return record


# ---------------------------
# Blackboard utilities
# ---------------------------

def append_blackboard(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def tail_blackboard(path: Path, k: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines: collections.deque[str] = collections.deque(maxlen=k)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records
