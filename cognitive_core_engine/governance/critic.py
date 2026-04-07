"""
Governance Critic — evaluates candidate proposals before admission.

Security hardening (BN-05 — FINAL):
  REQUIRE_HOLDOUT_METRICS is a module-level constant set to True.
  It cannot be overridden by evaluation_rules or any L1/L2 update.

  Proposals lacking holdout_metrics are ALWAYS rejected with
  verdict='reject' and reason='missing_holdout_metrics'.

  The hash_score field is removed from all return dicts.  It existed
  solely to serve the old fallback path; that path is gone.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from cognitive_core_engine.governance.utils import now_ms, sha256, read_json, write_json, safe_mkdir

# BN-05: hard constant — CANNOT be changed at runtime via evaluation_rules.
# Any code that tries to set evaluation_rules['require_holdout_metrics'] = False
# will be silently ignored because this module never reads that key for this check.
REQUIRE_HOLDOUT_METRICS: bool = True


def critic_evaluate_candidate_packet(
    packet: Dict[str, Any],
    invariants: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a governance proposal packet and return a verdict.

    BN-05 (final): require_holdout_metrics is enforced via the module-level
    REQUIRE_HOLDOUT_METRICS constant.  The evaluation_rules dict is NOT
    consulted for this check.

    Returns a dict with keys:
      verdict          : 'approve' | 'reject'
      score            : float (0.0 if holdout absent)
      reason           : str (empty string when approved)
      used_hash_fallback: bool (always False; kept for audit log compatibility)
      ... (other diagnostic fields)
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

    # BN-05: reject immediately if holdout_metrics absent.
    # REQUIRE_HOLDOUT_METRICS is a hard module-level constant (always True).
    if holdout_rate is None:
        return {
            "verdict": "reject",
            "score": 0.0,
            "reason": "missing_holdout_metrics",
            "used_hash_fallback": False,  # audit log compatibility
            "level": level,
            "min_score": min_score,
            "holdout_rate": None,
            "train_rate": train_rate,
            "gap": None,
            "holdout_ok": False,
            "gap_ok": True,
            "adversarial_ok": True,
            "shift_ok": True,
            "regression_ok": True,
            "holdout_cost_ok": False,
            "guardrails_ok": False,
            "evidence_ok": False,
            "meta_ok": True,
            "score_components": {
                "holdout_term": None,
                "gap_penalty": 0.0,
                "cost_penalty": 0.0,
            },
        }

    # --- Scoring (holdout_rate is confirmed present) ---
    holdout_weight = float(evaluation_rules.get("holdout_weight", 1.0))
    gap_penalty = float(evaluation_rules.get("generalization_gap_penalty", 0.75))
    cost_penalty = float(evaluation_rules.get("discovery_cost_penalty", 0.08))

    gap: Optional[float] = None
    if train_rate is not None:
        gap = abs(train_rate - holdout_rate)

    score = holdout_weight * holdout_rate
    score_components: Dict[str, Any] = {
        "holdout_term": score,
        "gap_penalty": 0.0,
        "cost_penalty": 0.0,
    }
    if gap is not None:
        penalty = gap_penalty * gap
        score -= penalty
        score_components["gap_penalty"] = penalty
    if holdout_cost is not None:
        penalty = cost_penalty * holdout_cost
        score -= penalty
        score_components["cost_penalty"] = penalty

    # --- Guardrail checks ---
    min_holdout = float(evaluation_rules.get("min_holdout_pass_rate", 0.3))
    max_gap = float(evaluation_rules.get("max_generalization_gap", 0.05))
    min_adversarial = float(evaluation_rules.get("min_adversarial_pass_rate", min_holdout))
    min_shift_holdout = float(evaluation_rules.get("min_shift_holdout_pass_rate", min_holdout))
    max_holdout_cost = float(evaluation_rules.get("max_holdout_discovery_cost", 4.0))

    holdout_ok = holdout_rate >= min_holdout
    gap_ok = gap is None or gap <= max_gap
    adversarial_ok = adversarial_rate is None or adversarial_rate >= min_adversarial
    shift_ok = shift_holdout_rate is None or shift_holdout_rate >= min_shift_holdout
    holdout_cost_ok = holdout_cost is None or holdout_cost <= max_holdout_cost

    regression_ok = True
    baseline = metrics.get("baseline")
    if isinstance(baseline, dict):
        baseline_train = _coerce_float(baseline.get("train_pass_rate"))
        baseline_holdout = _coerce_float(baseline.get("holdout_pass_rate"))
        if (
            train_rate is not None
            and baseline_train is not None
            and baseline_holdout is not None
            and train_rate > baseline_train
            and holdout_rate < baseline_holdout
        ):
            regression_ok = False

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

    meta_ok = True
    if level == "L2":
        proposed_rate = meta_update.get("l1_update_rate")
        bounds = invariants.get("l1_update_rate_bounds", (0.04, 0.20))
        if proposed_rate is None:
            meta_ok = False
        else:
            meta_ok = float(bounds[0]) <= float(proposed_rate) <= float(bounds[1])

    guardrails_ok = holdout_ok and gap_ok and adversarial_ok and shift_ok and regression_ok and holdout_cost_ok
    verdict = "approve" if score >= min_score and evidence_ok and meta_ok and guardrails_ok else "reject"
    approval_key = sha256(f"{proposal.get('proposal_id', '')}:{level}:{score}")[:12]

    return {
        "verdict": verdict,
        "score": score,
        "reason": "" if verdict == "approve" else "guardrails_or_score_failed",
        "used_hash_fallback": False,  # audit log compatibility
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
        return vals[-self.window:]

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
            "gen": gen, "task_id": task_id,
            "solver_hash": solver_hash or code_hash,
            "p1_hash": p1_hash or "default",
            "mode": mode, "score_hold": score_hold,
            "score_stress": score_stress, "score_test": score_test,
            "err_hold": err_hold if err_hold is not None else score_hold,
            "err_stress": err_stress if err_stress is not None else score_stress,
            "err_test": err_test if err_test is not None else score_test,
            "auc_window": auc_window, "delta_best_window": delta_best_window,
            "runtime_ms": runtime_ms, "nodes": nodes, "hash": code_hash,
            "accepted": accepted, "novelty": novelty,
            "meta_policy_params": meta_policy_params,
            "steps": steps, "timeout_rate": timeout_rate,
            "counterexample_count": counterexample_count,
            "library_size": library_size,
            "control_packet": control_packet or {},
            "task_descriptor": task_descriptor,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)
        return record


def append_blackboard(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def tail_blackboard(path: Path, k: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines: collections.deque = collections.deque(maxlen=k)
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
