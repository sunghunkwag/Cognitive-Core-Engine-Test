"""
ExternalBenchmarkConnector — BN-03 fix.

The anti-wireheading gate in AdvancedSelfReferentialModel checks that any
claimed self-improvement is correlated with a GENUINELY EXTERNAL benchmark.
Without this connector, self_improvement_score was hard-coded to 0.0 because
no external signal existed, blocking all governance proposals from advancing.

Design decisions:
- Offline-first: reads pre-computed JSON result files; never calls APIs at
  runtime.  This keeps governance deterministic and reproducible.
- Supports ARC-AGI and HumanEval result formats out of the box.  Additional
  benchmarks can be registered via `register_results()`.
- `correlation_score()` returns a float in [0, 1] that the anti-wireheading
  gate can use directly as `external_benchmark_correlation`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Supported benchmark formats
# ---------------------------------------------------------------------------

_KNOWN_FORMATS = frozenset({"arc_agi", "humaneval", "generic"})


def _parse_arc_agi(data: Dict[str, Any]) -> float:
    """Extract a scalar score from an ARC-AGI result dict.

    Expects either:
      {"score": 0.42}                   — direct scalar
      {"results": [{"pass": true/false}, ...]}  — list of task results
    """
    if "score" in data:
        return float(data["score"])
    results = data.get("results", [])
    if results:
        passed = sum(1 for r in results if r.get("pass", False))
        return passed / len(results)
    return 0.0


def _parse_humaneval(data: Dict[str, Any]) -> float:
    """Extract pass@1 from a HumanEval result dict.

    Expects either:
      {"pass@1": 0.35}
      {"results": {"pass@1": 0.35}}
    """
    if "pass@1" in data:
        return float(data["pass@1"])
    nested = data.get("results", {})
    if isinstance(nested, dict) and "pass@1" in nested:
        return float(nested["pass@1"])
    # Fallback: look for any float value
    for v in data.values():
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _parse_generic(data: Dict[str, Any]) -> float:
    """Best-effort scalar extraction from an unknown result dict."""
    for key in ("score", "accuracy", "pass_rate", "result"):
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                continue
    # Try any top-level numeric value
    for v in data.values():
        try:
            f = float(v)
            if 0.0 <= f <= 1.0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


_PARSERS = {
    "arc_agi": _parse_arc_agi,
    "humaneval": _parse_humaneval,
    "generic": _parse_generic,
}


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class ExternalBenchmarkConnector:
    """Loads external benchmark results and computes correlation with internal metrics.

    Parameters
    ----------
    results_path:
        Path to a JSON file containing benchmark results.  Can also be
        supplied later via `register_results()`.
    benchmark_format:
        One of 'arc_agi', 'humaneval', or 'generic'.
    """

    def __init__(
        self,
        results_path: Optional[Union[str, Path]] = None,
        benchmark_format: str = "generic",
    ) -> None:
        if benchmark_format not in _KNOWN_FORMATS:
            raise ValueError(
                f"Unknown benchmark format '{benchmark_format}'. "
                f"Choose from {sorted(_KNOWN_FORMATS)}."
            )
        self._format = benchmark_format
        self._raw: Dict[str, Any] = {}
        self._external_score: Optional[float] = None
        self._score_history: List[float] = []

        if results_path is not None:
            self.load(results_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path]) -> None:
        """Load benchmark results from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark results file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            self._raw = json.load(f)
        self._external_score = _PARSERS[self._format](self._raw)
        self._score_history.append(self._external_score)

    def register_results(self, data: Dict[str, Any]) -> None:
        """Register results from an in-memory dict (for testing or CI)."""
        self._raw = data
        self._external_score = _PARSERS[self._format](data)
        self._score_history.append(self._external_score)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @property
    def external_score(self) -> float:
        """Most recent external benchmark score, or 0.0 if not yet loaded."""
        return self._external_score if self._external_score is not None else 0.0

    def correlation_score(
        self,
        internal_metrics: Dict[str, float],
    ) -> float:
        """Return a correlation score in [0, 1] between external and internal metrics.

        The score reflects how well the agent's self-reported performance
        (internal_metrics) aligns with external ground truth.  A high score
        means the agent is not wireheading.

        Algorithm:
          weighted_internal = weighted average of internal metrics
          alignment = 1 - |weighted_internal - external_score|
          Returns alignment clamped to [0, 1].
        """
        if self._external_score is None:
            return 0.0

        if not internal_metrics:
            # No internal metrics to compare against; neutral score
            return 0.5

        # Weighted average: holdout_pass_rate gets double weight if present
        weights = {
            "holdout_pass_rate": 2.0,
            "train_pass_rate": 1.0,
            "adversarial_pass_rate": 1.5,
        }
        total_w = 0.0
        weighted_sum = 0.0
        for key, val in internal_metrics.items():
            w = weights.get(key, 1.0)
            weighted_sum += w * float(val)
            total_w += w
        weighted_internal = weighted_sum / total_w if total_w > 0 else 0.5

        alignment = 1.0 - abs(weighted_internal - self._external_score)
        return max(0.0, min(1.0, alignment))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._external_score is not None

    def summary(self) -> Dict[str, Any]:
        return {
            "format": self._format,
            "loaded": self.is_loaded(),
            "external_score": self.external_score,
            "score_history": list(self._score_history),
        }
