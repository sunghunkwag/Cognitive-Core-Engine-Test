"""
AGIProgressTracker — Measures progress across 5 AGI capability axes.

Serves AGI capability: provides quantitative evidence of progress toward
general intelligence across generalization, autonomy, self-improvement,
abstraction, and open-endedness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# --- Named constants (Rule 6) ---

# Target abstraction depth for scoring
TARGET_ABSTRACTION_DEPTH = 5

# Plateau detection: rounds with < this improvement
PLATEAU_IMPROVEMENT_THRESHOLD = 0.01
PLATEAU_WINDOW = 20


class AGIProgressTracker:
    """Tracks 5 AGI capability axes, each scored 0.0 to 1.0.

    Why it exists: provides measurable evidence of AGI-relevant progress
    beyond simple task performance metrics.

    Fallback: returns 0.0 for axes with no data.
    """

    def __init__(self) -> None:
        self._transfer_successes: List[float] = []
        self._transfer_attempts: int = 0
        self._self_generated_goals: int = 0
        self._total_goals: int = 0
        self._beneficial_self_mods: int = 0
        self._total_self_mods: int = 0
        self._concept_depth: int = 0
        self._new_domains: int = 0
        self._difficulty_increases: int = 0
        self._rounds_elapsed: int = 0
        self._composite_history: List[float] = []

    def update_transfer(self, success: float, attempted: bool) -> None:
        """Record a transfer learning attempt and outcome.

        Why: feeds the GENERALIZATION axis.
        Fallback: no-op if not attempted.
        """
        if attempted:
            self._transfer_attempts += 1
            self._transfer_successes.append(success)

    def update_goals(self, self_generated: int, total: int) -> None:
        """Record goal generation statistics.

        Why: feeds the AUTONOMY axis.
        Fallback: no-op.
        """
        self._self_generated_goals += self_generated
        self._total_goals += total

    def update_self_improvement(self, beneficial: bool, attempted: bool) -> None:
        """Record self-improvement attempt and outcome.

        Why: feeds the SELF-IMPROVEMENT axis.
        Fallback: no-op if not attempted.
        """
        if attempted:
            self._total_self_mods += 1
            if beneficial:
                self._beneficial_self_mods += 1

    def update_abstraction(self, depth: int) -> None:
        """Record concept graph depth.

        Why: feeds the ABSTRACTION axis.
        Fallback: no-op.
        """
        self._concept_depth = max(self._concept_depth, depth)

    def update_open_endedness(self, new_domains: int,
                              difficulty_increases: int) -> None:
        """Record environment expansion metrics.

        Why: feeds the OPEN-ENDEDNESS axis.
        Fallback: no-op.
        """
        self._new_domains += new_domains
        self._difficulty_increases += difficulty_increases

    def tick_round(self) -> None:
        """Record that a round has elapsed.

        Why: needed for rate-based metrics.
        Fallback: no-op.
        """
        self._rounds_elapsed += 1
        self._composite_history.append(self.composite_score())

    def score(self) -> Dict[str, float]:
        """Return all 5 AGI axis scores.

        Why: provides the full capability profile.
        Fallback: returns 0.0 for axes with no data.
        """
        return {
            "generalization": self._score_generalization(),
            "autonomy": self._score_autonomy(),
            "self_improvement": self._score_self_improvement(),
            "abstraction": self._score_abstraction(),
            "open_endedness": self._score_open_endedness(),
        }

    def composite_score(self) -> float:
        """Geometric mean of all 5 axes (all must improve, not just one).

        Why: prevents gaming a single axis while neglecting others.
        Fallback: returns 0.0 if any axis is 0.
        """
        scores = self.score()
        values = list(scores.values())

        # Add small epsilon to avoid zero product
        epsilon = 0.001
        adjusted = [max(v, epsilon) for v in values]

        # Geometric mean
        product = 1.0
        for v in adjusted:
            product *= v
        return product ** (1.0 / len(adjusted))

    def progress_report(self, round_idx: int) -> str:
        """Human-readable progress report.

        Why: enables monitoring of AGI capability development.
        Fallback: returns minimal report if no data.
        """
        scores = self.score()
        composite = self.composite_score()
        lines = [
            f"=== AGI Progress Report (Round {round_idx}) ===",
            f"  Generalization:    {scores['generalization']:.3f}",
            f"  Autonomy:          {scores['autonomy']:.3f}",
            f"  Self-Improvement:  {scores['self_improvement']:.3f}",
            f"  Abstraction:       {scores['abstraction']:.3f}",
            f"  Open-Endedness:    {scores['open_endedness']:.3f}",
            f"  Composite (geom):  {composite:.3f}",
            f"  Plateaued:         {self.is_plateaued()}",
        ]
        return "\n".join(lines)

    def is_plateaued(self) -> bool:
        """Detect if composite score hasn't improved in PLATEAU_WINDOW rounds.

        Why: triggers adaptive responses to break out of stagnation.
        Fallback: returns False if insufficient history.
        """
        if len(self._composite_history) < PLATEAU_WINDOW:
            return False

        recent = self._composite_history[-PLATEAU_WINDOW:]
        improvement = max(recent) - min(recent)
        return improvement < PLATEAU_IMPROVEMENT_THRESHOLD

    # --- Private scoring methods ---

    def _score_generalization(self) -> float:
        """GENERALIZATION: mean transfer success rate.

        Why: measures cross-domain knowledge reuse.
        Fallback: returns 0.0 if no transfers attempted.
        """
        if not self._transfer_successes:
            return 0.0
        # Normalize: positive transfer → higher score
        positive = [max(0.0, s) for s in self._transfer_successes]
        return min(1.0, sum(positive) / max(1, len(positive)))

    def _score_autonomy(self) -> float:
        """AUTONOMY: fraction of self-generated goals.

        Why: measures independence from hardcoded tasks.
        Fallback: returns 0.0 if no goals generated.
        """
        if self._total_goals == 0:
            return 0.0
        return min(1.0, self._self_generated_goals / self._total_goals)

    def _score_self_improvement(self) -> float:
        """SELF-IMPROVEMENT: beneficial modification rate.

        Why: measures ability to improve own parameters.
        Fallback: returns 0.0 if no modifications attempted.
        """
        if self._total_self_mods == 0:
            return 0.0
        return min(1.0, self._beneficial_self_mods / self._total_self_mods)

    def _score_abstraction(self) -> float:
        """ABSTRACTION: concept depth / target depth.

        Why: measures hierarchical concept formation capability.
        Fallback: returns 0.0 if no concepts formed.
        """
        return min(1.0, self._concept_depth / TARGET_ABSTRACTION_DEPTH)

    def _score_open_endedness(self) -> float:
        """OPEN-ENDEDNESS: growth rate of domains and difficulties.

        Why: measures ability to expand beyond initial problem space.
        Fallback: returns 0.0 if no growth observed.
        """
        if self._rounds_elapsed == 0:
            return 0.0
        growth_rate = (
            (self._new_domains + self._difficulty_increases) /
            self._rounds_elapsed
        )
        # Normalize: expect ~0.5 growth events per round for score 1.0
        return min(1.0, growth_rate / 0.5)
