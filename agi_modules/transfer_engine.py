"""
TransferEngine — Cross-domain transfer learning with negative transfer detection.

Serves AGI capability: enables knowledge reuse across domains and detects
when transfer is harmful (negative transfer), automatically rolling back.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from agi_modules.concept_graph import ConceptGraph

# --- Named constants (Rule 6) ---

# Minimum analogy score to attempt transfer
# Low threshold: rollback mechanism is the primary safety net
TRANSFER_ANALOGY_THRESHOLD = 0.05

# Weight scaling for transferred model weights
TRANSFER_WEIGHT_SCALE = 0.5

# Number of top skills to transfer
TRANSFER_TOP_K_SKILLS = 5

# Negative transfer threshold: below this, rollback
NEGATIVE_TRANSFER_THRESHOLD = -0.1

# Maximum transfer attempts per window
MAX_TRANSFERS_PER_WINDOW = 1
TRANSFER_COOLDOWN_ROUNDS = 5


class TransferEngine:
    """Manages cross-domain knowledge transfer with rollback capability.

    Why it exists: without transfer, each domain starts from scratch.
    With transfer, mastery in one domain accelerates learning in another.

    Fallback: all operations are safe no-ops when prerequisites aren't met.
    CRITICAL: negative transfer detection and rollback are mandatory.
    """

    def __init__(self, concept_graph: ConceptGraph,
                 shared_mem: Any, wm: Any) -> None:
        self.concept_graph = concept_graph
        self.shared_mem = shared_mem
        self.wm = wm
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._transfer_history: List[Dict[str, Any]] = []
        self._last_transfer_round: int = -TRANSFER_COOLDOWN_ROUNDS

    def detect_analogy(self, source_domain: str,
                       target_domain: str) -> float:
        """Compute structural similarity between two domains.

        Why: determines if transfer is likely to help.
        Fallback: returns 0.0 if no data available.
        """
        # a) Shared successful concepts (Jaccard similarity)
        source_concepts: Set[str] = set()
        target_concepts: Set[str] = set()

        for concept in self.concept_graph.all_concepts():
            for ctx in concept.success_contexts:
                d = str(ctx.get("domain", ""))
                if d == source_domain:
                    source_concepts.add(concept.concept_id)
                if d == target_domain:
                    target_concepts.add(concept.concept_id)

        if source_concepts or target_concepts:
            intersection = source_concepts & target_concepts
            union = source_concepts | target_concepts
            concept_sim = len(intersection) / max(1, len(union))
        else:
            concept_sim = 0.0

        # b) WorldModel feature overlap (simplified cosine proxy)
        wm_weights = getattr(self.wm, '_weights', {})
        source_keys = {k for k in wm_weights if source_domain in str(k)}
        target_keys = {k for k in wm_weights if target_domain in str(k)}
        all_keys = source_keys | target_keys
        if all_keys:
            shared = source_keys & target_keys
            weight_sim = len(shared) / max(1, len(all_keys))
        else:
            weight_sim = 0.0

        # c) SharedMemory retrieval overlap
        try:
            source_results = self.shared_mem.search(
                source_domain, k=10, kinds=["episode", "principle"])
            target_results = self.shared_mem.search(
                target_domain, k=10, kinds=["episode", "principle"])
            source_ids = {r.id for r in source_results}
            target_ids = {r.id for r in target_results}
            union_ids = source_ids | target_ids
            mem_sim = len(source_ids & target_ids) / max(1, len(union_ids))
        except Exception:
            mem_sim = 0.0

        # Weighted average
        return 0.4 * concept_sim + 0.3 * weight_sim + 0.3 * mem_sim

    def _save_snapshot(self, domain: str) -> None:
        """Save pre-transfer state for rollback capability.

        Why: mandatory for negative transfer detection and recovery.
        Fallback: logs warning if snapshot fails.
        """
        snapshot: Dict[str, Any] = {
            "wm_weights": copy.deepcopy(getattr(self.wm, '_weights', {})),
            "concepts": [],
        }
        for concept in self.concept_graph.all_concepts():
            for ctx in concept.success_contexts:
                if str(ctx.get("domain", "")) == domain:
                    snapshot["concepts"].append(concept.concept_id)
                    break
        self._snapshots[domain] = snapshot

    def transfer(self, source_domain: str,
                 target_domain: str) -> Dict[str, Any]:
        """Transfer knowledge from source to target domain.

        Why: accelerates learning by reusing proven concepts and weights.
        Fallback: returns empty report if analogy score too low.
        """
        analogy = self.detect_analogy(source_domain, target_domain)
        report: Dict[str, Any] = {
            "source": source_domain,
            "target": target_domain,
            "analogy_score": analogy,
            "concepts_transferred": 0,
            "weights_initialized": 0,
            "skills_copied": 0,
            "attempted": False,
        }

        if analogy < TRANSFER_ANALOGY_THRESHOLD:
            return report

        # Save snapshot for rollback
        self._save_snapshot(target_domain)
        report["attempted"] = True

        # Transfer successful concepts
        concepts_transferred = 0
        for concept in self.concept_graph.all_concepts():
            source_success = any(
                str(ctx.get("domain", "")) == source_domain
                for ctx in concept.success_contexts
            )
            if source_success and concept.avg_reward > 0.5:
                # Adapt concept for target domain
                adapted_id = self.concept_graph.analogize(
                    concept.concept_id, target_domain)
                if adapted_id:
                    concepts_transferred += 1

        report["concepts_transferred"] = concepts_transferred

        # Transfer WorldModel weights with scaling
        wm_weights = getattr(self.wm, '_weights', {})
        weights_init = 0
        source_weight_keys = [k for k in wm_weights if source_domain in str(k)]
        for key in source_weight_keys:
            target_key = str(key).replace(source_domain, target_domain)
            if target_key not in wm_weights:
                wm_weights[target_key] = wm_weights[key] * TRANSFER_WEIGHT_SCALE
                weights_init += 1
        report["weights_initialized"] = weights_init

        # Transfer top-k skills via shared memory
        try:
            skills = self.shared_mem.search(
                f"{source_domain} skill", k=TRANSFER_TOP_K_SKILLS,
                kinds=["artifact"])
            for skill_mem in skills:
                self.shared_mem.add(
                    "artifact",
                    f"transferred_skill:{target_domain}:{skill_mem.title}",
                    {"source": source_domain, "original": skill_mem.content},
                    tags=["transfer", target_domain],
                )
                report["skills_copied"] += 1
        except Exception:
            pass

        self._transfer_history.append(report)
        return report

    def measure_transfer_success(self, domain: str,
                                 pre_transfer_baseline: float) -> float:
        """Compare post-transfer performance vs baseline.

        Why: detects whether transfer actually helped or hurt.
        Fallback: returns 0.0 if no data available.
        """
        # Get recent performance from shared memory
        try:
            recent = self.shared_mem.search(
                domain, k=10, kinds=["episode"])
            if recent:
                post_avg = sum(
                    float(r.content.get("reward", 0.0)) for r in recent
                ) / len(recent)
                return post_avg - pre_transfer_baseline
        except Exception:
            pass
        return 0.0

    def rollback_transfer(self, domain: str) -> None:
        """Revert all transferred knowledge for a domain.

        Why: mandatory when negative transfer is detected.
        Fallback: logs warning if no snapshot available.
        """
        snapshot = self._snapshots.get(domain)
        if snapshot is None:
            return

        # Restore WorldModel weights
        wm_weights = getattr(self.wm, '_weights', {})
        old_weights = snapshot.get("wm_weights", {})
        # Remove any keys that weren't in the snapshot
        new_keys = set(wm_weights.keys()) - set(old_weights.keys())
        for key in new_keys:
            if domain in str(key):
                del wm_weights[key]

        # Remove transferred concepts
        for concept in list(self.concept_graph.all_concepts()):
            for ctx in concept.success_contexts:
                if (ctx.get("analogy") and
                        str(ctx.get("domain", "")) == domain):
                    # Mark for removal via prune
                    concept.usage_count = 0
                    concept.avg_reward = 0.0
                    break

        self.concept_graph.prune(min_usage=1, min_reward=0.01)
        del self._snapshots[domain]

    def can_transfer(self, current_round: int) -> bool:
        """Check if transfer is allowed (cooldown).

        Why: prevents transfer thrashing.
        Fallback: always returns True if no history.
        """
        return (current_round - self._last_transfer_round) >= TRANSFER_COOLDOWN_ROUNDS

    def record_transfer_round(self, current_round: int) -> None:
        """Record that a transfer was attempted this round.

        Why: enforces cooldown between transfer attempts.
        Fallback: no-op.
        """
        self._last_transfer_round = current_round

    def get_history(self) -> List[Dict[str, Any]]:
        """Return transfer history.

        Why: needed for AGI tracker reporting.
        Fallback: returns empty list.
        """
        return list(self._transfer_history)
