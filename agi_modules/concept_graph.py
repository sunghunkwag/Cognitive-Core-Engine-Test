"""
ConceptGraph — Hierarchical abstraction and concept formation.

Serves AGI capability: enables the system to form multi-level abstractions
from raw actions to meta-strategies, supporting transfer and generalization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# --- Named constants (Rule 6) ---

# Promotion thresholds (calibrated for env rewards ~0.02-0.15)
PROMOTE_MIN_USAGE = 3       # Minimum uses before promotion considered
PROMOTE_MIN_REWARD = 0.03   # Minimum avg reward for promotion (env baseline)
CO_OCCUR_WINDOW = 2         # Co-occurrence count required for bundling

# Pruning thresholds
PRUNE_MIN_USAGE = 3         # Below this usage count, eligible for pruning
PRUNE_MIN_REWARD = 0.2      # Below this avg reward, eligible for pruning

# Target abstraction depth for AGI scoring
TARGET_DEPTH = 5

# Maximum concept levels
MAX_LEVEL = 5


def _concept_hash(name: str, children: List[str]) -> str:
    """Create deterministic concept ID from name and children.

    Why: deduplication — avoid creating the same concept twice.
    Fallback: uses name-only hash if no children.
    """
    data = json.dumps({"name": name, "children": sorted(children)}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass
class ConceptNode:
    """A node in the concept hierarchy.

    Level 0 = raw action, 1 = skill, 2 = strategy, 3+ = meta-strategy.
    """
    concept_id: str
    name: str
    level: int
    children: List[str] = field(default_factory=list)
    success_contexts: List[Dict[str, Any]] = field(default_factory=list)
    failure_contexts: List[Dict[str, Any]] = field(default_factory=list)
    creation_round: int = 0
    usage_count: int = 0
    avg_reward: float = 0.0
    _total_reward: float = 0.0


class ConceptGraph:
    """Stores and manages hierarchical concept abstractions.

    Why it exists: without concept formation, the agent operates on flat
    action→reward mappings with no ability to generalize or compose strategies.

    Fallback: empty graph returns sensible defaults (depth=0, no promotions).
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ConceptNode] = {}
        self._co_occurrences: Dict[str, Dict[str, int]] = {}

    def add_concept(self, name: str, level: int, children: List[str],
                    context: Dict[str, Any],
                    creation_round: int = 0) -> str:
        """Create a new ConceptNode, deduplicating by children combo.

        Why: builds the concept hierarchy from observed action patterns.
        Fallback: returns existing concept_id if duplicate detected.
        """
        concept_id = _concept_hash(name, children)

        # Deduplicate
        if concept_id in self._nodes:
            return concept_id

        node = ConceptNode(
            concept_id=concept_id,
            name=name,
            level=min(level, MAX_LEVEL),
            children=list(children),
            success_contexts=[context] if context else [],
            creation_round=creation_round,
        )
        self._nodes[concept_id] = node
        return concept_id

    def record_usage(self, concept_id: str, reward: float,
                     context: Dict[str, Any], success: bool) -> None:
        """Record usage of a concept with outcome.

        Why: tracks statistics needed for promotion and pruning decisions.
        Fallback: no-op if concept_id not found.
        """
        node = self._nodes.get(concept_id)
        if node is None:
            return

        node.usage_count += 1
        node._total_reward += reward
        node.avg_reward = node._total_reward / node.usage_count

        if success:
            node.success_contexts.append(context)
            # Trim to prevent unbounded growth
            if len(node.success_contexts) > 50:
                node.success_contexts = node.success_contexts[-50:]
        else:
            node.failure_contexts.append(context)
            if len(node.failure_contexts) > 50:
                node.failure_contexts = node.failure_contexts[-50:]

    def record_co_occurrence(self, concept_a: str, concept_b: str) -> None:
        """Track co-occurrence of concepts for promotion bundling.

        Why: concepts that succeed together may form higher-level abstractions.
        Fallback: no-op if either concept not found.
        """
        if concept_a not in self._nodes or concept_b not in self._nodes:
            return
        if concept_a not in self._co_occurrences:
            self._co_occurrences[concept_a] = {}
        self._co_occurrences[concept_a][concept_b] = (
            self._co_occurrences[concept_a].get(concept_b, 0) + 1
        )

    def promote(self, concept_id: str, current_round: int = 0) -> Optional[str]:
        """Promote a concept to a higher level by bundling with co-occurring peers.

        Why: builds abstraction hierarchy — successful concept combos become strategies.
        Fallback: returns None if promotion criteria not met.
        """
        node = self._nodes.get(concept_id)
        if node is None:
            return None

        if node.usage_count < PROMOTE_MIN_USAGE:
            return None
        if node.avg_reward < PROMOTE_MIN_REWARD:
            return None

        # Find co-occurring concepts at the same level
        co_map = self._co_occurrences.get(concept_id, {})
        partners = [
            cid for cid, count in co_map.items()
            if count >= CO_OCCUR_WINDOW
            and cid in self._nodes
            and self._nodes[cid].level == node.level
            and self._nodes[cid].avg_reward >= PROMOTE_MIN_REWARD
        ]

        if not partners:
            return None

        # Bundle with top co-occurring partner
        partner_id = max(partners, key=lambda c: co_map[c])
        partner = self._nodes[partner_id]

        new_level = node.level + 1
        new_name = f"L{new_level}:{node.name}+{partner.name}"
        children = [concept_id, partner_id]

        # Merge contexts
        merged_context = {
            "source_concepts": [node.name, partner.name],
            "promotion_round": current_round,
        }

        return self.add_concept(new_name, new_level, children,
                                merged_context, current_round)

    def abstract(self, concept_id: str) -> Dict[str, Any]:
        """Return abstract representation of a concept's transfer radius.

        Why: determines which domains/difficulties a concept generalizes across.
        Fallback: returns empty dict if concept not found.
        """
        node = self._nodes.get(concept_id)
        if node is None:
            return {}

        domains: Set[str] = set()
        difficulties: Set[int] = set()

        for ctx in node.success_contexts:
            if "domain" in ctx:
                domains.add(str(ctx["domain"]))
            if "difficulty" in ctx:
                difficulties.add(int(ctx["difficulty"]))

        return {
            "concept_id": concept_id,
            "name": node.name,
            "level": node.level,
            "domains": sorted(domains),
            "difficulties": sorted(difficulties),
            "transfer_radius": len(domains) * len(difficulties),
            "usage_count": node.usage_count,
            "avg_reward": node.avg_reward,
        }

    def analogize(self, source_concept_id: str,
                  target_domain: str) -> Optional[str]:
        """Find if source concept structure could apply in target domain.

        Why: enables transfer learning by reusing proven abstractions.
        Fallback: returns None if no viable analogy found.
        """
        source = self._nodes.get(source_concept_id)
        if source is None:
            return None

        # Check if source has succeeded in diverse contexts
        source_domains = {
            str(ctx.get("domain", ""))
            for ctx in source.success_contexts
        }

        if len(source_domains) < 2:
            return None  # Not general enough to transfer

        # Create adapted concept for target domain
        adapted_name = f"analogy:{source.name}→{target_domain}"
        context = {
            "domain": target_domain,
            "source_concept": source_concept_id,
            "analogy": True,
        }
        return self.add_concept(
            adapted_name, source.level, source.children,
            context, source.creation_round
        )

    def prune(self, min_usage: int = PRUNE_MIN_USAGE,
              min_reward: float = PRUNE_MIN_REWARD) -> int:
        """Remove underperforming concepts. Never deletes high-usage concepts.

        Why: prevents concept bloat and focuses on useful abstractions.
        Fallback: returns 0 if nothing to prune.
        """
        to_remove = [
            cid for cid, node in self._nodes.items()
            if node.usage_count < min_usage and node.avg_reward < min_reward
            and node.usage_count <= 10  # Never prune well-used concepts
        ]

        for cid in to_remove:
            del self._nodes[cid]
            self._co_occurrences.pop(cid, None)

        return len(to_remove)

    def depth(self) -> int:
        """Return maximum concept level in the graph.

        Why: measures abstraction capability — AGI target is depth >= 5.
        Fallback: returns 0 if graph is empty.
        """
        if not self._nodes:
            return 0
        return max(node.level for node in self._nodes.values())

    def concepts_at_level(self, level: int) -> List[ConceptNode]:
        """Return all concepts at a given level.

        Why: used for hierarchical planning to select level-appropriate concepts.
        Fallback: returns empty list if none at that level.
        """
        return [n for n in self._nodes.values() if n.level == level]

    def get(self, concept_id: str) -> Optional[ConceptNode]:
        """Retrieve a concept by ID.

        Why: direct access needed by planner and transfer engine.
        Fallback: returns None if not found.
        """
        return self._nodes.get(concept_id)

    def all_concepts(self) -> List[ConceptNode]:
        """Return all concepts.

        Why: needed for iteration by tracker and diagnostics.
        Fallback: returns empty list.
        """
        return list(self._nodes.values())

    def size(self) -> int:
        """Return total concept count.

        Why: used by AGI tracker for abstraction scoring.
        Fallback: returns 0.
        """
        return len(self._nodes)
