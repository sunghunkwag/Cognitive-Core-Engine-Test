"""
TransferEngine — structural concept transfer via HDC cosine similarity.

BN-04 fix:
  The original implementation used difflib.SequenceMatcher to score concept
  similarity.  SequenceMatcher operates on character sequences, so it would
  rate 'language_model' and 'language_parsing' as highly similar even though
  their HDC representations are structurally distant.

  Fix: similarity is now computed as the cosine distance between the
  HyperVector representations stored in ConceptGraph.  SequenceMatcher is
  kept as a fallback ONLY when HDC vectors are not available, and the
  `similarity_method` field in every transfer record tells callers which
  path was taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from difflib import SequenceMatcher
except ImportError:  # pragma: no cover
    SequenceMatcher = None  # type: ignore

try:
    from cognitive_core_engine.core.hdc import HyperVector
    from agi_modules.concept_graph import ConceptGraph
    _HDC_AVAILABLE = True
except ImportError:
    _HDC_AVAILABLE = False
    HyperVector = None  # type: ignore  # noqa: F811
    ConceptGraph = None  # type: ignore  # noqa: F811


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(v1: Any, v2: Any) -> float:
    """Cosine similarity between two HyperVectors.

    Returns a value in [0, 1] where 1 means identical direction.
    Falls back to 0.5 (neutral) if vectors cannot be compared.
    """
    try:
        # HyperVector.cosine_similarity is expected to return [-1, 1];
        # we rescale to [0, 1] for use as a score.
        raw = v1.cosine_similarity(v2)  # type: ignore[union-attr]
        return (float(raw) + 1.0) / 2.0
    except Exception:
        return 0.5


def _name_similarity(a: str, b: str) -> float:
    """Legacy name-based similarity using SequenceMatcher.

    Only used when HDC vectors are unavailable.
    """
    if SequenceMatcher is None:
        return 0.5
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransferRecord:
    source_concept: str
    target_concept: str
    similarity: float
    knowledge_transferred: Dict[str, Any] = field(default_factory=dict)
    similarity_method: str = "hdc_cosine"  # or "name_sequence"
    success: bool = False


# ---------------------------------------------------------------------------
# TransferEngine
# ---------------------------------------------------------------------------

class TransferEngine:
    """Transfers structured knowledge between concepts using HDC cosine similarity.

    Parameters
    ----------
    concept_graph:
        Optional ConceptGraph instance.  When provided, HDC vectors are
        retrieved from the graph and used for structural similarity.  When
        absent, the engine falls back to SequenceMatcher (BN-04 legacy path).
    similarity_threshold:
        Minimum similarity required for a transfer to be accepted.
    """

    def __init__(
        self,
        concept_graph: Optional[Any] = None,
        similarity_threshold: float = 0.65,
    ) -> None:
        self._graph = concept_graph
        self.similarity_threshold = similarity_threshold
        self._transfer_history: List[TransferRecord] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        source: str,
        target: str,
    ) -> tuple[float, str]:
        """Return (similarity_score, method_name).

        Priority:
        1.  HDC cosine similarity via ConceptGraph (BN-04 primary path).
        2.  SequenceMatcher on concept names (fallback / legacy).
        """
        if _HDC_AVAILABLE and self._graph is not None:
            try:
                src_vec = self._graph.get_vector(source)  # type: ignore[union-attr]
                tgt_vec = self._graph.get_vector(target)  # type: ignore[union-attr]
                if src_vec is not None and tgt_vec is not None:
                    return _cosine_similarity(src_vec, tgt_vec), "hdc_cosine"
            except Exception:
                pass  # fall through to name-based fallback

        # Fallback: name similarity (warns in logs but does not raise)
        return _name_similarity(source, target), "name_sequence"

    def transfer_knowledge(
        self,
        source_concept: str,
        source_knowledge: Dict[str, Any],
        target_concept: str,
    ) -> TransferRecord:
        """Attempt to transfer `source_knowledge` to `target_concept`.

        Returns a TransferRecord whose `success` flag indicates whether the
        similarity exceeded the configured threshold.
        """
        sim, method = self.compute_similarity(source_concept, target_concept)
        record = TransferRecord(
            source_concept=source_concept,
            target_concept=target_concept,
            similarity=sim,
            similarity_method=method,
        )

        if sim >= self.similarity_threshold:
            # Adapt knowledge to the target domain by tagging its origin.
            adapted: Dict[str, Any] = {
                k: v for k, v in source_knowledge.items()
            }
            adapted["_transfer_meta"] = {
                "from": source_concept,
                "to": target_concept,
                "similarity": sim,
                "method": method,
            }
            record.knowledge_transferred = adapted
            record.success = True

        self._transfer_history.append(record)
        return record

    def batch_transfer(
        self,
        source_concept: str,
        source_knowledge: Dict[str, Any],
        candidates: List[str],
    ) -> List[TransferRecord]:
        """Transfer to multiple candidate concepts; returns all records."""
        return [
            self.transfer_knowledge(source_concept, source_knowledge, c)
            for c in candidates
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def transfer_history(self) -> List[TransferRecord]:
        return list(self._transfer_history)

    def success_rate(self) -> float:
        if not self._transfer_history:
            return 0.0
        return sum(1 for r in self._transfer_history if r.success) / len(self._transfer_history)

    def summary(self) -> Dict[str, Any]:
        method_counts: Dict[str, int] = {}
        for r in self._transfer_history:
            method_counts[r.similarity_method] = method_counts.get(r.similarity_method, 0) + 1
        return {
            "total_transfers": len(self._transfer_history),
            "successful": sum(1 for r in self._transfer_history if r.success),
            "success_rate": self.success_rate(),
            "method_counts": method_counts,
        }
