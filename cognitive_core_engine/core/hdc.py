"""Hyperdimensional Computing (HDC) Core extracted from NON_RSI_AGI_CORE_v5."""

from __future__ import annotations

import hashlib
import random
from typing import Any, List, Optional


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

    def fractional_bind(self, other: HyperVector, role_index: int) -> HyperVector:
        """Structure-preserving binding using permute-then-XOR (SDM principle).

        Unlike naive XOR which causes dimension collapse when binding many
        vectors, fractional binding permutes `other` by a unique role_index
        before XOR. This preserves each component's structural information
        in a distinct subspace of the hypervector, preventing the bound
        result from collapsing into noise.

        role_index: unique integer per component being bound (0, 1, 2, ...)
        """
        shifted = other.permute(role_index * 7 + 1)  # large prime-ish stride
        return HyperVector(self.val ^ shifted.val)

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

    def cosine_similarity(self, other: HyperVector) -> float:
        """Cosine-like similarity for binary vectors.

        Why: TransferEngine._cosine_similarity calls this. For binary HDC
        vectors, Hamming similarity is the analogous metric to cosine
        similarity in continuous spaces. Maps to [-1, 1] range.
        """
        ham = self.similarity(other)
        return 2.0 * ham - 1.0  # map [0,1] → [-1,1]

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
                # Deterministic tie-breaking using bit position parity
                # Avoids perturbing global random state
                if i % 2 == 0:
                    result_val |= (1 << i)

        return HyperVector(result_val)
