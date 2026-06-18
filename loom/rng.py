"""Deterministic randomness for Loom.

Everything stochastic in the generator goes through `Rng`, a thin wrapper over
`random.Random` (Mersenne Twister — stable across platforms and Python versions
for a given integer seed). Reproducibility (PRD §8.5: identical bytes from
identical seed) depends on *all* sampling being driven from here, and on the
generator never depending on hash/set iteration order (we only ever pick from
ordered lists).
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


class Rng:
    def __init__(self, seed: int):
        self.seed = seed
        self._r = random.Random(seed)

    def spawn(self, salt: int) -> "Rng":
        """Derive an independent child stream (used for per-example streams)."""
        return Rng((self.seed * 1_000_003 + salt) & 0xFFFFFFFFFFFF)

    def chance(self, p: float) -> bool:
        return self._r.random() < p

    def rand(self) -> float:
        return self._r.random()

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def choice(self, xs: Sequence[T]) -> T:
        return xs[self._r.randrange(len(xs))]

    def sample_k(self, xs: Sequence[T], k: int) -> list[T]:
        k = min(k, len(xs))
        return self._r.sample(list(xs), k)

    def shuffle(self, xs: list[T]) -> list[T]:
        ys = list(xs)
        self._r.shuffle(ys)
        return ys

    def weighted(self, items: Sequence[tuple[float, Callable[[], T]]]) -> T:
        """Pick a thunk by weight and call it. `items` is an ordered list of
        (weight, thunk); order must be deterministic for reproducibility."""
        total = sum(w for w, _ in items)
        r = self._r.random() * total
        upto = 0.0
        for w, thunk in items:
            upto += w
            if r <= upto:
                return thunk()
        return items[-1][1]()  # float fallback
