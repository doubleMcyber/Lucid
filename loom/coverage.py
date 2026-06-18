"""Coverage, diversity, and dedup tracking (PRD §8.4, §4 metrics).

`CoverageTracker` accumulates, over a run:
  * grammar-production coverage (which of `ALL_PRODUCTIONS` were exercised),
  * type-interaction coverage (e.g. foreach over #List[#Int]),
  * dedup state (canonical-AST hashes), and the distinct-AST diversity ratio.

It also exposes `under_covered()` so the pipeline can bias sampling toward
thin regions (coverage-guided generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lucid.ast import Program

from . import features as feat
from .features import ALL_PRODUCTIONS


@dataclass
class CoverageTracker:
    prod_counts: dict[str, int] = field(default_factory=lambda: {p: 0 for p in sorted(ALL_PRODUCTIONS)})
    interactions: set[str] = field(default_factory=set)
    seen_hashes: set[str] = field(default_factory=set)
    n_total: int = 0
    n_unique: int = 0

    def is_duplicate(self, ast_hash: str) -> bool:
        return ast_hash in self.seen_hashes

    def update(self, program: Program, ast_hash: str) -> bool:
        """Record a program. Returns True if it is new (not a duplicate)."""
        self.n_total += 1
        new = ast_hash not in self.seen_hashes
        if new:
            self.seen_hashes.add(ast_hash)
            self.n_unique += 1
        w = feat.analyze(program)
        for p in w.prods:
            if p in self.prod_counts:
                self.prod_counts[p] += 1
        for a, b in w.interactions:
            self.interactions.add(f"{a}::{b}")
        return new

    # -- metrics -----------------------------------------------------------
    def covered_productions(self) -> list[str]:
        return sorted(p for p, c in self.prod_counts.items() if c > 0)

    def missing_productions(self) -> list[str]:
        return sorted(p for p, c in self.prod_counts.items() if c == 0)

    def production_coverage(self) -> float:
        covered = sum(1 for c in self.prod_counts.values() if c > 0)
        return covered / len(self.prod_counts) if self.prod_counts else 0.0

    def diversity_ratio(self) -> float:
        return self.n_unique / self.n_total if self.n_total else 0.0

    def under_covered(self, threshold: int) -> list[str]:
        return sorted(p for p, c in self.prod_counts.items() if c < threshold)

    def report(self) -> dict:
        return {
            "production_coverage": round(self.production_coverage(), 5),
            "productions_covered": sum(1 for c in self.prod_counts.values() if c > 0),
            "productions_total": len(self.prod_counts),
            "missing_productions": self.missing_productions(),
            "type_interactions": sorted(self.interactions),
            "type_interaction_count": len(self.interactions),
            "diversity_ratio": round(self.diversity_ratio(), 5),
            "unique": self.n_unique,
            "total_seen": self.n_total,
        }
