"""Loom: a constrained, type-directed corpus generator for Lucid.

Pipeline (PRD §8): sample -> validate -> label -> shape -> write. Every stage
calls back into the one Lucid toolchain, so there is a single source of truth
for what a valid program is and what it means.
"""

from __future__ import annotations

GENERATOR_VERSION = "0.1.0"
