"""Curriculum control (PRD §8.4).

Produces an ordered list of `SamplerConfig` stages from easy to hard by toggling
feature flags on and scaling size/depth budgets up. The pipeline assigns each
emitted example to a stage so the dataset streams as an easy->hard progression;
each record carries its `curriculum_stage` and computed `difficulty`.
"""

from __future__ import annotations

from dataclasses import replace

from .config import SamplerConfig


def curriculum_stages() -> list[SamplerConfig]:
    """A fixed 6-stage progression. Each stage is a superset of the previous
    in features and/or budget."""
    stages: list[SamplerConfig] = []

    # Stage 0 — straight-line integer arithmetic only (no bool, lists, control
    # flow, or mutation): genuinely the simplest programs.
    stages.append(SamplerConfig(
        use_bool=False, use_lists=False,
        use_records=False, use_variants=False, use_match=False, use_cond=False,
        use_foreach=False, use_if=False, use_strings=False, use_higher_order=False,
        use_nested_lists=False,
        max_records=0, max_variants=0, min_helpers=0, max_helpers=1,
        min_params=1, max_params=2, max_block_stmts=2, block_depth=0,
        expr_fuel=2, helper_fuel=2, p_mutable=0.0,
    ))

    # Stage 1 — add booleans, lists, and control flow (foreach, if, cond).
    stages.append(SamplerConfig(
        use_bool=True, use_lists=True,
        use_records=False, use_variants=False, use_match=False, use_cond=True,
        use_foreach=True, use_if=True, use_strings=False, use_higher_order=False,
        use_nested_lists=False,
        max_records=0, max_variants=0, min_helpers=0, max_helpers=2,
        max_block_stmts=3, block_depth=1, expr_fuel=3, helper_fuel=2,
    ))

    # Stage 2 — add strings, deeper expressions.
    stages.append(SamplerConfig(
        use_records=False, use_variants=False, use_match=False, use_cond=True,
        use_foreach=True, use_if=True, use_strings=True, use_higher_order=False,
        use_nested_lists=False,
        max_records=0, max_variants=0, max_helpers=2,
        block_depth=2, expr_fuel=3,
    ))

    # Stage 3 — add records.
    stages.append(SamplerConfig(
        use_records=True, use_variants=False, use_match=False, use_cond=True,
        use_foreach=True, use_if=True, use_strings=True, use_higher_order=False,
        use_nested_lists=True, max_records=2, max_variants=0,
        block_depth=2, expr_fuel=4,
    ))

    # Stage 4 — add variants + match.
    stages.append(SamplerConfig(
        use_records=True, use_variants=True, use_match=True, use_cond=True,
        use_foreach=True, use_if=True, use_strings=True, use_higher_order=False,
        use_nested_lists=True,
        block_depth=2, expr_fuel=4,
    ))

    # Stage 5 — everything, including higher-order functions (full default).
    stages.append(SamplerConfig())

    return stages


def difficulty_of_stage(stage: int) -> int:
    return stage + 1
