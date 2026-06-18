"""Configuration for the Loom sampler (PRD §8.1 controls).

Every knob that shapes the distribution lives here: size/depth budgets,
per-construct frequency weights, and feature flags selecting which language
features are in play. The curriculum controller (PRD §8.4) produces a series of
these configs to drive an easy->hard progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_words() -> list[str]:
    return [
        "", "a", "ab", "abc", "x", "foo", "bar", "baz", "hello", "world",
        "lucid", "loom", "node", "leaf", "key", "val", "lo", "hi",
    ]


@dataclass
class SamplerConfig:
    # ---- feature flags (curriculum / ablation) ----
    use_bool: bool = True
    use_lists: bool = True
    use_records: bool = True
    use_variants: bool = True
    use_match: bool = True
    use_cond: bool = True
    use_foreach: bool = True
    use_if: bool = True
    use_strings: bool = True
    use_higher_order: bool = True
    use_nested_lists: bool = True

    # ---- size / shape limits ----
    max_records: int = 2
    max_variants: int = 2
    max_fields: int = 3
    max_ctors: int = 3
    max_payload: int = 2
    min_helpers: int = 1
    max_helpers: int = 3
    min_params: int = 1
    max_params: int = 3
    min_block_stmts: int = 0
    max_block_stmts: int = 3
    max_loop_stmts: int = 1
    max_list_lit: int = 3
    block_depth: int = 2      # max nesting of foreach/if in the entry
    helper_depth: int = 1
    expr_fuel: int = 3        # expression depth budget in the entry
    helper_fuel: int = 2

    # ---- literal ranges ----
    int_min: int = 0
    int_max: int = 20
    words: list[str] = field(default_factory=_default_words)

    # ---- behavioral probabilities ----
    p_mutable: float = 0.4
    p_if_return: float = 0.25

    # ---- producer weights ----
    w_var: float = 3.0
    w_lit: float = 2.5
    w_call: float = 1.2
    w_field: float = 1.0
    w_cond: float = 0.8
    w_match: float = 0.8
    w_builtin: float = 1.0
    w_hof: float = 0.7
    w_listlit: float = 1.0
    w_record: float = 1.0
    w_variant: float = 1.0
    w_bind: float = 2.0
    w_set: float = 1.5
    w_foreach: float = 1.5
    w_if: float = 1.0
