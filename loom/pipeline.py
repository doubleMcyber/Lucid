"""Loom pipeline: sample -> validate -> label -> shape -> write (PRD §8, §9).

Data flows one direction. Each accepted, de-duplicated program is turned into a
multi-pair training record under the documented schema and streamed to sharded
JSONL with full provenance. The run is reproducible: identical
(seed, generator_version, language_version, config) yields identical bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Optional

from lucid import LANGUAGE_VERSION
from lucid.hashing import ast_hash
from lucid.interp import ExecConfig
from lucid.printer import print_program

from . import GENERATOR_VERSION
from . import labelers as lab
from .config import SamplerConfig
from .coverage import CoverageTracker
from .curriculum import curriculum_stages
from .features import difficulty as difficulty_of
from .rng import Rng
from .sampler import Sampler
from .validator import Validator
from .writer import ShardWriter


@dataclass
class GenSpec:
    out_dir: str
    n_examples: int = 100
    base_seed: int = 0
    shard_size: int = 1000
    use_curriculum: bool = True
    sampler_config: Optional[SamplerConfig] = None  # used when not curriculum
    n_inputs: int = 4
    exec_config: ExecConfig = field(default_factory=ExecConfig)
    # labeler toggles (off => faster)
    want_trace: bool = True
    want_completion: bool = True
    want_repair: bool = True
    want_refactor: bool = True
    want_spec: bool = True
    dedup: bool = True
    coverage_guided: bool = True
    max_attempts_factor: int = 20
    # optional pluggable LLM paraphrase + back-translator
    paraphraser: Optional[lab.Paraphraser] = None
    synthesizer: Optional[lab.Synthesizer] = None


@dataclass
class GenStats:
    written: int = 0
    sampled: int = 0
    duplicates: int = 0
    typecheck_bugs: int = 0
    discarded_no_io: int = 0
    attempts: int = 0
    manifest_path: str = ""
    coverage: dict = field(default_factory=dict)


def _config_fingerprint(spec: GenSpec) -> str:
    parts = [
        LANGUAGE_VERSION, GENERATOR_VERSION, str(spec.base_seed),
        str(spec.use_curriculum), repr(spec.sampler_config),
        str(spec.n_inputs), repr(spec.exec_config),
        str(spec.want_trace), str(spec.want_completion), str(spec.want_repair),
        str(spec.want_refactor), str(spec.want_spec), str(spec.dedup),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class Pipeline:
    def __init__(self, spec: GenSpec):
        self.spec = spec
        self.validator = Validator(spec.exec_config, n_inputs=spec.n_inputs)
        if spec.use_curriculum:
            self.stages = curriculum_stages()
        else:
            self.stages = [spec.sampler_config or SamplerConfig()]
        self.tracker = CoverageTracker()  # over written records
        self.seen: set[str] = set()
        self._n_sampled = 0
        self._n_dup = 0

    def _stage_for(self, accepted: int) -> int:
        if len(self.stages) == 1:
            return 0
        per = max(1, (self.spec.n_examples + len(self.stages) - 1) // len(self.stages))
        return min(accepted // per, len(self.stages) - 1)

    def _active_config(self, stage: int, boost: bool) -> SamplerConfig:
        cfg = self.stages[stage]
        if boost:
            # Only scale weights for under-covered constructs; never enable a
            # feature the current curriculum stage has turned off (that would
            # corrupt the curriculum_stage label).
            cfg = replace(cfg, w_builtin=cfg.w_builtin * 2.5, w_hof=cfg.w_hof * 2.5)
        return cfg

    def run(self) -> GenStats:
        spec = self.spec
        base = Rng(spec.base_seed)
        writer = ShardWriter(spec.out_dir, spec.shard_size)
        stats = GenStats()

        accepted = 0
        attempt = 0
        max_attempts = max(spec.n_examples * spec.max_attempts_factor, 50)

        while accepted < spec.n_examples and attempt < max_attempts:
            stage = self._stage_for(accepted)
            boost = (
                spec.coverage_guided
                and accepted > spec.n_examples * 0.3
                and self.tracker.production_coverage() < 0.99
            )
            cfg = self._active_config(stage, boost)

            ex_rng = base.spawn(attempt)
            attempt += 1
            stats.attempts = attempt

            module = Sampler(cfg).sample_program(ex_rng.spawn(1))
            self._n_sampled += 1
            stats.sampled = self._n_sampled

            ah = ast_hash(module.program)
            if spec.dedup and ah in self.seen:
                self._n_dup += 1
                stats.duplicates = self._n_dup
                continue

            report = self.validator.validate(module, ex_rng.spawn(2))
            if not report.typecheck_ok:
                stats.typecheck_bugs += 1
                continue
            if not report.accepted:
                stats.discarded_no_io += 1
                continue

            self.seen.add(ah)
            record = self._build_record(module, report, ex_rng, accepted, stage, ah)
            writer.write(record)
            self.tracker.update(module.program, ah)
            accepted += 1
            stats.written = accepted

        cov = self.tracker.report()
        # Diversity = distinct-canonical-AST ratio over WRITTEN records (PRD §8.4
        # "within a shard"). self.seen is a set, so its size is the unique count;
        # this is correct whether or not dedup skipping is enabled.
        unique_written = len(self.seen)
        cov["diversity_ratio"] = round(unique_written / accepted, 5) if accepted else 0.0
        cov["unique_written"] = unique_written
        cov["written"] = accepted
        cov["duplicates_written"] = accepted - unique_written
        cov["sampled"] = self._n_sampled
        cov["duplicates_skipped"] = self._n_dup
        stats.coverage = cov

        manifest = {
            "language_version": LANGUAGE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "base_seed": spec.base_seed,
            "config_fingerprint": _config_fingerprint(spec),
            "use_curriculum": spec.use_curriculum,
            "n_requested": spec.n_examples,
            "n_written": accepted,
            "attempts": attempt,
            "typecheck_bugs": stats.typecheck_bugs,
            "discarded_no_io": stats.discarded_no_io,
            "coverage": cov,
        }
        stats.manifest_path = writer.close(manifest)
        return stats

    # -- record assembly ---------------------------------------------------
    def _build_record(self, module, report, ex_rng, idx, stage, ah) -> dict:
        spec = self.spec
        canon = print_program(module.program)
        rec: dict = {
            "id": f"luc-{idx:06d}",
            "language_version": LANGUAGE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "seed": ex_rng.seed,
            "curriculum_stage": stage,
            "difficulty": difficulty_of(module.program),
            "program_canonical": canon,
            "ast_hash": ah,
            "io_examples": lab.io_examples(report, limit=spec.n_inputs),
            "trace_available": False,
            "spec_verified": False,
        }
        pair_types = ["spec_to_code", "io_to_code"]

        if spec.want_spec:
            sp = lab.spec_for(module)
            rec["type_signature"] = sp["type_signature"]
            rec["spec_structured"] = sp["spec_structured"]
            rec["features"] = sp["features"]
            if spec.paraphraser is not None:
                vp = lab.validated_paraphrase(
                    sp, module, report, spec.paraphraser, spec.synthesizer,
                    spec.exec_config,
                )
                if vp is not None:
                    rec["spec_nl_paraphrase"] = vp["spec_nl_paraphrase"]
                    rec["spec_verified"] = vp["spec_verified"]

        if spec.want_trace:
            tr = lab.execution_trace(module, report, spec.exec_config)
            if tr is not None:
                rec["trace"] = tr
                rec["trace_available"] = True
                pair_types.append("trace")

        if spec.want_completion:
            cps = lab.completion_pairs(canon, ex_rng.spawn(3))
            if cps:
                rec["completion_pairs"] = cps
                pair_types.append("completion")

        if spec.want_repair:
            rps = lab.repair_pairs(module, canon, report, ex_rng.spawn(4), spec.exec_config)
            if rps:
                rec["repair_pairs"] = rps
                pair_types.append("repair")

        if spec.want_refactor:
            fps = lab.refactor_pairs(module, canon, report, ex_rng.spawn(5), spec.exec_config)
            if fps:
                rec["refactor_pairs"] = fps
                pair_types.append("refactor")

        rec["pair_types"] = pair_types
        return rec


def generate(spec: GenSpec) -> GenStats:
    return Pipeline(spec).run()
