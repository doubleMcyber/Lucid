"""Semantic validator (PRD §8.2).

Two jobs:
  1. Typecheck. This should *always* pass (the sampler builds well-typed ASTs by
     construction). A failure here is a generator bug, flagged loudly via
     `typecheck_ok=False` so the pipeline can log it.
  2. Execute on sampled inputs under resource bounds to obtain IO pairs. Runs
     that trip a bound (ResourceError) or fault are skipped; a program that
     produces no successful run on any sampled input is discarded.

The validator hands downstream labelers a bundle of successful (raw input, raw
output) pairs so every labeler shares one execution truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from lucid.errors import RuntimeError_, TypeError_
from lucid.interp import ExecConfig, Interpreter, value_to_json
from lucid.parser import Module
from lucid.typechecker import typecheck
from lucid.types import TFn, Type

from .rng import Rng
from .value_sampler import ValueSampler


@dataclass
class IOPair:
    raw_input: list
    raw_output: Any
    input_json: list
    output_json: Any


@dataclass
class ValidationReport:
    typecheck_ok: bool
    typecheck_error: Optional[str] = None
    io: list[IOPair] = field(default_factory=list)
    n_attempts: int = 0
    discard_reason: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.typecheck_ok and bool(self.io)


class Validator:
    def __init__(self, exec_config: Optional[ExecConfig] = None,
                 n_inputs: int = 6, value_int_min: int = -20, value_int_max: int = 20,
                 value_max_list: int = 5):
        self.exec_config = exec_config or ExecConfig()
        self.n_inputs = n_inputs
        self.v_int_min = value_int_min
        self.v_int_max = value_int_max
        self.v_max_list = value_max_list

    def validate(self, module: Module, rng: Rng) -> ValidationReport:
        # 1. Typecheck — must pass; loud failure if not.
        try:
            typecheck(module)
        except TypeError_ as ex:
            return ValidationReport(typecheck_ok=False, typecheck_error=str(ex),
                                    discard_reason="typecheck_bug")

        entry = module.program.entry()
        param_types: list[Type] = [t for _, t in entry.params]

        fn_names_by_type = self._fn_names_by_type(module)
        vs = ValueSampler(module.tenv, rng, fn_names_by_type,
                          int_min=self.v_int_min, int_max=self.v_int_max,
                          max_list=self.v_max_list)

        report = ValidationReport(typecheck_ok=True)
        seen_inputs: set[str] = set()
        attempts = 0
        for _ in range(self.n_inputs * 3):  # oversample to dedup / skip errors
            if len(report.io) >= self.n_inputs:
                break
            attempts += 1
            inputs = vs.sample_inputs(param_types)
            try:
                inputs_json = [value_to_json(v) for v in inputs]
            except TypeError:
                continue
            key = repr(inputs_json)
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            try:
                interp = Interpreter(module, self.exec_config)
                result = interp.run_entry(inputs)
            except (RuntimeError_,) as ex:  # ResourceError is a subclass
                _ = ex
                continue
            try:
                out_json = value_to_json(result.value)
            except TypeError:
                continue
            report.io.append(IOPair(inputs, result.value, inputs_json, out_json))

        report.n_attempts = attempts
        if not report.io:
            report.discard_reason = "no_successful_execution"
        return report

    @staticmethod
    def _fn_names_by_type(module: Module) -> dict:
        out: dict = {}
        for fn in module.program.functions():
            ft = TFn(tuple(t for _, t in fn.params), fn.ret)
            out.setdefault(ft, []).append(fn.name)
        return out
