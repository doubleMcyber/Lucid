"""Streamed, sharded dataset writer with provenance (PRD §8.5).

Records are written as JSON Lines into rotating shards. JSON is emitted
deterministically (sorted keys, compact separators) so that identical
(seed, generator-version, language-version, config) reproduces identical bytes.
A manifest records the run's provenance, per-shard counts, and the coverage
report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _dumps(obj) -> str:
    # Deterministic, compact, ascii-safe JSON.
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


@dataclass
class ShardWriter:
    out_dir: str
    shard_size: int = 1000
    prefix: str = "luc"
    _shard_idx: int = 0
    _in_shard: int = 0
    _fh: object = None
    shards: list[dict] = field(default_factory=list)
    total: int = 0

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)

    def _open_new_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
        name = f"{self.prefix}-{self._shard_idx:05d}.jsonl"
        path = os.path.join(self.out_dir, name)
        self._fh = open(path, "w", encoding="utf-8")
        self.shards.append({"name": name, "count": 0})
        self._in_shard = 0

    def write(self, record: dict) -> None:
        if self._fh is None or self._in_shard >= self.shard_size:
            if self._fh is not None:
                self._shard_idx += 1
            self._open_new_shard()
        self._fh.write(_dumps(record) + "\n")
        self._in_shard += 1
        self.total += 1
        self.shards[-1]["count"] = self._in_shard

    def close(self, manifest: dict) -> str:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        manifest = dict(manifest)
        manifest["shards"] = self.shards
        manifest["total_records"] = self.total
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2))
        return path
