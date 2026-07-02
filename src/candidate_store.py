"""In-memory candidate lookup by id (for API profile enrichment)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import orjson

    def _parse_line(line: bytes | str) -> dict[str, Any]:
        if isinstance(line, str):
            line = line.encode("utf-8")
        return orjson.loads(line)

except ImportError:

    def _parse_line(line: bytes | str) -> dict[str, Any]:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        return json.loads(line)


class CandidateStore:
    """Load JSONL once; O(1) lookup by candidate_id."""

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    @classmethod
    def load(cls, path: Path) -> CandidateStore:
        store = cls()
        with path.open("rb") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                cand = _parse_line(raw)
                cid = cand.get("candidate_id")
                if cid:
                    store._by_id[str(cid)] = cand
        return store

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        return self._by_id.get(candidate_id)
