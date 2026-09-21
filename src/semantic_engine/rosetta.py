# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 11: Rosetta mapping — Latin semantic IDs → SUBLEQ IR.

Only canonical vocabulary reaches this layer.
Direct arbitrary Latin→machine mapping is prohibited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .latin import LatinCanonicalForm, LatinToken
from .lexer import LexResult, LexedToken
from .types import FSMState, RosettaError


@dataclass(frozen=True)
class RosettaEntry:
    semantic_id: str
    subleq_op: str
    template: str
    args: tuple[str, ...]


@dataclass
class RosettaMapping:
    """Result of mapping a LatinCanonicalForm to SUBLEQ operations."""
    entries: list[tuple[LatinToken, RosettaEntry]] = field(default_factory=list)
    unmapped: list[LatinToken] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.unmapped) == 0


# ── Table loader ──────────────────────────────────────────────────────────────

def _load_table() -> dict[str, RosettaEntry]:
    path = Path(__file__).parent.parent.parent / "mappings" / "rosetta_table.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        sem_id: RosettaEntry(
            semantic_id=sem_id,
            subleq_op=entry["subleq_op"],
            template=entry["template"],
            args=tuple(entry.get("args", [])),
        )
        for sem_id, entry in data.get("mappings", {}).items()
    }


_TABLE: dict[str, RosettaEntry] = {}


def table() -> dict[str, RosettaEntry]:
    global _TABLE
    if not _TABLE:
        _TABLE = _load_table()
    return _TABLE


# ── Mapper ────────────────────────────────────────────────────────────────────

class RosettaMapper:
    """Map LatinCanonicalForm tokens to SUBLEQ operations via the Rosetta table."""

    def map(self, form: LatinCanonicalForm) -> RosettaMapping:
        tbl = table()
        mapping = RosettaMapping()

        for tok in form.tokens:
            entry = tbl.get(tok.semantic_id)
            if entry is None:
                mapping.unmapped.append(tok)
            else:
                mapping.entries.append((tok, entry))

        if not mapping.complete:
            unmapped_ids = [t.semantic_id for t in mapping.unmapped]
            raise RosettaError(
                FSMState.ROSETTA_MAPPING,
                f"Unmapped semantic IDs: {unmapped_ids}",
            )

        return mapping

    def map_lex_result(self, result: LexResult, form: LatinCanonicalForm) -> RosettaMapping:
        """Map from a lexed result, cross-referencing the canonical form for semantic IDs."""
        return self.map(form)
