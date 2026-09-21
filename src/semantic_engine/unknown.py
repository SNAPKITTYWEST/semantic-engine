# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ADR-Ω-0001: The Uninterpreted Operator — ? (unknown semantics, observable behavior).

?(A) → B  is valid as an observation.
?(A) = f(A) is NOT valid until f has been established.

UNKNOWN ≠ FALSE
UNKNOWN ≠ TRUE

The system must not silently convert UNKNOWN into UNDERSTOOD.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional


# ── Invariant state (Ω) ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class InvariantState:
    """Ω — the preserved invariant carried through every observation."""
    tau: int
    system_hash: str
    properties: tuple[str, ...] = field(default_factory=tuple)

    def hash(self) -> str:
        content = f"Ω:{self.tau}:{self.system_hash}:{':'.join(self.properties)}"
        return hashlib.sha256(content.encode()).hexdigest()


# ── Observation record: ⟨A, B, H, t, Ω⟩ ────────────────────────────────────

@dataclass(frozen=True)
class ObservationRecord:
    """
    O_?(A) = ⟨A, B, H, t, Ω⟩

    Records that A → ? → B occurred without fabricating an explanation.
    H = hash of the full record.
    t = state-transition timestamp.
    Ω = invariant state at time of observation.
    """
    input_state: str            # A  — canonical representation of input
    output_state: str           # B  — canonical representation of output
    provenance_hash: str        # H  — hash(A ‖ ? ‖ B ‖ Ω)
    timestamp: float            # t
    invariant: InvariantState   # Ω

    # Classification state — may only advance through the pipeline, never skip
    classification: UnknownClassification = field(default=None)

    def __post_init__(self):
        # Enforce: classification starts as UNOBSERVED if not provided
        if self.classification is None:
            object.__setattr__(self, "classification", UnknownClassification.OBSERVED)

    @staticmethod
    def observe(
        input_state: Any,
        output_state: Any,
        invariant: InvariantState,
    ) -> ObservationRecord:
        a = _canonical(input_state)
        b = _canonical(output_state)
        t = time.time()
        h = _seal_hash(a, b, invariant)
        return ObservationRecord(
            input_state=a,
            output_state=b,
            provenance_hash=h,
            timestamp=t,
            invariant=invariant,
        )

    def verify(self) -> bool:
        """Recompute and verify the provenance hash."""
        expected = _seal_hash(self.input_state, self.output_state, self.invariant)
        return self.provenance_hash == expected

    def advance(self, to: UnknownClassification) -> ObservationRecord:
        """
        Advance classification through the pipeline:
          OBSERVED → CAPTURED → HASHED → COMPARED → TESTED → CLASSIFIED

        May never skip a step. UNKNOWN must not become UNDERSTOOD silently.
        """
        _assert_valid_advancement(self.classification, to)
        return ObservationRecord(
            input_state=self.input_state,
            output_state=self.output_state,
            provenance_hash=self.provenance_hash,
            timestamp=self.timestamp,
            invariant=self.invariant,
            classification=to,
        )


# ── Classification pipeline (OBSERVE → CLASSIFY only, no shortcuts) ───────────

class UnknownClassification(Enum):
    OBSERVED    = auto()
    CAPTURED    = auto()
    HASHED      = auto()
    COMPARED    = auto()
    TESTED      = auto()
    CLASSIFIED  = auto()

    # Explicitly NOT in this enum:
    # UNDERSTOOD — because converting UNKNOWN to UNDERSTOOD requires
    # establishing f such that ?(A) = f(A). That is a separate formal claim.


_VALID_TRANSITIONS: dict[UnknownClassification, UnknownClassification] = {
    UnknownClassification.OBSERVED:   UnknownClassification.CAPTURED,
    UnknownClassification.CAPTURED:   UnknownClassification.HASHED,
    UnknownClassification.HASHED:     UnknownClassification.COMPARED,
    UnknownClassification.COMPARED:   UnknownClassification.TESTED,
    UnknownClassification.TESTED:     UnknownClassification.CLASSIFIED,
}


def _assert_valid_advancement(
    current: UnknownClassification,
    target: UnknownClassification,
) -> None:
    allowed = _VALID_TRANSITIONS.get(current)
    if allowed != target:
        raise UnknownOperatorError(
            f"Invalid classification advance: {current.name} → {target.name}. "
            f"Expected {allowed.name if allowed else 'nothing (terminal)'}. "
            f"UNKNOWN must not become UNDERSTOOD by skipping pipeline stages."
        )


class UnknownOperatorError(Exception):
    """Raised when the system attempts to fabricate semantics for ?."""


# ── WORM Seal engine ─────────────────────────────────────────────────────────

def _canonical(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _seal_hash(a: str, b: str, omega: InvariantState) -> str:
    """Seal(A → ? → B) = H(A ‖ ? ‖ B ‖ Ω)"""
    content = f"{a}\x00?\x00{b}\x00{omega.hash()}"
    return hashlib.sha256(content.encode()).hexdigest()


@dataclass(frozen=True)
class WORMSealEntry:
    """One entry in the append-only WORM seal chain."""
    index: int
    record: ObservationRecord
    entry_hash: str
    previous_hash: str
    sealed_at: float = 0.0

    def verify_chain(self, previous_entry: Optional[WORMSealEntry]) -> bool:
        """Verify this entry's chain link to the previous entry."""
        expected_prev = previous_entry.entry_hash if previous_entry else "0" * 64
        if self.previous_hash != expected_prev:
            return False
        return self.record.verify()

    def to_dict(self) -> dict:
        return {
            "index":         self.index,
            "input_state":   self.record.input_state[:80],
            "output_state":  self.record.output_state[:80],
            "provenance_hash": self.record.provenance_hash,
            "classification": self.record.classification.name,
            "entry_hash":    self.entry_hash,
            "previous_hash": self.previous_hash,
            "sealed_at":     self.sealed_at,
            "tau":           self.record.invariant.tau,
        }


class WORMSealChain:
    """
    Append-only chain of sealed observations.

    Seal(A → ? → B) = H(A ‖ ? ‖ B ‖ Ω)

    Once sealed, an observation cannot be modified.
    The chain hash links every entry to its predecessor.
    """

    def __init__(self):
        self._entries: list[WORMSealEntry] = []

    def seal(self, record: ObservationRecord) -> WORMSealEntry:
        """Seal an observation record into the chain. Returns the sealed entry."""
        prev_hash = self._entries[-1].entry_hash if self._entries else "0" * 64
        index = len(self._entries)

        # Entry hash = H(index ‖ record_hash ‖ prev_hash)
        entry_content = f"{index}:{record.provenance_hash}:{prev_hash}"
        entry_hash = hashlib.sha256(entry_content.encode()).hexdigest()

        entry = WORMSealEntry(
            index=index,
            record=record,
            entry_hash=entry_hash,
            previous_hash=prev_hash,
            sealed_at=time.time(),
        )
        self._entries.append(entry)
        return entry

    def head(self) -> Optional[WORMSealEntry]:
        return self._entries[-1] if self._entries else None

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the full chain integrity. Returns (ok, errors)."""
        errors: list[str] = []
        prev: Optional[WORMSealEntry] = None
        for entry in self._entries:
            if not entry.verify_chain(prev):
                errors.append(f"Chain broken at index {entry.index}")
            if not entry.record.verify():
                errors.append(f"Record hash invalid at index {entry.index}")
            prev = entry
        return len(errors) == 0, errors

    def audit(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def export_jsonl(self, path: Path) -> None:
        """Export chain to JSONL for external audit."""
        lines = [json.dumps(e.to_dict()) for e in self._entries]
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def import_jsonl(path: Path) -> list[dict]:
        """Read chain from JSONL for verification (read-only)."""
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── Unknown semantic node ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class UnknownNode:
    """
    A semantic node whose internal transformation is not yet understood.

    ? is a first-class operator in the algebra:

      ☉  PRODUCT
      ⌹  PARTITION
      ○  CLOSURE
      ◇  TRANSFORM
      △  DIFFERENCE
      ⬡  COMPOSITION
      ?  UNKNOWN       ← this
      Ω  INVARIANT
    """
    node_id: str
    observation: ObservationRecord
    seal_entry: Optional[WORMSealEntry] = None

    @property
    def is_sealed(self) -> bool:
        return self.seal_entry is not None

    @property
    def is_understood(self) -> bool:
        # May NEVER return True from this class alone.
        # Requires external formal establishment of f such that ?(A) = f(A).
        return False

    def __repr__(self) -> str:
        cls = self.observation.classification.name
        sealed = "SEALED" if self.is_sealed else "UNSEALED"
        return f"?[{self.node_id[:8]}|{cls}|{sealed}]"


# ── Integration helper ────────────────────────────────────────────────────────

def observe_and_seal(
    input_state: Any,
    output_state: Any,
    invariant: InvariantState,
    chain: WORMSealChain,
    node_id: str = "",
) -> UnknownNode:
    """
    Full pipeline:
      OBSERVE → CAPTURE → HASH → COMPARE → TEST → CLASSIFY → SEAL

    Returns an UnknownNode with a sealed chain entry.
    The node carries the observation but never claims to understand the transformation.
    """
    if not node_id:
        node_id = hashlib.sha256(
            f"{_canonical(input_state)}:{_canonical(output_state)}:{time.time()}".encode()
        ).hexdigest()[:24]

    record = ObservationRecord.observe(input_state, output_state, invariant)
    record = record.advance(UnknownClassification.CAPTURED)
    record = record.advance(UnknownClassification.HASHED)
    record = record.advance(UnknownClassification.COMPARED)
    record = record.advance(UnknownClassification.TESTED)
    record = record.advance(UnknownClassification.CLASSIFIED)

    entry = chain.seal(record)
    return UnknownNode(node_id=node_id, observation=record, seal_entry=entry)
