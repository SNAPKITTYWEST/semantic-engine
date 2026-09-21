# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 2: Semantic IR — typed intermediate representation, the canonical source of truth."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import (
    ArtifactType, IntentClass, Provenance, SemanticNodeType,
    TemporalViolation, VerificationState, FSMState,
)


# ── Geometry ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Geometry:
    """Explicit dimensional description of an operation's operands."""
    row: Optional[int] = None
    col: Optional[int] = None
    contraction: Optional[int] = None
    shape: tuple[int, ...] = field(default_factory=tuple)
    index_labels: tuple[str, ...] = field(default_factory=tuple)

    def compatible_with(self, other: Geometry) -> bool:
        """Check whether self and other can be contracted (matrix multiply)."""
        if self.col is None or other.row is None:
            return True  # undetermined — not a violation yet
        return self.col == other.row

    def output_geometry(self, other: Geometry) -> Geometry:
        if self.row is None or other.col is None:
            return Geometry()
        return Geometry(row=self.row, col=other.col, contraction=self.col)


# ── Temporal state ────────────────────────────────────────────────────────────

@dataclass
class TemporalState:
    tau: int  # monotonic integer tick
    label: str = ""
    locked: bool = False

    def advance(self, reason: str = "") -> TemporalState:
        if self.locked:
            raise TemporalViolation(FSMState.TEMPORAL_LOCK, "Cannot advance a locked temporal state")
        return TemporalState(tau=self.tau + 1, label=reason)

    def lock(self) -> TemporalState:
        return TemporalState(tau=self.tau, label=self.label, locked=True)


@dataclass(frozen=True)
class TemporalTransition:
    from_tau: int
    to_tau: int
    reason: str
    stage: str

    def is_valid(self) -> bool:
        return self.to_tau == self.from_tau + 1 or self.to_tau == self.from_tau


@dataclass
class TemporalLock:
    transitions: list[TemporalTransition] = field(default_factory=list)

    def record(self, t: TemporalTransition) -> None:
        if not t.is_valid():
            raise TemporalViolation(
                FSMState.TEMPORAL_LOCK,
                f"Invalid temporal jump: τ{t.from_tau} → τ{t.to_tau} (expected +1 or 0)",
            )
        self.transitions.append(t)

    def audit(self) -> list[TemporalTransition]:
        return list(self.transitions)


# ── Core IR nodes ─────────────────────────────────────────────────────────────

@dataclass
class Operand:
    name: str
    type_tag: str
    value: Any = None
    geometry: Optional[Geometry] = None

    @property
    def node_id(self) -> str:
        return hashlib.sha256(f"operand:{self.name}:{self.type_tag}".encode()).hexdigest()[:16]


@dataclass
class Constraint:
    constraint_id: str
    description: str
    predicate: str
    operands: list[str] = field(default_factory=list)

    @staticmethod
    def make(description: str, predicate: str, operands: list[str] | None = None) -> Constraint:
        cid = hashlib.sha256(f"{predicate}:{description}".encode()).hexdigest()[:16]
        return Constraint(
            constraint_id=cid,
            description=description,
            predicate=predicate,
            operands=operands or [],
        )


@dataclass
class SemanticOperation:
    name: str
    arity: int
    description: str = ""
    associative: bool = False
    commutative: bool = False

    @property
    def op_id(self) -> str:
        return hashlib.sha256(f"op:{self.name}:{self.arity}".encode()).hexdigest()[:16]


@dataclass
class SemanticNode:
    id: str
    type: SemanticNodeType
    operation: Optional[SemanticOperation]
    operands: list[Operand]
    constraints: list[Constraint]
    geometry: Optional[Geometry]
    temporal_state: TemporalState
    provenance: Provenance
    children: list[SemanticNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(
        node_type: SemanticNodeType,
        operation: Optional[SemanticOperation],
        operands: list[Operand],
        constraints: list[Constraint],
        geometry: Optional[Geometry],
        temporal_state: TemporalState,
        provenance: Provenance,
    ) -> SemanticNode:
        content = f"{node_type}:{operation}:{[o.name for o in operands]}"
        node_id = hashlib.sha256(content.encode()).hexdigest()[:24]
        return SemanticNode(
            id=node_id,
            type=node_type,
            operation=operation,
            operands=operands,
            constraints=constraints,
            geometry=geometry,
            temporal_state=temporal_state,
            provenance=provenance,
        )

    def hash(self) -> str:
        content = f"{self.id}:{self.type}:{self.operation}:{len(self.operands)}"
        return hashlib.sha256(content.encode()).hexdigest()


# ── Semantic IR root ──────────────────────────────────────────────────────────

@dataclass
class SemanticIR:
    ir_id: str
    intent: IntentClass
    root: SemanticNode
    nodes: list[SemanticNode]
    temporal_lock: TemporalLock
    provenance: Provenance
    verification_state: VerificationState = VerificationState.UNVERIFIED
    output_mode: str = "natural_language"
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        intent: IntentClass,
        root: SemanticNode,
        nodes: list[SemanticNode],
        temporal_lock: TemporalLock,
        provenance: Provenance,
    ) -> SemanticIR:
        ir_id = str(uuid.uuid4())
        return SemanticIR(
            ir_id=ir_id,
            intent=intent,
            root=root,
            nodes=nodes,
            temporal_lock=temporal_lock,
            provenance=provenance,
        )

    def hash(self) -> str:
        node_hashes = ":".join(n.hash() for n in self.nodes)
        return hashlib.sha256(f"{self.ir_id}:{self.intent}:{node_hashes}".encode()).hexdigest()

    def add_node(self, node: SemanticNode) -> None:
        self.nodes.append(node)

    def find_node(self, node_id: str) -> Optional[SemanticNode]:
        return next((n for n in self.nodes if n.id == node_id), None)


# ── Artifact IR ───────────────────────────────────────────────────────────────

@dataclass
class Artifact:
    id: str
    type: ArtifactType
    semantic_root: str  # SemanticIR.ir_id
    template: str
    content: str
    provenance: Provenance
    temporal_state: TemporalState
    verification_state: VerificationState = VerificationState.UNVERIFIED
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        artifact_type: ArtifactType,
        semantic_root: str,
        template: str,
        content: str,
        provenance: Provenance,
        temporal_state: TemporalState,
    ) -> Artifact:
        artifact_id = hashlib.sha256(
            f"{artifact_type}:{semantic_root}:{content[:64]}".encode()
        ).hexdigest()[:24]
        return Artifact(
            id=artifact_id,
            type=artifact_type,
            semantic_root=semantic_root,
            template=template,
            content=content,
            provenance=provenance,
            temporal_state=temporal_state,
        )

    def hash(self) -> str:
        return hashlib.sha256(
            f"{self.id}:{self.type}:{self.content}".encode()
        ).hexdigest()
