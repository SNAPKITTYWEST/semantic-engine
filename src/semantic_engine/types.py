# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 1: Core types — all canonical IR types, enums, and base structures."""

from __future__ import annotations

import hashlib
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ── Enums ──────────────────────────────────────────────────────────────────────

class FSMState(Enum):
    INPUT               = auto()
    NORMALIZE           = auto()
    LEXICAL_ANALYSIS    = auto()
    INTENT_DETECTION    = auto()
    ENTITY_BINDING      = auto()
    SEMANTIC_BINDING    = auto()
    TEMPORAL_LOCK       = auto()
    GEOMETRY_BINDING    = auto()
    MATRIX_FORM         = auto()
    BOOLEAN_ROOT        = auto()
    BOOLEAN_NORMALIZATION = auto()
    TOKENIZATION        = auto()
    LATIN_CANONICALIZATION = auto()
    LATIN_LEXING        = auto()
    ROSETTA_MAPPING     = auto()
    SUBLEQ_IR           = auto()
    VERIFICATION        = auto()
    ARTIFACT_SELECTION  = auto()
    EMISSION            = auto()
    COMPLETE            = auto()
    ERROR               = auto()


class VerificationState(Enum):
    UNVERIFIED  = auto()
    VALIDATING  = auto()
    VERIFIED    = auto()
    REJECTED    = auto()


class ArtifactType(Enum):
    PROMPT      = "prompt"
    DOCUMENT    = "document"
    SOURCE      = "source"
    CONFIG      = "config"
    TEST        = "test"
    SCHEMA      = "schema"
    REPORT      = "report"
    BACKEND     = "backend"


class SemanticNodeType(Enum):
    OPERATION   = "operation"
    ENTITY      = "entity"
    CONSTRAINT  = "constraint"
    RELATION    = "relation"
    INTENT      = "intent"
    GEOMETRY    = "geometry"
    TEMPORAL    = "temporal"
    LITERAL     = "literal"


class BooleanOperator(Enum):
    AND         = "AND"
    OR          = "OR"
    NOT         = "NOT"
    XOR         = "XOR"
    IMPLIES     = "IMPLIES"
    EQUIVALENCE = "EQUIVALENCE"
    TRUE        = "TRUE"
    FALSE       = "FALSE"


class IntentClass(Enum):
    BUILD       = "build"
    TRANSFORM   = "transform"
    VALIDATE    = "validate"
    GENERATE    = "generate"
    PARSE       = "parse"
    COMPUTE     = "compute"
    RECONSTRUCT = "reconstruct"
    UNKNOWN     = "unknown"


class OutputMode(Enum):
    NATURAL_LANGUAGE = "natural_language"
    CODE             = "code"
    ARTIFACT         = "artifact"


# ── Input types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RawInput:
    text: str
    source: str = "stdin"
    timestamp: float = field(default_factory=time.time)
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class NormalizedInput:
    original: RawInput
    text: str
    tokens: tuple[str, ...]
    sentences: tuple[str, ...]

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass
class SemanticCandidate:
    input_ref: str
    intent: IntentClass
    entities: list[str]
    operators: list[str]
    constraints: list[str]
    relationships: list[tuple[str, str, str]]
    confidence: float
    ambiguous: bool
    raw_embeddings: Optional[list[float]] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")


# ── Provenance ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Provenance:
    source_input_id: str
    stage: str
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    transform_hash: str = ""

    def chain_hash(self, data: str) -> str:
        return hashlib.sha256(f"{self.transform_hash}:{data}".encode()).hexdigest()


# ── FSM Transition record ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FSMTransition:
    source_state: FSMState
    input_class: str
    operation: str
    constraints: tuple[str, ...]
    destination_state: FSMState
    invariant: str

    def validate(self, current_state: FSMState, input_class: str) -> bool:
        return self.source_state == current_state and self.input_class == input_class


# ── Pipeline error ────────────────────────────────────────────────────────────

class PipelineError(Exception):
    def __init__(self, stage: FSMState, message: str, input_ref: str = ""):
        self.stage = stage
        self.input_ref = input_ref
        super().__init__(f"[{stage.name}] {message}")


class GeometryError(PipelineError):
    pass


class TemporalViolation(PipelineError):
    pass


class VerificationError(PipelineError):
    pass


class TokenError(PipelineError):
    pass


class RosettaError(PipelineError):
    pass
