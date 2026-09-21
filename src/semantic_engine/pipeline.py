# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Full pipeline — orchestrates all stages from RawInput to Artifact."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from .types import (
    ArtifactType, FSMState, IntentClass, OutputMode,
    PipelineError, Provenance, RawInput, SemanticNodeType, VerificationState,
)
from .ir import (
    Artifact, Constraint, Geometry, Operand, SemanticIR, SemanticNode,
    SemanticOperation, TemporalLock, TemporalState,
)
from .nlp import BertDecoder, normalize_input
from .fsm import FSMContext, validate_constraints
from .temporal import apply_temporal, stamp
from .geometry import enforce_geometry
from .matrix import SemanticMatrixBuilder
from .boolean_engine import semantic_constraints_to_boolean, normalize as bool_normalize
from .token_engine import BoolEncoder, encode_binary
from .latin import LatinCanonicalizer
from .lexer import LatinLexer
from .rosetta import RosettaMapper
from .subleq import build_program_from_rosetta, SubleqVM
from .verifier import Verifier
from .artifacts import ArtifactEngine


@dataclass
class PipelineResult:
    ir: Optional[SemanticIR] = None
    artifact: Optional[Artifact] = None
    subleq_listing: str = ""
    trace: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    verification_state: VerificationState = VerificationState.UNVERIFIED
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.verification_state == VerificationState.VERIFIED


class SemanticPipeline:
    def __init__(self, bert_model: Optional[str] = None, output_mode: OutputMode = OutputMode.NATURAL_LANGUAGE):
        self._bert    = BertDecoder(bert_model)
        self._matrix  = SemanticMatrixBuilder()
        self._latin   = LatinCanonicalizer()
        self._lexer   = LatinLexer()
        self._rosetta = RosettaMapper()
        self._verifier = Verifier()
        self._artifacts = ArtifactEngine()
        self._output_mode = output_mode

    def run(self, raw: RawInput, output_mode: Optional[OutputMode] = None) -> PipelineResult:
        mode = output_mode or self._output_mode
        ctx = FSMContext()
        result = PipelineResult()
        result.trace.append(f"INPUT: {raw.input_id}")
        result.hashes["input"] = raw.hash

        try:
            # PHASE: INPUT → NORMALIZE
            ctx.transition("raw_input")
            normalized = normalize_input(raw)
            ctx.set("raw_text", normalized.text)
            ctx.set("tokens", normalized.tokens)
            result.trace.append(f"NORMALIZE: {len(normalized.tokens)} tokens")

            # PHASE: NORMALIZE → LEXICAL_ANALYSIS
            ctx.transition("normalized_input")
            result.trace.append(f"LEXICAL_ANALYSIS: {len(normalized.sentences)} sentences")

            # PHASE: LEXICAL_ANALYSIS → INTENT_DETECTION
            ctx.transition("token_stream")
            candidate = self._bert.decode(normalized)
            ctx.set("candidate", candidate)
            ctx.set("intent", candidate.intent.value)
            result.trace.append(f"INTENT_DETECTION: {candidate.intent.value} ({candidate.confidence:.2f})")

            if candidate.ambiguous:
                result.trace.append("WARNING: ambiguous input detected")

            # PHASE: INTENT_DETECTION → ENTITY_BINDING
            ctx.transition("intent_class")
            result.trace.append(f"ENTITY_BINDING: {candidate.entities}")

            # PHASE: ENTITY_BINDING → SEMANTIC_BINDING
            ctx.transition("entities")
            provenance = Provenance(
                source_input_id=raw.input_id,
                stage="semantic_binding",
                transform_hash=raw.hash,
            )
            t0 = TemporalState(tau=0, label="initial")
            lock = TemporalLock()

            # Build semantic nodes from candidate
            operands = [Operand(name=e, type_tag="entity") for e in candidate.entities]
            constraints = [Constraint.make(c, c) for c in candidate.constraints]

            op = None
            if candidate.operators:
                op = SemanticOperation(name=candidate.operators[0], arity=len(operands))

            node = SemanticNode.make(
                node_type=SemanticNodeType.OPERATION if op else SemanticNodeType.INTENT,
                operation=op,
                operands=operands,
                constraints=constraints,
                geometry=None,
                temporal_state=t0,
                provenance=provenance,
            )

            ir = SemanticIR.create(
                intent=candidate.intent,
                root=node,
                nodes=[node],
                temporal_lock=lock,
                provenance=provenance,
            )
            ctx.set("candidate", candidate)
            result.trace.append(f"SEMANTIC_BINDING: IR {ir.ir_id[:8]}")
            result.hashes["semantic_ir"] = ir.hash()

            # PHASE: SEMANTIC_BINDING → TEMPORAL_LOCK
            ctx.transition("semantic_candidate")
            _, t1 = apply_temporal(lambda: None, t0, lock, "temporal_lock")
            result.trace.append(f"TEMPORAL_LOCK: τ{t0.tau}→τ{t1.tau}")

            # PHASE: TEMPORAL_LOCK → GEOMETRY_BINDING
            ctx.transition("temporal_state")
            for n in ir.nodes:
                geom = enforce_geometry(n)
                n.geometry = geom
            result.trace.append("GEOMETRY_BINDING: dimensions validated")

            # PHASE: GEOMETRY_BINDING → MATRIX_FORM
            ctx.transition("geometry")
            sem_matrix = self._matrix.build(ir.root)
            result.trace.append(f"MATRIX_FORM: {sem_matrix.rows}x{sem_matrix.cols}")

            # PHASE: MATRIX_FORM → BOOLEAN_ROOT
            ctx.transition("semantic_matrix")
            bnf = semantic_constraints_to_boolean(ir.root.constraints)
            result.trace.append(f"BOOLEAN_ROOT: {bnf.normalized}")

            # PHASE: BOOLEAN_ROOT → BOOLEAN_NORMALIZATION
            ctx.transition("boolean_expr")
            norm = bool_normalize(bnf.normalized)
            bnf_report = self._verifier.verify_boolean(bnf)
            self._verifier.reject_if_failed(bnf_report, FSMState.BOOLEAN_NORMALIZATION)
            result.trace.append(f"BOOLEAN_NORMALIZATION: stable={bnf.stable}")

            # PHASE: BOOLEAN_NORMALIZATION → TOKENIZATION
            ctx.transition("normalized_boolean")
            encoder = BoolEncoder()
            token_stream = encoder.encode_expr(bnf.normalized)
            binary = encode_binary(token_stream)
            tok_report = self._verifier.verify_tokens(token_stream)
            self._verifier.reject_if_failed(tok_report, FSMState.TOKENIZATION)
            result.trace.append(f"TOKENIZATION: {len(token_stream)} tokens, {len(binary)} bytes")

            # PHASE: TOKENIZATION → LATIN_CANONICALIZATION
            ctx.transition("token_stream")
            latin_form = self._latin.canonicalize(token_stream)
            lat_report = self._verifier.verify_latin(latin_form)
            self._verifier.reject_if_failed(lat_report, FSMState.LATIN_CANONICALIZATION)
            result.trace.append(f"LATIN_CANONICALIZATION: {latin_form.text()[:60]}")

            # PHASE: LATIN_CANONICALIZATION → LATIN_LEXING
            ctx.transition("latin_form")
            lex_result = self._lexer.lex_canonical(latin_form)
            result.trace.append(f"LATIN_LEXING: {len(lex_result.tokens)} lex tokens")

            # PHASE: LATIN_LEXING → ROSETTA_MAPPING
            ctx.transition("latin_tokens")
            mapping = self._rosetta.map(latin_form)
            ros_report = self._verifier.verify_rosetta(mapping)
            self._verifier.reject_if_failed(ros_report, FSMState.ROSETTA_MAPPING)
            result.trace.append(f"ROSETTA_MAPPING: {len(mapping.entries)} entries")

            # PHASE: ROSETTA_MAPPING → SUBLEQ_IR
            ctx.transition("rosetta_mapping")
            prog = build_program_from_rosetta(mapping)
            sub_report = self._verifier.verify_subleq(prog)
            self._verifier.reject_if_failed(sub_report, FSMState.SUBLEQ_IR)
            result.subleq_listing = prog.disassemble_text()
            result.hashes["subleq"] = hashlib.sha256(prog.assemble()).hexdigest()
            result.trace.append(f"SUBLEQ_IR: {len(prog.instructions)} instructions")

            # PHASE: SUBLEQ_IR → VERIFICATION
            ctx.transition("subleq_program")
            ir_report = self._verifier.verify_ir(ir)
            self._verifier.reject_if_failed(ir_report, FSMState.VERIFICATION)
            ir.verification_state = VerificationState.VERIFIED
            ctx.set("verified", True)
            result.trace.append("VERIFICATION: PASSED")

            # PHASE: VERIFICATION → ARTIFACT_SELECTION
            ctx.transition("verified_ir")
            ctx.set("template", "prompt.mustache")
            result.trace.append(f"ARTIFACT_SELECTION: mode={mode.value}")

            # PHASE: ARTIFACT_SELECTION → EMISSION
            ctx.transition("artifact_type")
            if mode == OutputMode.NATURAL_LANGUAGE:
                artifact = self._artifacts.build_document(ir)
            elif mode == OutputMode.CODE:
                artifact = self._artifacts.build_backend(ir)
            else:
                artifact = self._artifacts.build_prompt(ir)

            art_report = self._verifier.verify_artifact(artifact, ir)
            self._verifier.reject_if_failed(art_report, FSMState.EMISSION)
            artifact.verification_state = VerificationState.VERIFIED
            ctx.set("artifact", artifact)
            result.hashes["artifact"] = artifact.hash()
            result.trace.append(f"EMISSION: artifact {artifact.id[:8]}")

            # PHASE: EMISSION → COMPLETE
            ctx.transition("artifact")

            result.ir = ir
            result.artifact = artifact
            result.verification_state = VerificationState.VERIFIED
            result.trace.append("COMPLETE")

        except PipelineError as e:
            result.error = str(e)
            result.verification_state = VerificationState.REJECTED
            result.trace.append(f"ERROR: {e}")

        return result
