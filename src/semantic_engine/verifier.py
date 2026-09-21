# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 14: Verification engine — all checks before artifact emission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .boolean_engine import BooleanNormalForm, normalize
from .ir import Artifact, SemanticIR
from .latin import LatinCanonicalForm, vocabulary
from .rosetta import RosettaMapping
from .subleq import SubleqProgram, SubleqVM
from .token_engine import BooleanTokenStream, round_trip
from .types import FSMState, VerificationError, VerificationState


# ── Check result ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"{status} {self.name}: {self.message}"


@dataclass
class VerificationReport:
    checks: list[CheckResult] = field(default_factory=list)
    state: VerificationState = VerificationState.UNVERIFIED

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def summary(self) -> str:
        lines = [repr(c) for c in self.checks]
        lines.append(f"\nResult: {'VERIFIED' if self.passed else 'REJECTED'}")
        return "\n".join(lines)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_type_correctness(ir: SemanticIR) -> CheckResult:
    for node in ir.nodes:
        if node.operation and node.operation.arity != len(node.operands):
            if node.operation.arity > 0:
                return CheckResult(
                    "type_correctness", False,
                    f"Node {node.id}: arity {node.operation.arity} but {len(node.operands)} operands",
                )
    return CheckResult("type_correctness", True, "all arities match")


def check_geometry(ir: SemanticIR) -> CheckResult:
    for node in ir.nodes:
        if node.geometry and node.operation and node.operation.name == "multiply":
            if len(node.operands) >= 2:
                a = node.operands[0].geometry
                b = node.operands[1].geometry
                if a and b and a.col is not None and b.row is not None:
                    if a.col != b.row:
                        return CheckResult(
                            "geometry", False,
                            f"Matrix dim mismatch: A.col={a.col} ≠ B.row={b.row}",
                        )
    return CheckResult("geometry", True, "all dimensions compatible")


def check_temporal_state(ir: SemanticIR) -> CheckResult:
    transitions = ir.temporal_lock.audit()
    for t in transitions:
        if not t.is_valid():
            return CheckResult(
                "temporal_state", False,
                f"Invalid τ transition: {t.from_tau} → {t.to_tau}",
            )
    return CheckResult("temporal_state", True, f"{len(transitions)} temporal transitions valid")


def check_boolean_normalization(bnf: BooleanNormalForm) -> CheckResult:
    if not bnf.stable:
        bnf.verify_stability()
    if not bnf.stable:
        return CheckResult("boolean_normalization", False, "normalize(normalize(x)) ≠ normalize(x)")
    return CheckResult("boolean_normalization", True, "normalization idempotent")


def check_token_round_trip(stream: BooleanTokenStream) -> CheckResult:
    try:
        from .token_engine import decode_binary, encode_binary
        binary = encode_binary(stream)
        decoded = decode_binary(binary)
        if len(decoded.tokens) != len(stream.tokens):
            return CheckResult("token_round_trip", False,
                               f"Token count mismatch: {len(stream.tokens)} → {len(decoded.tokens)}")
        for orig, dec in zip(stream.tokens, decoded.tokens):
            if orig.token_type != dec.token_type or orig.payload != dec.payload:
                return CheckResult("token_round_trip", False,
                                   f"Token mismatch at pos {orig.position}")
        return CheckResult("token_round_trip", True, "encode/decode invariant holds")
    except Exception as e:
        return CheckResult("token_round_trip", False, str(e))


def check_latin_vocabulary(form: LatinCanonicalForm) -> CheckResult:
    vocab = vocabulary()
    for tok in form.tokens:
        if tok.lexeme not in vocab and tok.lexeme not in ("OPUS",):
            return CheckResult(
                "latin_vocabulary", False,
                f"Lexeme '{tok.lexeme}' not in controlled vocabulary",
            )
    return CheckResult("latin_vocabulary", True, "all lexemes in vocabulary")


def check_rosetta_mapping(mapping: RosettaMapping) -> CheckResult:
    if not mapping.complete:
        unmapped = [t.semantic_id for t in mapping.unmapped]
        return CheckResult("rosetta_mapping", False, f"Unmapped: {unmapped}")
    return CheckResult("rosetta_mapping", True, f"{len(mapping.entries)} entries mapped")


def check_subleq_validity(prog: SubleqProgram) -> CheckResult:
    if not prog.instructions:
        return CheckResult("subleq_validity", False, "Empty SUBLEQ program")
    vm = SubleqVM(max_steps=10_000)
    halts = vm.verify_halts(prog)
    if not halts:
        return CheckResult("subleq_validity", False, "Program does not halt within limit")
    return CheckResult("subleq_validity", True, f"{len(prog.instructions)} instructions, halts")


def check_artifact_provenance(artifact: Artifact, ir: SemanticIR) -> CheckResult:
    if artifact.semantic_root != ir.ir_id:
        return CheckResult(
            "artifact_provenance", False,
            f"Artifact root {artifact.semantic_root} ≠ IR id {ir.ir_id}",
        )
    return CheckResult("artifact_provenance", True, "provenance chain intact")


def check_template_bindings(artifact: Artifact) -> CheckResult:
    if not artifact.template:
        return CheckResult("template_bindings", False, "No template bound to artifact")
    if not artifact.content:
        return CheckResult("template_bindings", False, "Artifact content is empty")
    return CheckResult("template_bindings", True, "template bound and content non-empty")


# ── Verification engine ───────────────────────────────────────────────────────

class Verifier:
    def verify_ir(self, ir: SemanticIR) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_type_correctness(ir))
        report.add(check_geometry(ir))
        report.add(check_temporal_state(ir))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def verify_boolean(self, bnf: BooleanNormalForm) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_boolean_normalization(bnf))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def verify_tokens(self, stream: BooleanTokenStream) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_token_round_trip(stream))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def verify_latin(self, form: LatinCanonicalForm) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_latin_vocabulary(form))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def verify_rosetta(self, mapping: RosettaMapping) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_rosetta_mapping(mapping))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def verify_subleq(self, prog: SubleqProgram) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_subleq_validity(prog))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def verify_artifact(self, artifact: Artifact, ir: SemanticIR) -> VerificationReport:
        report = VerificationReport(state=VerificationState.VALIDATING)
        report.add(check_artifact_provenance(artifact, ir))
        report.add(check_template_bindings(artifact))
        report.state = VerificationState.VERIFIED if report.passed else VerificationState.REJECTED
        return report

    def reject_if_failed(self, report: VerificationReport, stage: FSMState) -> None:
        if not report.passed:
            raise VerificationError(stage, "Verification failed:\n" + report.summary())
