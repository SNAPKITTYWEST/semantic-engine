# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phase 2 — Semantic IR."""

import pytest
from semantic_engine.ir import (
    Geometry, TemporalState, TemporalLock, TemporalTransition,
    Operand, Constraint, SemanticOperation, SemanticNode,
    SemanticIR, Artifact,
)
from semantic_engine.types import (
    ArtifactType, FSMState, IntentClass, Provenance,
    SemanticNodeType, TemporalViolation, VerificationState,
)


def make_provenance():
    return Provenance("input-1", "test")


def make_temporal():
    return TemporalState(tau=0)


def test_geometry_matmul_compatible():
    a = Geometry(row=3, col=4)
    b = Geometry(row=4, col=5)
    assert a.compatible_with(b)
    out = a.output_geometry(b)
    assert out.row == 3
    assert out.col == 5


def test_geometry_matmul_incompatible():
    a = Geometry(row=3, col=4)
    b = Geometry(row=5, col=5)
    assert not a.compatible_with(b)


def test_temporal_advance():
    t = TemporalState(tau=0)
    t1 = t.advance("step1")
    assert t1.tau == 1
    assert t.tau == 0  # original unchanged


def test_temporal_lock_prevents_advance():
    t = TemporalState(tau=0, locked=True)
    with pytest.raises(TemporalViolation):
        t.advance("should fail")


def test_temporal_lock_records_transitions():
    lock = TemporalLock()
    t = TemporalTransition(from_tau=0, to_tau=1, reason="step", stage="test")
    lock.record(t)
    assert len(lock.audit()) == 1


def test_temporal_lock_rejects_invalid_jump():
    lock = TemporalLock()
    bad = TemporalTransition(from_tau=0, to_tau=5, reason="jump", stage="test")
    with pytest.raises(TemporalViolation):
        lock.record(bad)


def test_semantic_node_stable_id():
    p = make_provenance()
    t = make_temporal()
    op = SemanticOperation("build", 1)
    n1 = SemanticNode.make(SemanticNodeType.OPERATION, op, [], [], None, t, p)
    n2 = SemanticNode.make(SemanticNodeType.OPERATION, op, [], [], None, t, p)
    assert n1.id == n2.id


def test_semantic_ir_create():
    p = make_provenance()
    t = make_temporal()
    op = SemanticOperation("build", 0)
    node = SemanticNode.make(SemanticNodeType.INTENT, op, [], [], None, t, p)
    ir = SemanticIR.create(IntentClass.BUILD, node, [node], TemporalLock(), p)
    assert ir.ir_id
    assert ir.intent == IntentClass.BUILD
    assert ir.verification_state == VerificationState.UNVERIFIED


def test_semantic_ir_find_node():
    p = make_provenance()
    t = make_temporal()
    op = SemanticOperation("build", 0)
    node = SemanticNode.make(SemanticNodeType.INTENT, op, [], [], None, t, p)
    ir = SemanticIR.create(IntentClass.BUILD, node, [node], TemporalLock(), p)
    found = ir.find_node(node.id)
    assert found is node
    assert ir.find_node("nonexistent") is None


def test_artifact_create():
    p = make_provenance()
    t = make_temporal()
    a = Artifact.create(ArtifactType.DOCUMENT, "root-id", "tmpl", "content", p, t)
    assert a.type == ArtifactType.DOCUMENT
    assert a.content == "content"
    assert a.semantic_root == "root-id"
