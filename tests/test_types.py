# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phase 1 — Core types."""

import pytest
from semantic_engine.types import (
    RawInput, NormalizedInput, SemanticCandidate,
    FSMState, VerificationState, ArtifactType, BooleanOperator,
    IntentClass, Provenance, FSMTransition, PipelineError,
)


def test_raw_input_immutable():
    r = RawInput("hello world")
    with pytest.raises(AttributeError):
        r.text = "changed"


def test_raw_input_hash_deterministic():
    r1 = RawInput("hello")
    r2 = RawInput("hello")
    assert r1.hash == r2.hash


def test_raw_input_unique_id():
    r1 = RawInput("hello")
    r2 = RawInput("hello")
    assert r1.input_id != r2.input_id


def test_semantic_candidate_confidence_range():
    with pytest.raises(ValueError):
        SemanticCandidate("id", IntentClass.BUILD, [], [], [], [], 1.5, False)
    with pytest.raises(ValueError):
        SemanticCandidate("id", IntentClass.BUILD, [], [], [], [], -0.1, False)


def test_semantic_candidate_valid():
    c = SemanticCandidate("id", IntentClass.BUILD, ["matrix"], ["build"], [], [], 0.8, False)
    assert c.confidence == 0.8


def test_fsm_states_unique():
    values = [s.value for s in FSMState]
    assert len(values) == len(set(values))


def test_provenance_chain_hash():
    p = Provenance("input-1", "test", transform_hash="abc")
    h1 = p.chain_hash("data")
    h2 = p.chain_hash("data")
    assert h1 == h2


def test_fsm_transition_validate():
    t = FSMTransition(
        FSMState.INPUT, "raw_input", "normalize",
        ("text_not_empty",), FSMState.NORMALIZE, "input preserved",
    )
    assert t.validate(FSMState.INPUT, "raw_input")
    assert not t.validate(FSMState.COMPLETE, "raw_input")
    assert not t.validate(FSMState.INPUT, "wrong_class")


def test_pipeline_error_message():
    e = PipelineError(FSMState.TOKENIZATION, "token broken")
    assert "TOKENIZATION" in str(e)
    assert "token broken" in str(e)
