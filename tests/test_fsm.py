# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phase 3 — Decoder FSM."""

import pytest
from semantic_engine.fsm import FSMContext, TRANSITION_TABLE, validate_constraints
from semantic_engine.types import FSMState, PipelineError


def test_initial_state():
    ctx = FSMContext()
    assert ctx.state == FSMState.INPUT


def test_valid_transition():
    ctx = FSMContext()
    ctx.set("raw_text", "hello world")
    t = ctx.transition("raw_input")
    assert ctx.state == FSMState.NORMALIZE
    assert t.source_state == FSMState.INPUT


def test_invalid_transition_fails_closed():
    ctx = FSMContext()
    with pytest.raises(PipelineError) as exc_info:
        ctx.transition("invalid_class_xyz")
    assert "fail" in str(exc_info.value).lower() or "transition" in str(exc_info.value).lower()


def test_transition_history_recorded():
    ctx = FSMContext()
    ctx.set("raw_text", "hello")
    ctx.transition("raw_input")
    assert len(ctx.history) == 1
    src, inp, dst = ctx.history[0]
    assert src == FSMState.INPUT
    assert dst == FSMState.NORMALIZE


def test_all_transitions_explicit():
    """Every transition in the table must have source, input_class, destination."""
    for t in TRANSITION_TABLE:
        assert t.source_state is not None
        assert t.input_class
        assert t.destination_state is not None
        assert t.invariant


def test_transition_table_no_duplicate_keys():
    keys = [(t.source_state, t.input_class) for t in TRANSITION_TABLE]
    assert len(keys) == len(set(keys)), "Duplicate transition keys found"


def test_is_terminal():
    ctx = FSMContext()
    assert not ctx.is_terminal()
    ctx.state = FSMState.COMPLETE
    assert ctx.is_terminal()
    ctx.state = FSMState.ERROR
    assert ctx.is_terminal()


def test_context_set_get():
    ctx = FSMContext()
    ctx.set("foo", 42)
    assert ctx.get("foo") == 42
    assert ctx.get("missing") is None
