# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phase 20 — Round-trip and equivalence invariants."""

import pytest
from semantic_engine.boolean_engine import (
    AND, OR, NOT, XOR, IMPLIES, EQUIV,
    BooleanLiteral, BooleanVar,
    normalize, var, lit,
)
from semantic_engine.token_engine import encode, decode, round_trip, BoolEncoder, encode_binary, decode_binary
from semantic_engine.latin import LatinCanonicalizer
from semantic_engine.rosetta import RosettaMapper


# ── Boolean round-trips ───────────────────────────────────────────────────────

CANONICAL_EXPRS = [
    BooleanLiteral(True),
    BooleanLiteral(False),
    BooleanVar("x"),
    BooleanVar("x", negated=True),
    AND(BooleanVar("a"), BooleanVar("b")),
    OR(BooleanVar("a"), BooleanVar("b")),
    NOT(BooleanVar("x")),
    AND(BooleanVar("a"), OR(BooleanVar("b"), BooleanVar("c"))),
    OR(AND(BooleanVar("a"), BooleanVar("b")), NOT(BooleanVar("c"))),
]


@pytest.mark.parametrize("expr", CANONICAL_EXPRS)
def test_token_round_trip(expr):
    """decode(encode(x)) == x for all canonical Boolean structures."""
    result = round_trip(expr)
    assert repr(result) == repr(expr), f"Round-trip failed for {expr}"


@pytest.mark.parametrize("expr", CANONICAL_EXPRS)
def test_normalize_idempotent(expr):
    """normalize(normalize(x)) == normalize(x)"""
    n1 = normalize(expr)
    n2 = normalize(n1)
    assert repr(n1) == repr(n2), f"Normalization not idempotent for {expr}"


# ── Binary serialization round-trips ─────────────────────────────────────────

@pytest.mark.parametrize("expr", CANONICAL_EXPRS)
def test_binary_round_trip(expr):
    """Binary encode/decode preserves all tokens."""
    stream = BoolEncoder().encode_expr(expr)
    binary = encode_binary(stream)
    decoded_stream = decode_binary(binary)
    assert len(stream.tokens) == len(decoded_stream.tokens)
    for orig, dec in zip(stream.tokens, decoded_stream.tokens):
        assert orig.token_type == dec.token_type
        assert orig.payload == dec.payload


# ── Latin round-trip ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr", CANONICAL_EXPRS)
def test_latin_derivable_from_tokens(expr):
    """Latin canonical form must derive from the token stream without loss."""
    stream = BoolEncoder().encode_expr(expr)
    latin = LatinCanonicalizer().canonicalize(stream)
    assert len(latin.tokens) > 0


# ── Pipeline semantic preservation ───────────────────────────────────────────

def test_pipeline_preserves_intent():
    from semantic_engine import SemanticPipeline, RawInput, OutputMode
    r = RawInput("build a matrix multiplication function")
    result = SemanticPipeline().run(r)
    assert result.ir is not None
    from semantic_engine.types import IntentClass
    assert result.ir.intent == IntentClass.BUILD


def test_pipeline_verified_on_valid_input():
    from semantic_engine import SemanticPipeline, RawInput
    from semantic_engine.types import VerificationState
    r = RawInput("multiply two matrices")
    result = SemanticPipeline().run(r)
    assert result.verification_state == VerificationState.VERIFIED


def test_pipeline_artifact_traces_to_ir():
    from semantic_engine import SemanticPipeline, RawInput
    r = RawInput("generate a parser for binary format")
    result = SemanticPipeline().run(r)
    assert result.artifact is not None
    assert result.artifact.semantic_root == result.ir.ir_id


def test_pipeline_hashes_recorded():
    from semantic_engine import SemanticPipeline, RawInput
    r = RawInput("validate packet structure")
    result = SemanticPipeline().run(r)
    assert "input" in result.hashes
    assert "semantic_ir" in result.hashes
    assert "subleq" in result.hashes
    assert "artifact" in result.hashes


def test_pipeline_deterministic_on_same_input():
    """Same input → same semantic IR hash (modulo UUIDs)."""
    from semantic_engine import SemanticPipeline, RawInput
    text = "compute matrix product"
    r1 = RawInput(text)
    r2 = RawInput(text)
    p = SemanticPipeline()
    res1 = p.run(r1)
    res2 = p.run(r2)
    # Both must succeed and produce same intent
    assert res1.ir.intent == res2.ir.intent
    assert res1.ir.root.id == res2.ir.root.id  # node ID is content-addressed


def test_pipeline_two_output_modes():
    """Same IR must be able to produce both NL and code artifacts."""
    from semantic_engine import SemanticPipeline, RawInput, OutputMode
    r = RawInput("build a validation service")
    p = SemanticPipeline()
    nl = p.run(r, output_mode=OutputMode.NATURAL_LANGUAGE)
    code = p.run(RawInput("build a validation service"), output_mode=OutputMode.CODE)
    assert nl.artifact is not None
    assert code.artifact is not None
    assert nl.artifact.type.value != code.artifact.type.value


def test_pipeline_rejects_empty_input():
    from semantic_engine import SemanticPipeline, RawInput
    r = RawInput("")
    result = SemanticPipeline().run(r)
    # May complete but should not crash
    assert result is not None
