# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phase 8 — Boolean token engine."""

import pytest
from semantic_engine.boolean_engine import AND, OR, NOT, var, lit, BooleanLiteral, BooleanVar
from semantic_engine.token_engine import (
    BoolEncoder, BoolDecoder, BooleanTokenStream,
    encode_binary, decode_binary, encode, decode, round_trip,
    TTYPE_AND, TTYPE_OR, TTYPE_NOT, TTYPE_TRUE, TTYPE_FALSE,
    TTYPE_VAR, TTYPE_VAR_NEG, TTYPE_OPEN, TTYPE_CLOSE,
)
from semantic_engine.types import FSMState, TokenError


def encode_and_check(expr):
    stream = BoolEncoder().encode_expr(expr)
    binary = encode_binary(stream)
    decoded = decode_binary(binary)
    return stream, binary, decoded


def test_encode_literal_true():
    stream, binary, decoded = encode_and_check(BooleanLiteral(True))
    assert len(decoded.tokens) == 1
    assert decoded.tokens[0].token_type == TTYPE_TRUE


def test_encode_literal_false():
    stream, binary, decoded = encode_and_check(BooleanLiteral(False))
    assert decoded.tokens[0].token_type == TTYPE_FALSE


def test_encode_var():
    stream, binary, decoded = encode_and_check(BooleanVar("x"))
    assert decoded.tokens[0].token_type == TTYPE_VAR
    assert decoded.tokens[0].payload == b"x"


def test_encode_var_negated():
    stream, binary, decoded = encode_and_check(BooleanVar("x", negated=True))
    assert decoded.tokens[0].token_type == TTYPE_VAR_NEG
    assert decoded.tokens[0].payload == b"x"


def test_encode_and():
    expr = AND(var("a"), var("b"))
    stream = BoolEncoder().encode_expr(expr)
    types = [t.token_type for t in stream]
    assert TTYPE_AND in types
    assert TTYPE_OPEN in types
    assert TTYPE_CLOSE in types


def test_encode_not():
    expr = NOT(var("x"))
    stream = BoolEncoder().encode_expr(expr)
    types = [t.token_type for t in stream]
    assert TTYPE_NOT in types


def test_round_trip_literal():
    expr = BooleanLiteral(True)
    result = round_trip(expr)
    assert isinstance(result, BooleanLiteral)
    assert result.value is True


def test_round_trip_var():
    expr = BooleanVar("myvar")
    result = round_trip(expr)
    assert isinstance(result, BooleanVar)
    assert result.name == "myvar"
    assert not result.negated


def test_round_trip_complex():
    expr = AND(var("a"), OR(var("b"), NOT(var("c"))))
    result = round_trip(expr)
    # repr should match after round-trip
    assert repr(result) == repr(expr)


def test_binary_bad_magic():
    with pytest.raises(TokenError):
        decode_binary(b"XXXX\x01\x00\x01")


def test_binary_too_short():
    with pytest.raises(TokenError):
        decode_binary(b"BO")


def test_token_positions():
    expr = AND(var("a"), var("b"))
    stream = BoolEncoder().encode_expr(expr)
    for i, tok in enumerate(stream):
        assert tok.position == i


def test_token_ids_unique():
    expr = AND(var("a"), OR(var("b"), var("c")))
    stream = BoolEncoder().encode_expr(expr)
    ids = [t.token_id for t in stream]
    assert len(ids) == len(set(ids))
