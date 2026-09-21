# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phase 7 — Boolean engine."""

import pytest
from semantic_engine.boolean_engine import (
    AND, OR, NOT, XOR, IMPLIES, EQUIV,
    BooleanLiteral, BooleanVar, BooleanExpression,
    var, lit, normalize, to_cnf, simplify,
    semantic_constraints_to_boolean,
)
from semantic_engine.types import BooleanOperator


def test_literal_repr():
    assert repr(BooleanLiteral(True)) == "⊤"
    assert repr(BooleanLiteral(False)) == "⊥"


def test_var_repr():
    assert repr(BooleanVar("x")) == "x"
    assert repr(BooleanVar("x", negated=True)) == "¬x"


def test_and_repr():
    e = AND(var("a"), var("b"))
    assert "∧" in repr(e)


def test_simplify_double_negation():
    e = NOT(NOT(var("x")))
    result = simplify(e)
    assert isinstance(result, BooleanVar)
    assert result.name == "x"
    assert not result.negated


def test_simplify_and_with_true():
    e = AND(BooleanLiteral(True), var("x"))
    result = simplify(e)
    assert isinstance(result, BooleanVar)
    assert result.name == "x"


def test_simplify_and_with_false():
    e = AND(BooleanLiteral(False), var("x"))
    result = simplify(e)
    assert isinstance(result, BooleanLiteral)
    assert not result.value


def test_simplify_or_with_true():
    e = OR(BooleanLiteral(True), var("x"))
    result = simplify(e)
    assert isinstance(result, BooleanLiteral)
    assert result.value


def test_implies_expansion():
    e = IMPLIES(var("a"), var("b"))
    result = simplify(e)
    assert repr(result)  # should not crash


def test_equivalence_expansion():
    e = EQUIV(var("a"), var("b"))
    result = simplify(e)
    assert repr(result)


def test_xor_expansion():
    e = XOR(var("a"), var("b"))
    result = simplify(e)
    assert repr(result)


def test_normalize_idempotent():
    e = AND(var("a"), OR(var("b"), NOT(var("c"))))
    n1 = normalize(e)
    n2 = normalize(n1)
    assert repr(n1) == repr(n2)


def test_normalize_literal_true():
    assert isinstance(normalize(BooleanLiteral(True)), BooleanLiteral)


def test_constraints_to_boolean_empty():
    bnf = semantic_constraints_to_boolean([])
    assert isinstance(bnf.normalized, BooleanLiteral)
    assert bnf.normalized.value is True


def test_constraints_to_boolean_single():
    from semantic_engine.ir import Constraint
    c = Constraint.make("deterministic", "deterministic")
    bnf = semantic_constraints_to_boolean([c])
    assert bnf.stable


def test_constraints_to_boolean_multiple():
    from semantic_engine.ir import Constraint
    cs = [Constraint.make("deterministic", "deterministic"),
          Constraint.make("immutable", "immutable")]
    bnf = semantic_constraints_to_boolean(cs)
    assert bnf.stable
