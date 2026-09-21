# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 7: Boolean engine — deterministic normalization to CNF/DNF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union
import hashlib

from .types import BooleanOperator, FSMState, PipelineError


# ── Boolean expression tree ───────────────────────────────────────────────────

@dataclass
class BooleanLiteral:
    value: bool

    def __repr__(self) -> str:
        return "⊤" if self.value else "⊥"

    def negate(self) -> BooleanLiteral:
        return BooleanLiteral(not self.value)


@dataclass
class BooleanVar:
    name: str
    negated: bool = False

    def __repr__(self) -> str:
        return f"¬{self.name}" if self.negated else self.name

    def negate(self) -> BooleanVar:
        return BooleanVar(self.name, not self.negated)


BooleanAtom = Union[BooleanLiteral, BooleanVar]


@dataclass
class BooleanExpression:
    operator: BooleanOperator
    operands: list[Union[BooleanExpression, BooleanAtom]] = field(default_factory=list)
    constraint_refs: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.operator == BooleanOperator.TRUE:
            return "⊤"
        if self.operator == BooleanOperator.FALSE:
            return "⊥"
        if self.operator == BooleanOperator.NOT:
            return f"¬({self.operands[0]})"
        sym = {
            BooleanOperator.AND:         "∧",
            BooleanOperator.OR:          "∨",
            BooleanOperator.XOR:         "⊕",
            BooleanOperator.IMPLIES:     "→",
            BooleanOperator.EQUIVALENCE: "↔",
        }.get(self.operator, str(self.operator))
        inner = f" {sym} ".join(str(o) for o in self.operands)
        return f"({inner})"

    def hash(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()[:16]


# ── Constructors ──────────────────────────────────────────────────────────────

def var(name: str, negated: bool = False) -> BooleanVar:
    return BooleanVar(name, negated)

def lit(value: bool) -> BooleanLiteral:
    return BooleanLiteral(value)

def AND(*args) -> BooleanExpression:
    return BooleanExpression(BooleanOperator.AND, list(args))

def OR(*args) -> BooleanExpression:
    return BooleanExpression(BooleanOperator.OR, list(args))

def NOT(x) -> BooleanExpression:
    return BooleanExpression(BooleanOperator.NOT, [x])

def XOR(a, b) -> BooleanExpression:
    return BooleanExpression(BooleanOperator.XOR, [a, b])

def IMPLIES(a, b) -> BooleanExpression:
    return BooleanExpression(BooleanOperator.IMPLIES, [a, b])

def EQUIV(a, b) -> BooleanExpression:
    return BooleanExpression(BooleanOperator.EQUIVALENCE, [a, b])


# ── Simplification ────────────────────────────────────────────────────────────

Expr = Union[BooleanExpression, BooleanAtom]


def simplify(expr: Expr) -> Expr:
    """Simplify constants and double negations."""
    if isinstance(expr, (BooleanLiteral, BooleanVar)):
        return expr

    expr = BooleanExpression(expr.operator, [simplify(o) for o in expr.operands], expr.constraint_refs)

    op = expr.operator

    # Constant folding
    if op == BooleanOperator.NOT:
        inner = expr.operands[0]
        if isinstance(inner, BooleanLiteral):
            return BooleanLiteral(not inner.value)
        if isinstance(inner, BooleanVar):
            return inner.negate()
        if isinstance(inner, BooleanExpression) and inner.operator == BooleanOperator.NOT:
            return inner.operands[0]  # ¬¬A → A

    if op == BooleanOperator.AND:
        if any(isinstance(o, BooleanLiteral) and not o.value for o in expr.operands):
            return BooleanLiteral(False)
        expr.operands = [o for o in expr.operands if not (isinstance(o, BooleanLiteral) and o.value)]
        if not expr.operands:
            return BooleanLiteral(True)
        if len(expr.operands) == 1:
            return expr.operands[0]

    if op == BooleanOperator.OR:
        if any(isinstance(o, BooleanLiteral) and o.value for o in expr.operands):
            return BooleanLiteral(True)
        expr.operands = [o for o in expr.operands if not (isinstance(o, BooleanLiteral) and not o.value)]
        if not expr.operands:
            return BooleanLiteral(False)
        if len(expr.operands) == 1:
            return expr.operands[0]

    # A → B  ≡  ¬A ∨ B
    if op == BooleanOperator.IMPLIES:
        a, b = expr.operands[0], expr.operands[1]
        return simplify(OR(NOT(a), b))

    # A ↔ B  ≡  (A → B) ∧ (B → A)
    if op == BooleanOperator.EQUIVALENCE:
        a, b = expr.operands[0], expr.operands[1]
        return simplify(AND(IMPLIES(a, b), IMPLIES(b, a)))

    # A ⊕ B  ≡  (A ∨ B) ∧ ¬(A ∧ B)
    if op == BooleanOperator.XOR:
        a, b = expr.operands[0], expr.operands[1]
        return simplify(AND(OR(a, b), NOT(AND(a, b))))

    return expr


def push_not_inward(expr: Expr) -> Expr:
    """Apply De Morgan's laws, push negations to leaves."""
    if isinstance(expr, (BooleanLiteral, BooleanVar)):
        return expr

    if expr.operator == BooleanOperator.NOT:
        inner = expr.operands[0]
        if isinstance(inner, BooleanLiteral):
            return BooleanLiteral(not inner.value)
        if isinstance(inner, BooleanVar):
            return inner.negate()
        if isinstance(inner, BooleanExpression):
            if inner.operator == BooleanOperator.NOT:
                return push_not_inward(inner.operands[0])
            if inner.operator == BooleanOperator.AND:
                return push_not_inward(OR(*[NOT(o) for o in inner.operands]))
            if inner.operator == BooleanOperator.OR:
                return push_not_inward(AND(*[NOT(o) for o in inner.operands]))

    return BooleanExpression(
        expr.operator,
        [push_not_inward(o) for o in expr.operands],
        expr.constraint_refs,
    )


def distribute_or_over_and(expr: Expr) -> Expr:
    """Distribute OR over AND to produce CNF."""
    if isinstance(expr, (BooleanLiteral, BooleanVar)):
        return expr

    expr = BooleanExpression(
        expr.operator,
        [distribute_or_over_and(o) for o in expr.operands],
        expr.constraint_refs,
    )

    if expr.operator == BooleanOperator.OR:
        # Find any AND among operands
        for i, op in enumerate(expr.operands):
            if isinstance(op, BooleanExpression) and op.operator == BooleanOperator.AND:
                others = expr.operands[:i] + expr.operands[i+1:]
                distributed = AND(*[
                    distribute_or_over_and(OR(clause, *others))
                    for clause in op.operands
                ])
                return distribute_or_over_and(distributed)

    return expr


def to_cnf(expr: Expr) -> Expr:
    """Convert to Conjunctive Normal Form."""
    e = simplify(expr)
    e = push_not_inward(e)
    e = distribute_or_over_and(e)
    e = simplify(e)
    return e


def normalize(expr: Expr) -> Expr:
    """
    Deterministic normalization. normalize(normalize(x)) == normalize(x).
    """
    result = to_cnf(expr)
    # Idempotency check
    again = to_cnf(result)
    if repr(again) != repr(result):
        result = again
    return result


# ── Boolean Normal Form container ─────────────────────────────────────────────

@dataclass
class BooleanNormalForm:
    original: Expr
    normalized: Expr
    form: str = "CNF"
    stable: bool = False

    def verify_stability(self) -> bool:
        again = normalize(self.normalized)
        self.stable = repr(again) == repr(self.normalized)
        return self.stable


# ── Semantic → Boolean ────────────────────────────────────────────────────────

def semantic_constraints_to_boolean(constraints: list) -> BooleanNormalForm:
    """
    Convert a list of Constraint objects into a Boolean expression.
    Each constraint becomes a BooleanVar. Constraints are AND-ed together.
    """
    if not constraints:
        expr: Expr = BooleanLiteral(True)
    else:
        vars_ = [var(c.predicate if hasattr(c, "predicate") else str(c)) for c in constraints]
        expr = vars_[0] if len(vars_) == 1 else AND(*vars_)

    norm = normalize(expr)
    bnf = BooleanNormalForm(original=expr, normalized=norm)
    bnf.verify_stability()
    return bnf
