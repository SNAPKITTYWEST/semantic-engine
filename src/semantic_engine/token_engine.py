# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 8: Boolean Token Engine — binary serialization of Boolean IR.

Input:  Boolean IR
Output: BooleanTokenStream

Invariant: Decode(Encode(x)) == x for all canonical Boolean structures.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Union

from .boolean_engine import (
    AND, NOT, OR, XOR, BooleanAtom, BooleanExpression,
    BooleanLiteral, BooleanVar, Expr,
)
from .types import BooleanOperator, FSMState, TokenError


# ── Token types ───────────────────────────────────────────────────────────────

TTYPE_LITERAL       = 0x01
TTYPE_VAR           = 0x02
TTYPE_VAR_NEG       = 0x03
TTYPE_AND           = 0x10
TTYPE_OR            = 0x11
TTYPE_NOT           = 0x12
TTYPE_XOR           = 0x13
TTYPE_IMPLIES       = 0x14
TTYPE_EQUIVALENCE   = 0x15
TTYPE_TRUE          = 0x20
TTYPE_FALSE         = 0x21
TTYPE_OPEN          = 0x30
TTYPE_CLOSE         = 0x31
TTYPE_ARITY         = 0x40

_OP_TO_TTYPE: dict[BooleanOperator, int] = {
    BooleanOperator.AND:         TTYPE_AND,
    BooleanOperator.OR:          TTYPE_OR,
    BooleanOperator.NOT:         TTYPE_NOT,
    BooleanOperator.XOR:         TTYPE_XOR,
    BooleanOperator.IMPLIES:     TTYPE_IMPLIES,
    BooleanOperator.EQUIVALENCE: TTYPE_EQUIVALENCE,
    BooleanOperator.TRUE:        TTYPE_TRUE,
    BooleanOperator.FALSE:       TTYPE_FALSE,
}

_TTYPE_TO_OP: dict[int, BooleanOperator] = {v: k for k, v in _OP_TO_TTYPE.items()}


# ── Token ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoolToken:
    token_id: int
    token_type: int
    arity: int
    position: int
    payload: bytes = b""
    semantic_reference: str = ""
    temporal_reference: str = ""
    geometry_reference: str = ""

    def __repr__(self) -> str:
        tname = {
            TTYPE_LITERAL: "LIT", TTYPE_VAR: "VAR", TTYPE_VAR_NEG: "NEGVAR",
            TTYPE_AND: "AND", TTYPE_OR: "OR", TTYPE_NOT: "NOT",
            TTYPE_XOR: "XOR", TTYPE_TRUE: "TRUE", TTYPE_FALSE: "FALSE",
            TTYPE_OPEN: "(", TTYPE_CLOSE: ")",
        }.get(self.token_type, f"0x{self.token_type:02x}")
        return f"Token({tname}, pos={self.position}, arity={self.arity})"


@dataclass
class BooleanTokenStream:
    tokens: list[BoolToken] = field(default_factory=list)

    def append(self, t: BoolToken) -> None:
        self.tokens.append(t)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


# ── Encoder ───────────────────────────────────────────────────────────────────

class BoolEncoder:
    def __init__(self):
        self._pos = 0
        self._id  = 0
        self._stream = BooleanTokenStream()

    def _tok(self, ttype: int, arity: int, payload: bytes = b"", sem_ref: str = "") -> BoolToken:
        t = BoolToken(
            token_id=self._id,
            token_type=ttype,
            arity=arity,
            position=self._pos,
            payload=payload,
            semantic_reference=sem_ref,
        )
        self._id += 1
        self._pos += 1
        return t

    def encode_expr(self, expr: Expr) -> BooleanTokenStream:
        self._pos = 0
        self._id  = 0
        self._stream = BooleanTokenStream()
        self._encode(expr)
        return self._stream

    def _encode(self, expr: Expr) -> None:
        if isinstance(expr, BooleanLiteral):
            ttype = TTYPE_TRUE if expr.value else TTYPE_FALSE
            self._stream.append(self._tok(ttype, 0, b"\x01" if expr.value else b"\x00"))
            return

        if isinstance(expr, BooleanVar):
            ttype = TTYPE_VAR_NEG if expr.negated else TTYPE_VAR
            payload = expr.name.encode("utf-8")
            self._stream.append(self._tok(ttype, 0, payload, sem_ref=expr.name))
            return

        op = expr.operator
        ttype = _OP_TO_TTYPE.get(op)
        if ttype is None:
            raise TokenError(FSMState.TOKENIZATION, f"Unknown operator: {op}")

        arity = len(expr.operands)
        self._stream.append(self._tok(TTYPE_OPEN, 0))
        self._stream.append(self._tok(ttype, arity))
        for operand in expr.operands:
            self._encode(operand)
        self._stream.append(self._tok(TTYPE_CLOSE, 0))


# ── Binary serialization ──────────────────────────────────────────────────────

_HEADER = b"BOOL"
_VERSION = 1


def encode_binary(stream: BooleanTokenStream) -> bytes:
    """
    Binary format:
      4 bytes magic: BOOL
      1 byte version
      4 bytes token count
      per token: 1 byte type, 1 byte arity, 2 bytes payload_len, N bytes payload
    """
    buf = bytearray()
    buf += _HEADER
    buf += struct.pack(">BH", _VERSION, len(stream.tokens))
    for tok in stream.tokens:
        pay = tok.payload[:255]
        buf += struct.pack(">BBH", tok.token_type, tok.arity, len(pay))
        buf += pay
    return bytes(buf)


def decode_binary(data: bytes) -> BooleanTokenStream:
    """Inverse of encode_binary. Raises TokenError on malformed input."""
    if len(data) < 7:
        raise TokenError(FSMState.TOKENIZATION, "Binary data too short")
    if data[:4] != _HEADER:
        raise TokenError(FSMState.TOKENIZATION, f"Bad magic: {data[:4]!r}")

    version, count = struct.unpack_from(">BH", data, 4)
    if version != _VERSION:
        raise TokenError(FSMState.TOKENIZATION, f"Unknown version {version}")

    offset = 7
    stream = BooleanTokenStream()
    for i in range(count):
        if offset + 4 > len(data):
            raise TokenError(FSMState.TOKENIZATION, f"Truncated at token {i}")
        ttype, arity, pay_len = struct.unpack_from(">BBH", data, offset)
        offset += 4
        payload = data[offset:offset + pay_len]
        offset += pay_len
        stream.append(BoolToken(
            token_id=i, token_type=ttype, arity=arity,
            position=i, payload=payload,
        ))
    return stream


# ── Decoder (token stream → Boolean IR) ──────────────────────────────────────

class BoolDecoder:
    def __init__(self, stream: BooleanTokenStream):
        self._tokens = list(stream.tokens)
        self._pos = 0

    def _peek(self) -> BoolToken | None:
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos]

    def _consume(self) -> BoolToken:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def decode(self) -> Expr:
        expr = self._decode_expr()
        return expr

    def _decode_expr(self) -> Expr:
        tok = self._peek()
        if tok is None:
            raise TokenError(FSMState.TOKENIZATION, "Unexpected end of token stream")

        if tok.token_type == TTYPE_TRUE:
            self._consume()
            return BooleanLiteral(True)
        if tok.token_type == TTYPE_FALSE:
            self._consume()
            return BooleanLiteral(False)
        if tok.token_type == TTYPE_VAR:
            self._consume()
            return BooleanVar(tok.payload.decode("utf-8"))
        if tok.token_type == TTYPE_VAR_NEG:
            self._consume()
            return BooleanVar(tok.payload.decode("utf-8"), negated=True)

        if tok.token_type == TTYPE_OPEN:
            self._consume()  # consume (
            op_tok = self._consume()
            op = _TTYPE_TO_OP.get(op_tok.token_type)
            if op is None:
                raise TokenError(FSMState.TOKENIZATION, f"Unknown op token 0x{op_tok.token_type:02x}")
            operands = []
            while True:
                nxt = self._peek()
                if nxt is None:
                    raise TokenError(FSMState.TOKENIZATION, "Missing CLOSE token")
                if nxt.token_type == TTYPE_CLOSE:
                    self._consume()
                    break
                operands.append(self._decode_expr())
            return BooleanExpression(op, operands)

        raise TokenError(FSMState.TOKENIZATION, f"Unexpected token type 0x{tok.token_type:02x}")


# ── Round-trip convenience ────────────────────────────────────────────────────

def encode(expr: Expr) -> bytes:
    stream = BoolEncoder().encode_expr(expr)
    return encode_binary(stream)


def decode(data: bytes) -> Expr:
    stream = decode_binary(data)
    return BoolDecoder(stream).decode()


def round_trip(expr: Expr) -> Expr:
    return decode(encode(expr))
