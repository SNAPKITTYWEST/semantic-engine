# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 10: Regex Latin lexer — token recognition and field extraction.

The lexer recognizes structure. It does NOT determine semantic meaning.
Invalid Latin blocks fail closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .latin import LatinCanonicalForm, LatinToken, vocabulary
from .types import FSMState, PipelineError


# ── Regex patterns ────────────────────────────────────────────────────────────

# Block delimiters
_RE_BLOCK_OPEN    = re.compile(r"\bINITIUM\b")
_RE_BLOCK_CLOSE   = re.compile(r"\bFINIS\b")

# Operators
_RE_OPERATORS     = re.compile(
    r"\b(ET|VEL|NON|AUT|SI|IFF|VERUM|FALSUM)\b"
)

# Operands with optional payload: OPUS[varname]
_RE_OPERAND       = re.compile(r"\b(OPUS|FORMA|COMPUTA|MULTIPLICA|ADDE|SUBTRAHE|PARSE|"
                                r"AEDIFICA|GENERA|VERIFICA|CONVERTE|RESTITUE|"
                                r"MATRIX|VECTOR|PARSER|SERVITIUM|FORMA_BINARIA|TEMPUS)"
                                r"(?:\[([^\]]*)\])?")

# Instruction fields: KEYWORD:VALUE
_RE_FIELD         = re.compile(r"\b([A-Z_]+):(\S+)")

# Delimiters — semicolons, pipes
_RE_DELIM         = re.compile(r"[;|]")


@dataclass(frozen=True)
class LexedToken:
    kind: str           # "operator", "operand", "open", "close", "field", "delim", "unknown"
    lexeme: str
    payload: str = ""
    position: int = 0


@dataclass
class LexResult:
    tokens: list[LexedToken] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    valid: bool = True


# ── Lexer ─────────────────────────────────────────────────────────────────────

class LatinLexer:
    """
    Regex-based lexer for Latin canonical form text.
    Produces LexResult with typed tokens.
    """

    def lex(self, text: str) -> LexResult:
        result = LexResult()
        vocab = vocabulary()
        pos = 0
        tpos = 0

        # Tokenize by scanning left to right
        while pos < len(text):
            # Skip whitespace
            ws = re.match(r"\s+", text[pos:])
            if ws:
                pos += ws.end()
                continue

            # Block open/close
            m = _RE_BLOCK_OPEN.match(text, pos)
            if m:
                result.tokens.append(LexedToken("open", "INITIUM", position=tpos))
                pos = m.end(); tpos += 1; continue

            m = _RE_BLOCK_CLOSE.match(text, pos)
            if m:
                result.tokens.append(LexedToken("close", "FINIS", position=tpos))
                pos = m.end(); tpos += 1; continue

            # Operands (with optional [payload])
            m = _RE_OPERAND.match(text, pos)
            if m:
                lexeme  = m.group(1)
                payload = m.group(2) or ""
                if lexeme not in vocab:
                    result.errors.append(f"Unknown lexeme '{lexeme}' at pos {pos}")
                    result.valid = False
                result.tokens.append(LexedToken("operand", lexeme, payload, tpos))
                pos = m.end(); tpos += 1; continue

            # Operators
            m = _RE_OPERATORS.match(text, pos)
            if m:
                lexeme = m.group(1)
                if lexeme not in vocab:
                    result.errors.append(f"Unknown operator '{lexeme}' at pos {pos}")
                    result.valid = False
                result.tokens.append(LexedToken("operator", lexeme, position=tpos))
                pos = m.end(); tpos += 1; continue

            # Field annotations
            m = _RE_FIELD.match(text, pos)
            if m:
                result.tokens.append(LexedToken("field", m.group(1), m.group(2), tpos))
                pos = m.end(); tpos += 1; continue

            # Delimiters
            m = _RE_DELIM.match(text, pos)
            if m:
                result.tokens.append(LexedToken("delim", m.group(0), position=tpos))
                pos = m.end(); tpos += 1; continue

            # Unknown character/word — fail closed
            word_m = re.match(r"\S+", text[pos:])
            unknown = word_m.group(0) if word_m else text[pos]
            result.errors.append(f"Unrecognized token '{unknown}' at position {pos}")
            result.valid = False
            pos += len(unknown); tpos += 1

        if not result.valid:
            raise PipelineError(
                FSMState.LATIN_LEXING,
                "Latin lexer errors: " + "; ".join(result.errors),
            )

        return result


    def lex_canonical(self, form: LatinCanonicalForm) -> LexResult:
        """Lex from a LatinCanonicalForm directly."""
        return self.lex(form.text())
