# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 9: Latin canonicalizer — controlled vocabulary mapping.

Boolean tokens → Semantic IDs → Latin lexemes.
Every Latin lexeme maps to a stable semantic_id. No uncontrolled synonyms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .token_engine import (
    BoolToken, BooleanTokenStream,
    TTYPE_AND, TTYPE_OR, TTYPE_NOT, TTYPE_XOR,
    TTYPE_IMPLIES, TTYPE_EQUIVALENCE, TTYPE_TRUE, TTYPE_FALSE,
    TTYPE_OPEN, TTYPE_CLOSE, TTYPE_VAR, TTYPE_VAR_NEG,
)
from .types import FSMState, PipelineError

# Map from token type → canonical Latin lexeme
_TTYPE_TO_LATIN: dict[int, str] = {
    TTYPE_AND:         "ET",
    TTYPE_OR:          "VEL",
    TTYPE_NOT:         "NON",
    TTYPE_XOR:         "AUT",
    TTYPE_IMPLIES:     "SI",
    TTYPE_EQUIVALENCE: "IFF",
    TTYPE_TRUE:        "VERUM",
    TTYPE_FALSE:       "FALSUM",
    TTYPE_OPEN:        "INITIUM",
    TTYPE_CLOSE:       "FINIS",
}

_LATIN_TO_TTYPE: dict[str, int] = {v: k for k, v in _TTYPE_TO_LATIN.items()}


# ── Latin token ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatinToken:
    lexeme: str
    semantic_id: str
    position: int
    payload: str = ""       # variable name for VAR tokens


@dataclass
class LatinLexeme:
    word: str
    semantic_id: str
    operator: Optional[str]
    arity: int


@dataclass
class LatinCanonicalForm:
    tokens: list[LatinToken] = field(default_factory=list)
    source_hash: str = ""

    def text(self) -> str:
        return " ".join(t.lexeme if not t.payload else f"{t.lexeme}[{t.payload}]" for t in self.tokens)


# ── Vocabulary loader ─────────────────────────────────────────────────────────

def _load_vocabulary() -> dict[str, LatinLexeme]:
    vocab_path = Path(__file__).parent.parent.parent / "lexicon" / "latin_vocabulary.json"
    if not vocab_path.exists():
        return {}
    data = json.loads(vocab_path.read_text())
    return {
        word: LatinLexeme(
            word=word,
            semantic_id=entry["semantic_id"],
            operator=entry.get("operator"),
            arity=entry.get("arity", 0),
        )
        for word, entry in data.get("lexemes", {}).items()
    }


_VOCABULARY: dict[str, LatinLexeme] = {}


def vocabulary() -> dict[str, LatinLexeme]:
    global _VOCABULARY
    if not _VOCABULARY:
        _VOCABULARY = _load_vocabulary()
    return _VOCABULARY


# ── Canonicalizer ─────────────────────────────────────────────────────────────

class LatinCanonicalizer:
    """Transform a BooleanTokenStream into a LatinCanonicalForm."""

    def canonicalize(self, stream: BooleanTokenStream) -> LatinCanonicalForm:
        vocab = vocabulary()
        tokens: list[LatinToken] = []

        for i, tok in enumerate(stream.tokens):
            if tok.token_type in (TTYPE_VAR, TTYPE_VAR_NEG):
                lexeme = "NON" if tok.token_type == TTYPE_VAR_NEG else "OPUS"
                sem_id = "BOOL_NOT" if tok.token_type == TTYPE_VAR_NEG else "SEMANTIC_OP"
                var_name = tok.payload.decode("utf-8") if tok.payload else ""
                if tok.token_type == TTYPE_VAR_NEG:
                    tokens.append(LatinToken(lexeme="NON", semantic_id="BOOL_NOT", position=i))
                tokens.append(LatinToken(
                    lexeme="OPUS",
                    semantic_id="SEMANTIC_OP",
                    position=i,
                    payload=var_name,
                ))
                continue

            latin = _TTYPE_TO_LATIN.get(tok.token_type)
            if latin is None:
                raise PipelineError(
                    FSMState.LATIN_CANONICALIZATION,
                    f"No Latin mapping for token type 0x{tok.token_type:02x} at position {i}",
                )

            lexeme_def = vocab.get(latin)
            if lexeme_def is None:
                raise PipelineError(
                    FSMState.LATIN_CANONICALIZATION,
                    f"Latin lexeme '{latin}' not in controlled vocabulary",
                )

            tokens.append(LatinToken(
                lexeme=latin,
                semantic_id=lexeme_def.semantic_id,
                position=i,
            ))

        import hashlib
        src_hash = hashlib.sha256(" ".join(t.lexeme for t in tokens).encode()).hexdigest()[:16]
        return LatinCanonicalForm(tokens=tokens, source_hash=src_hash)


# ── Inverse: Latin → semantic IDs ─────────────────────────────────────────────

def latin_to_semantic_ids(form: LatinCanonicalForm) -> list[tuple[str, str]]:
    """Return list of (lexeme, semantic_id) pairs."""
    return [(t.lexeme, t.semantic_id) for t in form.tokens]
