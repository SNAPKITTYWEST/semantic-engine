# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NLP subsystem — tokenization, entity extraction, intent classification, BERT interface.

BERT operates as an internal interpretive tool only. Output is SemanticCandidate,
never executable code. The FSM remains the admissibility gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .types import IntentClass, NormalizedInput, RawInput, SemanticCandidate


# ── Controlled vocabulary for intent ─────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[IntentClass, list[str]]] = [
    (IntentClass.BUILD,       ["build", "create", "make", "construct", "implement", "write"]),
    (IntentClass.TRANSFORM,   ["transform", "convert", "translate", "map", "rewrite", "change"]),
    (IntentClass.VALIDATE,    ["validate", "verify", "check", "confirm", "ensure", "test"]),
    (IntentClass.GENERATE,    ["generate", "produce", "emit", "output", "render"]),
    (IntentClass.PARSE,       ["parse", "read", "decode", "lex", "scan", "interpret"]),
    (IntentClass.COMPUTE,     ["compute", "calculate", "multiply", "add", "subtract", "solve"]),
    (IntentClass.RECONSTRUCT, ["reconstruct", "restore", "recover", "rebuild", "regenerate"]),
]

_OPERATOR_PATTERNS = re.compile(
    r"\b(multiply|add|subtract|divide|transform|convert|validate|parse|build|generate|"
    r"compute|verify|reconstruct|normalize|encode|decode|serialize|deserialize)\b",
    re.IGNORECASE,
)

_ENTITY_PATTERNS = re.compile(
    r"\b(matrix|vector|parser|backend|service|function|module|component|token|"
    r"format|stream|buffer|table|schema|graph|tree|list|array|string|integer|"
    r"boolean|packet|message|frame|field|register)\b",
    re.IGNORECASE,
)

_CONSTRAINT_PATTERNS = re.compile(
    r"\b(deterministic|immutable|idempotent|commutative|associative|invertible|"
    r"ordered|typed|bounded|validated|correct|minimal|canonical|stable)\b",
    re.IGNORECASE,
)

_RELATIONSHIP_PATTERN = re.compile(
    r"(\w+)\s+(of|from|to|into|for|with|by|through|via|as)\s+(\w+)",
    re.IGNORECASE,
)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def segment_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ── Rule-based NLP (no external model required) ───────────────────────────────

def classify_intent(text: str) -> tuple[IntentClass, float]:
    lower = text.lower()
    scores: dict[IntentClass, int] = {}
    for intent, keywords in _INTENT_PATTERNS:
        for kw in keywords:
            if kw in lower:
                scores[intent] = scores.get(intent, 0) + 1
    if not scores:
        return IntentClass.UNKNOWN, 0.3
    best = max(scores, key=lambda k: scores[k])
    confidence = min(0.5 + scores[best] * 0.15, 0.95)
    return best, confidence


def extract_entities(text: str) -> list[str]:
    return list({m.group(0).lower() for m in _ENTITY_PATTERNS.finditer(text)})


def extract_operators(text: str) -> list[str]:
    return list({m.group(0).lower() for m in _OPERATOR_PATTERNS.finditer(text)})


def extract_constraints(text: str) -> list[str]:
    return list({m.group(0).lower() for m in _CONSTRAINT_PATTERNS.finditer(text)})


def extract_relationships(text: str) -> list[tuple[str, str, str]]:
    return [(m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
            for m in _RELATIONSHIP_PATTERN.finditer(text)]


def detect_ambiguity(text: str) -> bool:
    ambiguous_markers = re.compile(
        r"\b(it|this|that|they|something|somehow|maybe|perhaps|possibly|"
        r"or something|etc|and so on)\b",
        re.IGNORECASE,
    )
    return bool(ambiguous_markers.search(text))


# ── BERT interface ────────────────────────────────────────────────────────────

class BertDecoder:
    """
    Internal BERT interface for semantic interpretation.

    If the `transformers` package is available and a model is configured,
    uses real BERT embeddings. Otherwise falls back to rule-based analysis.

    BERT output is always SemanticCandidate — never executable code.
    The FSM is the admissibility gate regardless of BERT confidence.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._use_bert = False

        if model_name:
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
                import torch  # type: ignore
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
                self._torch = torch
                self._use_bert = True
            except ImportError:
                pass  # fall back to rule-based

    def _bert_embeddings(self, text: str) -> Optional[list[float]]:
        if not self._use_bert:
            return None
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        return cls_embedding if isinstance(cls_embedding, list) else [cls_embedding]

    def decode(self, normalized: NormalizedInput) -> SemanticCandidate:
        text = normalized.text
        intent, confidence = classify_intent(text)
        entities    = extract_entities(text)
        operators   = extract_operators(text)
        constraints = extract_constraints(text)
        rels        = extract_relationships(text)
        ambiguous   = detect_ambiguity(text)
        embeddings  = self._bert_embeddings(text)

        return SemanticCandidate(
            input_ref=normalized.original.input_id,
            intent=intent,
            entities=entities,
            operators=operators,
            constraints=constraints,
            relationships=rels,
            confidence=confidence,
            ambiguous=ambiguous,
            raw_embeddings=embeddings,
        )


# ── Input normalization ───────────────────────────────────────────────────────

def normalize_input(raw: RawInput) -> NormalizedInput:
    text = raw.text.strip()
    text = re.sub(r"\s+", " ", text)
    tokens = tuple(tokenize(text))
    sentences = tuple(segment_sentences(text))
    return NormalizedInput(original=raw, text=text, tokens=tokens, sentences=sentences)
