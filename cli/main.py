#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 18: CLI — semantic-engine command line interface."""

import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from semantic_engine import SemanticPipeline, PipelineResult, RawInput, OutputMode
from semantic_engine.nlp import normalize_input
from semantic_engine.boolean_engine import semantic_constraints_to_boolean, normalize as bool_normalize
from semantic_engine.token_engine import BoolEncoder, encode_binary, decode, encode
from semantic_engine.latin import LatinCanonicalizer
from semantic_engine.lexer import LatinLexer
from semantic_engine.rosetta import RosettaMapper
from semantic_engine.subleq import build_program_from_rosetta
from semantic_engine.verifier import Verifier


def read_input(path: str) -> RawInput:
    p = Path(path)
    if not p.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return RawInput(text=p.read_text().strip(), source=str(p))


def cmd_parse(args):
    raw = read_input(args.input)
    normalized = normalize_input(raw)
    out = {
        "input_id": raw.input_id,
        "input_hash": raw.hash,
        "text": normalized.text,
        "tokens": list(normalized.tokens),
        "sentences": list(normalized.sentences),
    }
    print(json.dumps(out, indent=2) if args.json else f"Tokens: {list(normalized.tokens)}")


def cmd_inspect(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    if result.ir:
        ir = result.ir
        out = {
            "ir_id": ir.ir_id,
            "intent": ir.intent.value,
            "nodes": len(ir.nodes),
            "verification": ir.verification_state.name,
            "hash": ir.hash(),
        }
        print(json.dumps(out, indent=2) if args.json else str(out))
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_normalize(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    if result.ir:
        from semantic_engine.boolean_engine import semantic_constraints_to_boolean
        bnf = semantic_constraints_to_boolean(result.ir.root.constraints)
        print(f"Normalized: {bnf.normalized}")
        print(f"Stable:     {bnf.stable}")
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_boolean(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    if result.ir:
        bnf = semantic_constraints_to_boolean(result.ir.root.constraints)
        print(f"Boolean form: {bnf.normalized}")
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_tokens(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    if result.ir:
        bnf = semantic_constraints_to_boolean(result.ir.root.constraints)
        encoder = BoolEncoder()
        stream = encoder.encode_expr(bnf.normalized)
        for tok in stream:
            print(repr(tok))
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_latin(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    if result.ir:
        bnf = semantic_constraints_to_boolean(result.ir.root.constraints)
        encoder = BoolEncoder()
        stream = encoder.encode_expr(bnf.normalized)
        latin = LatinCanonicalizer().canonicalize(stream)
        print(latin.text())
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_subleq(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    if result.subleq_listing:
        print(result.subleq_listing)
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_verify(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)
    status = "VERIFIED" if result.ok else "REJECTED"
    print(f"Status: {status}")
    if result.error:
        print(f"Error:  {result.error}")
    for line in result.trace:
        print(f"  {line}")


def cmd_artifact(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw, output_mode=OutputMode.NATURAL_LANGUAGE)
    if result.artifact:
        print(result.artifact.content)
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_backend(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw, output_mode=OutputMode.CODE)
    if result.artifact:
        print(result.artifact.content)
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    raw = read_input(args.input)
    pipeline = SemanticPipeline()
    result = pipeline.run(raw)

    if args.trace:
        print("=== TRACE ===")
        for line in result.trace:
            print(f"  {line}")
        print()

    if args.subleq:
        print("=== SUBLEQ ===")
        print(result.subleq_listing)
        print()

    if args.provenance:
        print("=== PROVENANCE ===")
        if result.ir:
            print(f"  IR: {result.ir.ir_id}")
            print(f"  Stage: {result.ir.provenance.stage}")
        print()

    if args.tokens and result.ir:
        print("=== TOKENS ===")
        from semantic_engine.boolean_engine import semantic_constraints_to_boolean
        bnf = semantic_constraints_to_boolean(result.ir.root.constraints)
        encoder = BoolEncoder()
        stream = encoder.encode_expr(bnf.normalized)
        for tok in stream:
            print(f"  {repr(tok)}")
        print()

    if result.artifact:
        print("=== OUTPUT ===")
        print(result.artifact.content)
    else:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)

    print(f"\nStatus: {'VERIFIED' if result.ok else 'REJECTED'}")
    for k, v in result.hashes.items():
        print(f"  {k}: {v[:16]}")


def main():
    parser = argparse.ArgumentParser(
        prog="semantic-engine",
        description="Deterministic semantic transformation pipeline",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_text in [
        ("parse",     cmd_parse,     "Tokenize and segment input"),
        ("inspect",   cmd_inspect,   "Build and inspect the Semantic IR"),
        ("normalize", cmd_normalize, "Show Boolean normalization"),
        ("boolean",   cmd_boolean,   "Show Boolean IR"),
        ("tokens",    cmd_tokens,    "Show Boolean token stream"),
        ("latin",     cmd_latin,     "Show Latin canonical form"),
        ("subleq",    cmd_subleq,    "Show SUBLEQ IR listing"),
        ("verify",    cmd_verify,    "Run full verification and show results"),
        ("artifact",  cmd_artifact,  "Emit natural-language artifact"),
        ("backend",   cmd_backend,   "Emit backend source artifact"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("input", help="Path to input file")
        p.set_defaults(func=fn)

    run_p = sub.add_parser("run", help="Run full pipeline with optional trace flags")
    run_p.add_argument("input")
    run_p.add_argument("--trace",      action="store_true")
    run_p.add_argument("--subleq",     action="store_true")
    run_p.add_argument("--tokens",     action="store_true")
    run_p.add_argument("--provenance", action="store_true")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
