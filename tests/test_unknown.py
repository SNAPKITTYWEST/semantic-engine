# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: ADR-Ω-0001 — The Uninterpreted Operator and WORM Seal Engine."""

import pytest
from semantic_engine.unknown import (
    InvariantState, ObservationRecord, UnknownClassification,
    WORMSealChain, WORMSealEntry, UnknownNode, UnknownOperatorError,
    observe_and_seal, _seal_hash, _canonical,
)


# ── InvariantState ────────────────────────────────────────────────────────────

def test_invariant_hash_deterministic():
    omega = InvariantState(tau=0, system_hash="abc123", properties=("typed", "deterministic"))
    assert omega.hash() == omega.hash()


def test_invariant_hash_changes_with_tau():
    o1 = InvariantState(tau=0, system_hash="abc")
    o2 = InvariantState(tau=1, system_hash="abc")
    assert o1.hash() != o2.hash()


# ── ObservationRecord ─────────────────────────────────────────────────────────

def make_omega(tau=0):
    return InvariantState(tau=tau, system_hash="test_hash")


def test_observation_record_hash_valid():
    omega = make_omega()
    rec = ObservationRecord.observe("input_A", "output_B", omega)
    assert rec.verify()


def test_observation_record_fabricated_hash_fails():
    omega = make_omega()
    rec = ObservationRecord.observe("input_A", "output_B", omega)
    # Tamper with provenance hash
    tampered = ObservationRecord(
        input_state=rec.input_state,
        output_state=rec.output_state,
        provenance_hash="deadbeef" * 8,
        timestamp=rec.timestamp,
        invariant=rec.invariant,
        classification=rec.classification,
    )
    assert not tampered.verify()


def test_observation_starts_as_observed():
    rec = ObservationRecord.observe("A", "B", make_omega())
    assert rec.classification == UnknownClassification.OBSERVED


def test_classification_pipeline_valid():
    rec = ObservationRecord.observe("A", "B", make_omega())
    rec = rec.advance(UnknownClassification.CAPTURED)
    rec = rec.advance(UnknownClassification.HASHED)
    rec = rec.advance(UnknownClassification.COMPARED)
    rec = rec.advance(UnknownClassification.TESTED)
    rec = rec.advance(UnknownClassification.CLASSIFIED)
    assert rec.classification == UnknownClassification.CLASSIFIED


def test_classification_skip_not_allowed():
    """UNKNOWN must not become UNDERSTOOD by skipping stages."""
    rec = ObservationRecord.observe("A", "B", make_omega())
    with pytest.raises(UnknownOperatorError):
        rec.advance(UnknownClassification.CLASSIFIED)  # skip CAPTURED, HASHED, etc.


def test_classification_backward_not_allowed():
    rec = ObservationRecord.observe("A", "B", make_omega())
    rec = rec.advance(UnknownClassification.CAPTURED)
    with pytest.raises(UnknownOperatorError):
        rec.advance(UnknownClassification.OBSERVED)  # go backward


def test_unknown_is_not_true_or_false():
    """Fundamental property: UNKNOWN ≠ FALSE, UNKNOWN ≠ TRUE."""
    rec = ObservationRecord.observe("A", "B", make_omega())
    node = UnknownNode(node_id="test", observation=rec)
    assert not node.is_understood
    # The node never claims to know what ? means
    assert "?" in repr(node)


# ── Seal hash ─────────────────────────────────────────────────────────────────

def test_seal_hash_deterministic():
    omega = make_omega()
    h1 = _seal_hash("A", "B", omega)
    h2 = _seal_hash("A", "B", omega)
    assert h1 == h2


def test_seal_hash_changes_with_input():
    omega = make_omega()
    h1 = _seal_hash("A", "B", omega)
    h2 = _seal_hash("X", "B", omega)
    assert h1 != h2


def test_seal_hash_changes_with_output():
    omega = make_omega()
    h1 = _seal_hash("A", "B", omega)
    h2 = _seal_hash("A", "Z", omega)
    assert h1 != h2


def test_seal_hash_includes_omega():
    o1 = make_omega(tau=0)
    o2 = make_omega(tau=1)
    h1 = _seal_hash("A", "B", o1)
    h2 = _seal_hash("A", "B", o2)
    assert h1 != h2


# ── WORM Seal Chain ───────────────────────────────────────────────────────────

def test_worm_chain_starts_empty():
    chain = WORMSealChain()
    assert len(chain) == 0
    assert chain.head() is None


def test_worm_seal_appends():
    chain = WORMSealChain()
    rec = ObservationRecord.observe("A", "B", make_omega())
    entry = chain.seal(rec)
    assert len(chain) == 1
    assert entry.index == 0
    assert entry.previous_hash == "0" * 64


def test_worm_chain_links():
    chain = WORMSealChain()
    r1 = ObservationRecord.observe("A", "B", make_omega(0))
    r2 = ObservationRecord.observe("B", "C", make_omega(1))
    e1 = chain.seal(r1)
    e2 = chain.seal(r2)
    assert e2.previous_hash == e1.entry_hash
    assert e2.index == 1


def test_worm_chain_verify_intact():
    chain = WORMSealChain()
    for i in range(5):
        rec = ObservationRecord.observe(f"input_{i}", f"output_{i}", make_omega(i))
        chain.seal(rec)
    ok, errors = chain.verify_chain()
    assert ok
    assert errors == []


def test_worm_chain_audit_returns_all():
    chain = WORMSealChain()
    for i in range(3):
        rec = ObservationRecord.observe(f"A{i}", f"B{i}", make_omega(i))
        chain.seal(rec)
    audit = chain.audit()
    assert len(audit) == 3
    for entry in audit:
        assert "entry_hash" in entry
        assert "provenance_hash" in entry


def test_worm_seal_not_mutable():
    """Once sealed, the entry is frozen (frozen dataclass)."""
    chain = WORMSealChain()
    rec = ObservationRecord.observe("A", "B", make_omega())
    entry = chain.seal(rec)
    with pytest.raises((AttributeError, TypeError)):
        entry.entry_hash = "tampered"


def test_worm_export_import_jsonl(tmp_path):
    chain = WORMSealChain()
    for i in range(3):
        rec = ObservationRecord.observe(f"A{i}", f"B{i}", make_omega(i))
        chain.seal(rec)
    p = tmp_path / "chain.jsonl"
    chain.export_jsonl(p)
    rows = WORMSealChain.import_jsonl(p)
    assert len(rows) == 3
    assert rows[0]["index"] == 0
    assert rows[2]["index"] == 2


# ── observe_and_seal ──────────────────────────────────────────────────────────

def test_observe_and_seal_full_pipeline():
    chain = WORMSealChain()
    omega = make_omega(tau=5)
    node = observe_and_seal("my_input", "my_output", omega, chain)
    assert node.is_sealed
    assert not node.is_understood
    assert node.observation.classification == UnknownClassification.CLASSIFIED
    assert len(chain) == 1


def test_observe_and_seal_chain_verifies():
    chain = WORMSealChain()
    omega = make_omega()
    observe_and_seal("A", "B", omega, chain)
    observe_and_seal("B", "C", omega, chain)
    ok, errors = chain.verify_chain()
    assert ok


def test_observe_and_seal_repr():
    chain = WORMSealChain()
    node = observe_and_seal("x", "y", make_omega(), chain)
    r = repr(node)
    assert "?" in r
    assert "CLASSIFIED" in r
    assert "SEALED" in r


# ── Algebra invariants ────────────────────────────────────────────────────────

def test_unknown_not_equal_to_understood():
    """
    The fundamental ADR invariant:
    UNKNOWN ≠ FALSE
    UNKNOWN ≠ TRUE
    """
    rec = ObservationRecord.observe("A", "B", make_omega())
    node = UnknownNode(node_id="abc", observation=rec)
    assert node.is_understood is False
    # Trying to assert it's True or False is a bug — the node never claims either
    assert isinstance(node.is_understood, bool)
    assert node.is_understood is not True


def test_classification_terminal_at_classified():
    """Once CLASSIFIED, no further advancement is defined."""
    rec = ObservationRecord.observe("A", "B", make_omega())
    for step in [
        UnknownClassification.CAPTURED,
        UnknownClassification.HASHED,
        UnknownClassification.COMPARED,
        UnknownClassification.TESTED,
        UnknownClassification.CLASSIFIED,
    ]:
        rec = rec.advance(step)

    # No valid next step exists
    with pytest.raises(UnknownOperatorError):
        rec.advance(UnknownClassification.OBSERVED)
