# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 5: Geometry engine — explicit dimensional constraints for operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import FSMState, GeometryError
from .ir import Geometry, SemanticNode


@dataclass
class DimCheck:
    """Result of a dimensional compatibility check."""
    ok: bool
    reason: str
    output: Optional[Geometry] = None


def check_matmul(a: Geometry, b: Geometry) -> DimCheck:
    """A ∈ R^(m×n), B ∈ R^(n×p) → C ∈ R^(m×p). Requires a.col == b.row."""
    if a.col is None or b.row is None:
        return DimCheck(ok=True, reason="underdetermined", output=Geometry())
    if a.col != b.row:
        return DimCheck(
            ok=False,
            reason=f"contraction mismatch: A.col={a.col} ≠ B.row={b.row}",
        )
    return DimCheck(
        ok=True,
        reason="compatible",
        output=Geometry(row=a.row, col=b.col, contraction=a.col),
    )


def check_elementwise(a: Geometry, b: Geometry, op: str) -> DimCheck:
    """Element-wise ops (add, subtract, XOR) require identical shapes."""
    if a.shape and b.shape:
        if a.shape != b.shape:
            return DimCheck(ok=False, reason=f"{op}: shape mismatch {a.shape} ≠ {b.shape}")
    if a.row is not None and b.row is not None and a.row != b.row:
        return DimCheck(ok=False, reason=f"{op}: row mismatch {a.row} ≠ {b.row}")
    if a.col is not None and b.col is not None and a.col != b.col:
        return DimCheck(ok=False, reason=f"{op}: col mismatch {a.col} ≠ {b.col}")
    return DimCheck(ok=True, reason="compatible", output=a)


def enforce_geometry(node: SemanticNode) -> Geometry:
    """
    Validate geometry for a semantic node's operation.
    Raises GeometryError if dimensions are incompatible.
    Returns the output geometry.
    """
    op = node.operation
    operands = node.operands

    if op is None:
        return node.geometry or Geometry()

    if op.name in ("multiply", "matmul"):
        if len(operands) < 2:
            return node.geometry or Geometry()  # underdetermined — not a violation
        a_geom = operands[0].geometry or Geometry()
        b_geom = operands[1].geometry or Geometry()
        result = check_matmul(a_geom, b_geom)
        if not result.ok:
            raise GeometryError(FSMState.GEOMETRY_BINDING, result.reason)
        return result.output or Geometry()

    if op.name in ("add", "subtract"):
        if len(operands) < 2:
            raise GeometryError(FSMState.GEOMETRY_BINDING, f"{op.name} requires 2 operands")
        a_geom = operands[0].geometry or Geometry()
        b_geom = operands[1].geometry or Geometry()
        result = check_elementwise(a_geom, b_geom, op.name)
        if not result.ok:
            raise GeometryError(FSMState.GEOMETRY_BINDING, result.reason)
        return result.output or Geometry()

    # Unknown op — pass through without geometry enforcement
    return node.geometry or Geometry()


def parse_geometry(spec: str) -> Geometry:
    """
    Parse a geometry spec string like "3x4", "(3,4)", "R^(3x4)", "scalar".
    Returns a Geometry instance.
    """
    import re
    spec = spec.strip()

    if spec.lower() in ("scalar", "1", "1x1"):
        return Geometry(row=1, col=1, shape=(1, 1))

    m = re.match(r"[Rr]\^?\(?(\d+)[x×,](\d+)\)?", spec)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        return Geometry(row=r, col=c, shape=(r, c))

    m = re.match(r"(\d+)[x×,](\d+)", spec)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        return Geometry(row=r, col=c, shape=(r, c))

    m = re.match(r"(\d+)", spec)
    if m:
        n = int(m.group(1))
        return Geometry(row=n, col=1, shape=(n,))

    return Geometry()
