# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 6: Matrix semantic layer — semantic operations represented as matrix structures.

Semantic matrix ≠ numerical matrix.
A SemanticMatrix captures the structure of an operation (operand slots, geometry,
relationships) in a matrix-like form. Matrix data and semantic matrix are distinct types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .ir import Geometry, Operand, SemanticNode, SemanticOperation
from .types import FSMState, GeometryError, SemanticNodeType


# ── Semantic matrix cell ──────────────────────────────────────────────────────

@dataclass
class SemanticCell:
    row: int
    col: int
    operand: Optional[Operand] = None
    label: str = ""
    role: str = ""          # "input", "output", "contraction", "constraint"


# ── Semantic matrix ───────────────────────────────────────────────────────────

@dataclass
class SemanticMatrix:
    operation: SemanticOperation
    rows: int
    cols: int
    cells: list[list[SemanticCell]] = field(default_factory=list)
    geometry: Optional[Geometry] = None
    node_id: str = ""

    def __post_init__(self):
        if not self.cells:
            self.cells = [
                [SemanticCell(row=r, col=c) for c in range(self.cols)]
                for r in range(self.rows)
            ]

    def cell(self, row: int, col: int) -> SemanticCell:
        if row >= self.rows or col >= self.cols:
            raise IndexError(f"SemanticMatrix[{row},{col}] out of bounds ({self.rows}x{self.cols})")
        return self.cells[row][col]

    def set_cell(self, row: int, col: int, operand: Optional[Operand], role: str, label: str = "") -> None:
        c = self.cell(row, col)
        c.operand = operand
        c.role = role
        c.label = label

    def column(self, col: int) -> list[SemanticCell]:
        return [self.cells[r][col] for r in range(self.rows)]

    def row_vec(self, row: int) -> list[SemanticCell]:
        return list(self.cells[row])

    def to_dict(self) -> dict:
        return {
            "operation": self.operation.name,
            "shape": (self.rows, self.cols),
            "geometry": {
                "row": self.geometry.row if self.geometry else None,
                "col": self.geometry.col if self.geometry else None,
            },
            "cells": [
                [{"row": c.row, "col": c.col, "role": c.role, "label": c.label} for c in row]
                for row in self.cells
            ],
        }


# ── Builder ───────────────────────────────────────────────────────────────────

class SemanticMatrixBuilder:
    """Construct a SemanticMatrix from a SemanticNode."""

    def build(self, node: SemanticNode) -> SemanticMatrix:
        op = node.operation
        if op is None:
            op = SemanticOperation(name="identity", arity=1)

        operands = node.operands
        n_ops = len(operands)

        if op.name in ("multiply", "matmul"):
            return self._build_matmul(op, operands, node.geometry)

        if op.name in ("add", "subtract"):
            return self._build_binary(op, operands, node.geometry)

        # Generic: one row per operand, two cols (input / output)
        mat = SemanticMatrix(operation=op, rows=max(n_ops, 1), cols=2, geometry=node.geometry)
        for i, operand in enumerate(operands):
            mat.set_cell(i, 0, operand, role="input", label=operand.name)
        return mat

    def _build_matmul(
        self, op: SemanticOperation, operands: list[Operand], geom: Optional[Geometry]
    ) -> SemanticMatrix:
        """
        Matrix multiply: A(m×n) @ B(n×p) → C(m×p)
        Represent as 3x3:
          [A_input] [contract] [B_input]
          [m_dim]   [n_dim]   [p_dim]
          [output]  [output]  [output]
        """
        mat = SemanticMatrix(operation=op, rows=3, cols=3, geometry=geom)
        a = operands[0] if len(operands) > 0 else None
        b = operands[1] if len(operands) > 1 else None

        mat.set_cell(0, 0, a, role="input", label=a.name if a else "A")
        mat.set_cell(0, 2, b, role="input", label=b.name if b else "B")
        mat.set_cell(0, 1, None, role="contraction", label="n")

        if geom:
            mat.set_cell(1, 0, None, role="dim", label=str(geom.row or "m"))
            mat.set_cell(1, 1, None, role="dim", label=str(geom.contraction or "n"))
            mat.set_cell(1, 2, None, role="dim", label=str(geom.col or "p"))

        mat.set_cell(2, 0, None, role="output", label="C")
        mat.set_cell(2, 1, None, role="output", label="(m×p)")

        return mat

    def _build_binary(
        self, op: SemanticOperation, operands: list[Operand], geom: Optional[Geometry]
    ) -> SemanticMatrix:
        mat = SemanticMatrix(operation=op, rows=2, cols=2, geometry=geom)
        a = operands[0] if len(operands) > 0 else None
        b = operands[1] if len(operands) > 1 else None
        mat.set_cell(0, 0, a, role="input", label=a.name if a else "A")
        mat.set_cell(0, 1, b, role="input", label=b.name if b else "B")
        mat.set_cell(1, 0, None, role="output", label="result")
        return mat


# ── Transform interface ───────────────────────────────────────────────────────

def transform(a: SemanticMatrix, b: SemanticMatrix) -> SemanticMatrix:
    """
    Compose two semantic matrices — analogous to function composition.
    a: R^(m×n), b: R^(n×p) → R^(m×p)
    """
    if a.cols != b.rows:
        raise GeometryError(
            FSMState.MATRIX_FORM,
            f"SemanticMatrix composition: a.cols={a.cols} ≠ b.rows={b.rows}",
        )
    composed_op = SemanticOperation(
        name=f"{a.operation.name}∘{b.operation.name}",
        arity=a.operation.arity,
    )
    out_geom: Optional[Geometry] = None
    if a.geometry and b.geometry:
        out_geom = a.geometry.output_geometry(b.geometry)

    mat = SemanticMatrix(operation=composed_op, rows=a.rows, cols=b.cols, geometry=out_geom)
    # Populate diagonal — first operands of a flow into outputs of b
    for r in range(min(a.rows, b.rows)):
        src = a.cells[r][0]
        mat.set_cell(r, 0, src.operand, role="composed", label=src.label)
    return mat
