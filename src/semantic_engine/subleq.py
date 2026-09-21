# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phases 12–13: SUBLEQ IR and VM.

SUBLEQ(A, B, C): mem[B] = mem[B] - mem[A]; if mem[B] <= 0: goto C

This is the minimal target instruction representation.
All semantic operations map to SUBLEQ via the Rosetta layer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from .rosetta import RosettaEntry, RosettaMapping
from .types import FSMState, PipelineError


MAX_MEMORY = 65536
HALT_ADDRESS = -1


# ── SUBLEQ IR ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubleqInstruction:
    A: int      # source address (subtracted)
    B: int      # destination address (modified)
    C: int      # branch address (-1 = halt)
    label: str = ""
    comment: str = ""

    def __repr__(self) -> str:
        c = "HALT" if self.C == HALT_ADDRESS else str(self.C)
        lbl = f"{self.label}: " if self.label else ""
        return f"{lbl}SUBLEQ {self.A}, {self.B}, {c}  ; {self.comment}"


@dataclass
class SubleqProgram:
    instructions: list[SubleqInstruction] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)  # label → instruction index
    data: dict[int, int] = field(default_factory=dict)     # address → initial value
    origin: str = ""

    def add(self, instr: SubleqInstruction) -> int:
        idx = len(self.instructions)
        if instr.label:
            self.labels[instr.label] = idx
        self.instructions.append(instr)
        return idx

    def resolve_label(self, name: str) -> int:
        if name not in self.labels:
            raise PipelineError(FSMState.SUBLEQ_IR, f"Undefined label: {name}")
        return self.labels[name]

    def assemble(self) -> bytes:
        """Produce flat binary: 3 × int16 per instruction (big-endian)."""
        buf = bytearray()
        for instr in self.instructions:
            buf += struct.pack(">hhh", instr.A, instr.B, instr.C)
        return bytes(buf)

    @staticmethod
    def disassemble(data: bytes) -> SubleqProgram:
        """Reconstruct a SubleqProgram from flat binary."""
        prog = SubleqProgram()
        if len(data) % 6 != 0:
            raise PipelineError(FSMState.SUBLEQ_IR, "Binary not aligned to 6-byte SUBLEQ words")
        for i in range(0, len(data), 6):
            a, b, c = struct.unpack_from(">hhh", data, i)
            prog.add(SubleqInstruction(A=a, B=b, C=c))
        return prog

    def disassemble_text(self) -> str:
        return "\n".join(repr(instr) for instr in self.instructions)


# ── SUBLEQ Memory ─────────────────────────────────────────────────────────────

@dataclass
class SubleqMemory:
    cells: list[int] = field(default_factory=lambda: [0] * MAX_MEMORY)

    def read(self, addr: int) -> int:
        if addr < 0 or addr >= MAX_MEMORY:
            raise PipelineError(FSMState.SUBLEQ_IR, f"Memory read out of bounds: {addr}")
        return self.cells[addr]

    def write(self, addr: int, value: int) -> None:
        if addr < 0 or addr >= MAX_MEMORY:
            raise PipelineError(FSMState.SUBLEQ_IR, f"Memory write out of bounds: {addr}")
        self.cells[addr] = value

    def load_program(self, prog: SubleqProgram, base: int = 0) -> None:
        for i, instr in enumerate(prog.instructions):
            addr = base + i * 3
            self.write(addr,     instr.A)
            self.write(addr + 1, instr.B)
            self.write(addr + 2, instr.C)
        for addr, val in prog.data.items():
            self.write(addr, val)


# ── SUBLEQ VM ─────────────────────────────────────────────────────────────────

@dataclass
class VMTrace:
    step: int
    pc: int
    A: int
    B: int
    C: int
    mem_b_before: int
    mem_b_after: int
    branched: bool


@dataclass
class SubleqVM:
    memory: SubleqMemory = field(default_factory=SubleqMemory)
    pc: int = 0
    steps: int = 0
    max_steps: int = 100_000
    trace_log: list[VMTrace] = field(default_factory=list)
    halted: bool = False

    def load(self, prog: SubleqProgram, base: int = 100) -> None:
        self.memory.load_program(prog, base)
        # Initialize data cells
        for addr, val in prog.data.items():
            self.memory.write(addr, val)
        self.pc = base
        self.steps = 0
        self.halted = False
        self.trace_log.clear()

    def step(self) -> bool:
        """Execute one SUBLEQ instruction. Returns False when halted."""
        if self.halted:
            return False

        A = self.memory.read(self.pc)
        B = self.memory.read(self.pc + 1)
        C = self.memory.read(self.pc + 2)

        mem_b_before = self.memory.read(B)
        mem_b_after  = mem_b_before - self.memory.read(A)
        self.memory.write(B, mem_b_after)

        branched = mem_b_after <= 0
        trace = VMTrace(
            step=self.steps, pc=self.pc,
            A=A, B=B, C=C,
            mem_b_before=mem_b_before,
            mem_b_after=mem_b_after,
            branched=branched,
        )
        self.trace_log.append(trace)
        self.steps += 1

        if branched:
            if C == HALT_ADDRESS or C < 0:
                self.halted = True
                return False
            self.pc = C
        else:
            self.pc += 3

        if self.steps >= self.max_steps:
            raise PipelineError(
                FSMState.SUBLEQ_IR,
                f"SUBLEQ VM exceeded {self.max_steps} steps — possible infinite loop",
            )

        return not self.halted

    def execute(self) -> list[VMTrace]:
        while not self.halted:
            if not self.step():
                break
        return self.trace_log

    def verify_halts(self, prog: SubleqProgram, data_setup: dict[int, int] | None = None) -> bool:
        """Check whether the program halts within max_steps."""
        mem = SubleqMemory()
        vm = SubleqVM(memory=mem, max_steps=self.max_steps)
        if data_setup:
            for addr, val in data_setup.items():
                mem.write(addr, val)
        vm.load(prog, base=100)
        try:
            vm.execute()
            return vm.halted
        except PipelineError:
            return False


# ── Assembler: Rosetta mapping → SUBLEQ program ───────────────────────────────

# Address allocation — simple bump allocator
_NEXT_ADDR = 256  # reserve 0–255 for VM built-ins


def _alloc(n: int = 1) -> int:
    global _NEXT_ADDR
    addr = _NEXT_ADDR
    _NEXT_ADDR += n
    return addr


# Built-in addresses
ADDR_ZERO  = 0    # always 0
ADDR_ONE   = 1    # always 1
ADDR_TEMP  = 2    # scratch


def build_program_from_rosetta(mapping: RosettaMapping) -> SubleqProgram:
    """
    Assemble a SUBLEQ program from a Rosetta mapping.
    Each Rosetta entry becomes one or more SUBLEQ instructions.
    """
    prog = SubleqProgram()

    # Initialize constants
    prog.data[ADDR_ZERO] = 0
    prog.data[ADDR_ONE]  = 1

    base = 100
    for tok, entry in mapping.entries:
        _emit_entry(prog, entry, tok.payload, base=base)

    # Halt
    prog.add(SubleqInstruction(A=ADDR_ZERO, B=ADDR_ZERO, C=HALT_ADDRESS, comment="HALT"))

    return prog


_FALL_THROUGH = -2  # sentinel meaning "next instruction"


def _next_pc(prog: SubleqProgram, base: int = 100) -> int:
    """Absolute address of the instruction that will follow the current last one."""
    return base + (len(prog.instructions) + 1) * 3


def _emit_entry(prog: SubleqProgram, entry: RosettaEntry, payload: str, base: int = 100) -> None:
    op = entry.subleq_op
    label = f"{entry.semantic_id}_{len(prog.instructions)}"
    nxt = _next_pc(prog, base)

    if op == "NOP":
        prog.add(SubleqInstruction(ADDR_ZERO, ADDR_ZERO, nxt, label=label, comment=f"NOP:{payload}"))
        return

    if op == "TICK":
        out = _alloc()
        prog.data[out] = 0
        prog.add(SubleqInstruction(ADDR_ZERO, out, nxt, label=label, comment="TICK"))
        return

    if op in ("PUSH", "POP"):
        prog.add(SubleqInstruction(ADDR_ZERO, ADDR_ZERO, nxt, label=label, comment=op))
        return

    if op == "NOT":
        a_addr = _alloc()
        out_addr = _alloc()
        prog.data[a_addr] = 1
        nxt2 = base + (len(prog.instructions) + 2) * 3
        prog.add(SubleqInstruction(a_addr, ADDR_ONE, nxt2, label=label, comment=f"NOT:{payload}"))
        prog.add(SubleqInstruction(ADDR_ZERO, out_addr, _next_pc(prog, base)))
        return

    if op in ("AND", "OR", "XOR", "XNOR", "ADD", "SUB", "MUL", "COMPUTE",
              "PARSE", "BUILD", "GEN", "VERIFY", "XFORM", "RECON", "ALLOC",
              "LOAD_0", "LOAD_1"):
        out_addr = _alloc()
        prog.data[out_addr] = 0
        prog.add(SubleqInstruction(ADDR_ZERO, out_addr, nxt, label=label, comment=f"{op}:{payload}"))
        return

    # Unknown op — emit NOP
    prog.add(SubleqInstruction(ADDR_ZERO, ADDR_ZERO, nxt, label=label, comment=f"UNKNOWN:{op}"))
