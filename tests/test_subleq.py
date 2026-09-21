# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests: Phases 12-13 — SUBLEQ IR and VM."""

import pytest
from semantic_engine.subleq import (
    SubleqInstruction, SubleqProgram, SubleqMemory, SubleqVM,
    HALT_ADDRESS, MAX_MEMORY,
)
from semantic_engine.types import PipelineError


def halt_prog() -> SubleqProgram:
    prog = SubleqProgram()
    prog.data[0] = 0  # zero cell
    prog.add(SubleqInstruction(A=0, B=0, C=HALT_ADDRESS, comment="HALT"))
    return prog


def test_instruction_repr():
    instr = SubleqInstruction(A=1, B=2, C=3)
    r = repr(instr)
    assert "SUBLEQ" in r


def test_instruction_halt_repr():
    instr = SubleqInstruction(A=0, B=0, C=HALT_ADDRESS)
    assert "HALT" in repr(instr)


def test_program_add_and_label():
    prog = SubleqProgram()
    prog.add(SubleqInstruction(A=0, B=0, C=HALT_ADDRESS, label="start"))
    assert "start" in prog.labels
    assert prog.labels["start"] == 0


def test_program_resolve_label():
    prog = SubleqProgram()
    prog.add(SubleqInstruction(A=0, B=0, C=HALT_ADDRESS, label="end"))
    assert prog.resolve_label("end") == 0


def test_program_resolve_missing_label():
    prog = SubleqProgram()
    with pytest.raises(PipelineError):
        prog.resolve_label("missing")


def test_assemble_disassemble_roundtrip():
    prog = SubleqProgram()
    prog.add(SubleqInstruction(A=1, B=2, C=3))
    prog.add(SubleqInstruction(A=4, B=5, C=HALT_ADDRESS))
    binary = prog.assemble()
    recovered = SubleqProgram.disassemble(binary)
    assert len(recovered.instructions) == 2
    assert recovered.instructions[0].A == 1
    assert recovered.instructions[0].B == 2
    assert recovered.instructions[1].C == HALT_ADDRESS


def test_assemble_bad_alignment():
    with pytest.raises(PipelineError):
        SubleqProgram.disassemble(b"\x00\x01\x02\x03")


def test_memory_read_write():
    mem = SubleqMemory()
    mem.write(100, 42)
    assert mem.read(100) == 42


def test_memory_out_of_bounds():
    mem = SubleqMemory()
    with pytest.raises(PipelineError):
        mem.read(MAX_MEMORY + 1)
    with pytest.raises(PipelineError):
        mem.write(-1, 0)


def test_vm_halts():
    prog = halt_prog()
    vm = SubleqVM()
    vm.load(prog)
    vm.execute()
    assert vm.halted


def test_vm_step_returns_false_on_halt():
    prog = halt_prog()
    vm = SubleqVM()
    vm.load(prog)
    while not vm.halted:
        result = vm.step()
    assert not result


def test_vm_trace_recorded():
    prog = halt_prog()
    vm = SubleqVM()
    vm.load(prog)
    trace = vm.execute()
    assert len(trace) >= 1
    assert trace[0].pc >= 0  # PC starts at base offset


def test_vm_max_steps_exceeded():
    prog = SubleqProgram()
    # Infinite loop: SUBLEQ 0 0 0
    prog.data[0] = 0
    prog.add(SubleqInstruction(A=0, B=0, C=0))
    vm = SubleqVM(max_steps=10)
    vm.load(prog)
    with pytest.raises(PipelineError) as exc:
        vm.execute()
    assert "max" in str(exc.value).lower() or "steps" in str(exc.value).lower()


def test_verify_halts():
    prog = halt_prog()
    vm = SubleqVM()
    assert vm.verify_halts(prog)


def test_subleq_arithmetic():
    """SUBLEQ(A, B, C): mem[B] -= mem[A]; branch if <=0"""
    prog = SubleqProgram()
    # Use addresses 10, 11 for data (well below base=100 for instructions)
    prog.data[10] = 5
    prog.data[11] = 3
    prog.add(SubleqInstruction(A=10, B=11, C=HALT_ADDRESS))
    vm = SubleqVM()
    vm.load(prog)
    vm.execute()
    assert vm.memory.read(11) == -2
