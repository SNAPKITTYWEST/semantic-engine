# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 4: Temporal lock — every transformation receives and preserves τ."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TypeVar, Callable

from .types import FSMState, TemporalViolation
from .ir import TemporalLock, TemporalState, TemporalTransition

T = TypeVar("T")


def stamp() -> int:
    """Monotonic integer tick based on nanoseconds."""
    return time.monotonic_ns()


def apply_temporal(
    fn: Callable[..., T],
    t_in: TemporalState,
    lock: TemporalLock,
    stage: str,
    *args,
    **kwargs,
) -> tuple[T, TemporalState]:
    """
    Run fn(*args, **kwargs) under a temporal contract.

    - Records τn → τn+1 transition.
    - Raises TemporalViolation if the state was locked.
    - Returns (result, new_temporal_state).
    """
    if t_in.locked:
        raise TemporalViolation(
            FSMState.TEMPORAL_LOCK,
            f"Stage '{stage}' attempted to transform a locked temporal state τ{t_in.tau}",
        )

    result = fn(*args, **kwargs)
    t_out = t_in.advance(reason=stage)

    lock.record(TemporalTransition(
        from_tau=t_in.tau,
        to_tau=t_out.tau,
        reason=stage,
        stage=stage,
    ))

    return result, t_out


def preserve_temporal(
    fn: Callable[..., T],
    t_in: TemporalState,
    lock: TemporalLock,
    stage: str,
    *args,
    **kwargs,
) -> tuple[T, TemporalState]:
    """
    Run fn but preserve τ (no advance). Used for read-only passes.
    Records a τ→τ no-op transition.
    """
    result = fn(*args, **kwargs)
    lock.record(TemporalTransition(
        from_tau=t_in.tau,
        to_tau=t_in.tau,
        reason=f"preserve:{stage}",
        stage=stage,
    ))
    return result, t_in


def assert_temporal_invariant(t_before: TemporalState, t_after: TemporalState, stage: str) -> None:
    """Assert τ_after == τ_before + 1 (for transforms) or == τ_before (for reads)."""
    if t_after.tau not in (t_before.tau, t_before.tau + 1):
        raise TemporalViolation(
            FSMState.TEMPORAL_LOCK,
            f"Stage '{stage}' illegally mutated τ: {t_before.tau} → {t_after.tau}",
        )
