# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Phase 17: Agent harness — agents propose artifacts; the system verifies before acceptance.

Agent output never bypasses FSM, IR validation, temporal lock, or verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .ir import Artifact, SemanticIR
from .types import ArtifactType, FSMState, PipelineError, VerificationState
from .verifier import Verifier


# ── Agent task ────────────────────────────────────────────────────────────────

@dataclass
class AgentTask:
    task_id: str
    description: str
    semantic_constraints: list[str]
    available_operations: list[str]
    artifact_schema: dict
    verification_requirements: list[str]
    ir_ref: Optional[SemanticIR] = None


# ── Agent proposal ────────────────────────────────────────────────────────────

@dataclass
class AgentProposal:
    agent_id: str
    task_id: str
    artifact: Artifact
    rationale: str = ""


# ── Agent interface ───────────────────────────────────────────────────────────

AgentFn = Callable[[AgentTask], AgentProposal]


@dataclass
class Agent:
    agent_id: str
    fn: AgentFn
    capabilities: list[ArtifactType] = field(default_factory=list)


# ── Harness ───────────────────────────────────────────────────────────────────

class AgentHarness:
    """
    Accepts agent proposals, runs them through the verifier, accepts or rejects.
    Agents may NOT bypass any pipeline stage.
    """

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._verifier = Verifier()
        self._accepted: list[Artifact] = []
        self._rejected: list[tuple[AgentProposal, str]] = []

    def register(self, agent: Agent) -> None:
        self._agents[agent.agent_id] = agent

    def dispatch(self, task: AgentTask, agent_id: str) -> Artifact:
        agent = self._agents.get(agent_id)
        if agent is None:
            raise PipelineError(FSMState.ARTIFACT_SELECTION, f"Unknown agent: {agent_id}")

        proposal = agent.fn(task)

        if task.ir_ref is None:
            raise PipelineError(
                FSMState.VERIFICATION,
                "Agent task has no IR reference — cannot verify provenance",
            )

        report = self._verifier.verify_artifact(proposal.artifact, task.ir_ref)

        if not report.passed:
            self._rejected.append((proposal, report.summary()))
            raise PipelineError(
                FSMState.VERIFICATION,
                f"Agent {agent_id} proposal rejected:\n{report.summary()}",
            )

        proposal.artifact.verification_state = VerificationState.VERIFIED
        self._accepted.append(proposal.artifact)
        return proposal.artifact

    def accepted(self) -> list[Artifact]:
        return list(self._accepted)

    def rejected(self) -> list[tuple[AgentProposal, str]]:
        return list(self._rejected)
