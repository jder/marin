# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from experiments.probabilistic_dataflow.documents import AttentionLayout, Document, Output, OutputSlot, Record
from experiments.probabilistic_dataflow.programs import (
    SAMPLED_FEEDBACK,
    DocumentProgram,
    DocumentRequest,
    DocumentResponse,
    parallel_programs,
)

QUERY_TOKEN = 0
DETAILED_PLAN_TOKEN = 10
ACCEPT_TOKEN = 20
REJECT_TOKEN = 21


@dataclass(frozen=True)
class InspectionPlan:
    refine_geometry: bool


@dataclass(frozen=True)
class CompositionSolution:
    plan: InspectionPlan
    geometry_token: int
    chemistry_token: int
    accepted: bool
    retries: int


@dataclass
class SpecialistResources:
    """Observable resources held while adaptive specialists are suspended."""

    active: set[str] = field(default_factory=set)
    acquired: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)

    @contextmanager
    def acquire(self, name: str) -> Iterator[None]:
        if name in self.active:
            raise ValueError(f"Specialist resource {name!r} is already active")
        self.active.add(name)
        self.acquired.append(name)
        try:
            yield
        finally:
            self.active.remove(name)
            self.released.append(name)


def planning_program(example_id: str) -> DocumentProgram[InspectionPlan]:
    """Choose whether the geometry specialist should refine its proposal."""
    slot = OutputSlot(example_id, "plan", 0)
    response = yield DocumentRequest(
        f"{example_id}/planning",
        (_query_document(f"{example_id}/planning", slot, ()),),
        SAMPLED_FEEDBACK,
    )
    return InspectionPlan(refine_geometry=_single_token(response) == DETAILED_PLAN_TOKEN)


def geometry_program(
    example_id: str,
    plan: InspectionPlan,
    resources: SpecialistResources,
) -> DocumentProgram[int]:
    """Predict geometry, optionally using an adaptive second document."""
    slot = OutputSlot(example_id, "geometry", 0)
    with resources.acquire("geometry"):
        response = yield DocumentRequest(
            f"{example_id}/geometry/coarse",
            (_query_document(f"{example_id}/geometry/coarse", slot, ()),),
            SAMPLED_FEEDBACK,
        )
        geometry_token = _single_token(response)
        if not plan.refine_geometry:
            return geometry_token

        response = yield DocumentRequest(
            f"{example_id}/geometry/refine",
            (_query_document(f"{example_id}/geometry/refine", slot, (geometry_token,)),),
            SAMPLED_FEEDBACK,
        )
        return _single_token(response)


def chemistry_program(example_id: str, resources: SpecialistResources) -> DocumentProgram[int]:
    """Predict chemistry in one document while geometry adapts independently."""
    slot = OutputSlot(example_id, "chemistry", 0)
    with resources.acquire("chemistry"):
        response = yield DocumentRequest(
            f"{example_id}/chemistry",
            (_query_document(f"{example_id}/chemistry", slot, ()),),
            SAMPLED_FEEDBACK,
        )
        return _single_token(response)


def verification_program(
    example_id: str,
    geometry_token: int,
    chemistry_token: int,
    *,
    attempt: int,
) -> DocumentProgram[bool]:
    """Check a pair of specialist predictions in a separate document."""
    slot = OutputSlot(example_id, "accepted", attempt)
    response = yield DocumentRequest(
        f"{example_id}/verification/{attempt}",
        (
            _query_document(
                f"{example_id}/verification/{attempt}",
                slot,
                (geometry_token, chemistry_token),
            ),
        ),
        SAMPLED_FEEDBACK,
    )
    return _single_token(response) == ACCEPT_TOKEN


def retry_geometry_program(
    example_id: str,
    previous_geometry_token: int,
    chemistry_token: int,
    resources: SpecialistResources,
) -> DocumentProgram[int]:
    """Retry rejected geometry with the chemistry prediction as added context."""
    slot = OutputSlot(example_id, "geometry", 0)
    with resources.acquire("geometry"):
        response = yield DocumentRequest(
            f"{example_id}/geometry/retry",
            (
                _query_document(
                    f"{example_id}/geometry/retry",
                    slot,
                    (previous_geometry_token, chemistry_token),
                ),
            ),
            SAMPLED_FEEDBACK,
        )
        return _single_token(response)


def composition_program(example_id: str, resources: SpecialistResources) -> DocumentProgram[CompositionSolution]:
    """Plan, mix adaptive specialists, verify, and conditionally retry."""
    plan = yield from planning_program(example_id)
    geometry_token, chemistry_token = yield from parallel_programs(
        (
            geometry_program(example_id, plan, resources),
            chemistry_program(example_id, resources),
        ),
        request_prefix=f"{example_id}/specialists",
    )
    accepted = yield from verification_program(
        example_id,
        geometry_token,
        chemistry_token,
        attempt=0,
    )
    if accepted:
        return CompositionSolution(plan, geometry_token, chemistry_token, accepted=True, retries=0)

    geometry_token = yield from retry_geometry_program(
        example_id,
        geometry_token,
        chemistry_token,
        resources,
    )
    accepted = yield from verification_program(
        example_id,
        geometry_token,
        chemistry_token,
        attempt=1,
    )
    return CompositionSolution(plan, geometry_token, chemistry_token, accepted=accepted, retries=1)


def _query_document(document_id: str, slot: OutputSlot, context_tokens: tuple[int, ...]) -> Document:
    records = tuple(Record(token, position_id=0) for token in context_tokens)
    query = Record(QUERY_TOKEN, position_id=0, output=Output(slot))
    return Document(document_id, (*records, query), AttentionLayout.FULL)


def _single_token(response: DocumentResponse) -> int:
    observations = response.observations
    if len(observations) != 1:
        raise ValueError(f"Expected one prediction observation, got {len(observations)}")
    return observations[0].token_id
