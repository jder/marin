# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from experiments.probabilistic_dataflow.documents import QUERY, AttentionLayout, Coordinate, Document
from experiments.probabilistic_dataflow.programs import Program, Result, parallel

QUERY_TOKEN = 0
DETAILED_PLAN_TOKEN = 10
ACCEPT_TOKEN = 20
REJECT_TOKEN = 21

TASK = Coordinate("task")
ATTEMPT = Coordinate("attempt")
STAGE = Coordinate("stage")
PLAN_TASK = 0
GEOMETRY_TASK = 1
CHEMISTRY_TASK = 2
VERIFICATION_TASK = 3
COARSE_STAGE = 0
REFINE_STAGE = 1
RETRY_STAGE = 2


@dataclass(frozen=True)
class CompositionSolution:
    refine_geometry: bool
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


def planning_program() -> Program[bool]:
    """Choose whether the geometry specialist should refine its proposal."""
    results = yield (_query_document(PLAN_TASK, ()),)
    return _single_token(results) == DETAILED_PLAN_TOKEN


def geometry_program(
    refine_geometry: bool,
    resources: SpecialistResources,
) -> Program[int]:
    """Predict geometry, optionally using an adaptive second document."""
    with resources.acquire("geometry"):
        results = yield (_query_document(GEOMETRY_TASK, (), extra_coordinates=((STAGE, COARSE_STAGE),)),)
        geometry_token = _single_token(results)
        if not refine_geometry:
            return geometry_token

        results = yield (_query_document(GEOMETRY_TASK, (geometry_token,), extra_coordinates=((STAGE, REFINE_STAGE),)),)
        return _single_token(results)


def chemistry_program(resources: SpecialistResources) -> Program[int]:
    """Predict chemistry in one document while geometry adapts independently."""
    with resources.acquire("chemistry"):
        results = yield (_query_document(CHEMISTRY_TASK, ()),)
        return _single_token(results)


def verification_program(
    geometry_token: int,
    chemistry_token: int,
    *,
    attempt: int,
) -> Program[bool]:
    """Check a pair of specialist predictions in a separate document."""
    results = yield (
        _query_document(
            VERIFICATION_TASK,
            (geometry_token, chemistry_token),
            extra_coordinates=((ATTEMPT, attempt),),
        ),
    )
    return _single_token(results) == ACCEPT_TOKEN


def retry_geometry_program(
    previous_geometry_token: int,
    chemistry_token: int,
    resources: SpecialistResources,
) -> Program[int]:
    """Retry rejected geometry with the chemistry prediction as added context."""
    with resources.acquire("geometry"):
        results = yield (
            _query_document(
                GEOMETRY_TASK,
                (previous_geometry_token, chemistry_token),
                extra_coordinates=((STAGE, RETRY_STAGE),),
            ),
        )
        return _single_token(results)


def composition_program(resources: SpecialistResources) -> Program[CompositionSolution]:
    """Plan, mix adaptive specialists, verify, and conditionally retry."""
    refine_geometry = yield from planning_program()
    geometry_token, chemistry_token = yield from parallel(
        (
            geometry_program(refine_geometry, resources),
            chemistry_program(resources),
        )
    )
    accepted = yield from verification_program(
        geometry_token,
        chemistry_token,
        attempt=0,
    )
    if accepted:
        return CompositionSolution(refine_geometry, geometry_token, chemistry_token, accepted=True, retries=0)

    geometry_token = yield from retry_geometry_program(
        geometry_token,
        chemistry_token,
        resources,
    )
    accepted = yield from verification_program(
        geometry_token,
        chemistry_token,
        attempt=1,
    )
    return CompositionSolution(refine_geometry, geometry_token, chemistry_token, accepted=accepted, retries=1)


def _query_document(
    task_id: int,
    context_tokens: tuple[int, ...],
    *,
    extra_coordinates: tuple[tuple[Coordinate, int], ...] = (),
) -> Document:
    query = Document(
        (QUERY_TOKEN,),
        {
            TASK: (task_id,),
            QUERY: (True,),
            **{coordinate: (value,) for coordinate, value in extra_coordinates},
        },
        attention=AttentionLayout.FULL,
    )
    if not context_tokens:
        return query
    return Document(context_tokens, attention=AttentionLayout.FULL) + query


def _single_token(results: tuple[Result, ...]) -> int:
    if len(results) != 1 or len(results[0].predictions) != 1:
        raise ValueError("Expected one document containing one prediction")
    return results[0].predictions[0].token_id
