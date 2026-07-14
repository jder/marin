# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Generator
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Generic, Protocol, TypeVar, overload

import numpy as np

from experiments.probabilistic_dataflow.documents import (
    AttentionLayout,
    Document,
    OutputSlot,
    PackedDocuments,
    PredictionValue,
    pack_documents,
)

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class FeedbackOrigin(StrEnum):
    SAMPLED = "sampled"
    SUPERVISED = "supervised"
    CORRUPTED = "corrupted"


SAMPLED_FEEDBACK = frozenset({FeedbackOrigin.SAMPLED})
GENERATED_FEEDBACK = frozenset({FeedbackOrigin.SAMPLED, FeedbackOrigin.CORRUPTED})


class ProgramReplayError(ValueError):
    pass


@dataclass(frozen=True)
class PredictionObservation:
    """One uncommitted observation produced for a logical output slot."""

    slot: OutputSlot
    token_id: int
    logprob: float

    def __post_init__(self) -> None:
        if not isfinite(self.logprob) or self.logprob > 0:
            raise ValueError(f"Prediction logprob must be finite and non-positive, got {self.logprob}")

    def prediction_value(self) -> PredictionValue:
        return PredictionValue(self.slot, self.token_id)


@dataclass(frozen=True)
class DocumentResult:
    """Predictions for one document occurrence in request order.

    ``document_id`` is descriptive and may repeat. A response routes results
    positionally: ``results[i]`` satisfies ``request.documents[i]``.
    """

    document_id: str
    observations: tuple[PredictionObservation, ...]
    origin: FeedbackOrigin


@dataclass(frozen=True)
class DocumentRequest:
    """One barriered wave of documents that may execute in parallel."""

    id: str
    documents: tuple[Document, ...]
    accepted_origins: frozenset[FeedbackOrigin]

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("A document request requires at least one document")
        if not self.accepted_origins:
            raise ValueError("A document request requires at least one accepted feedback origin")


@dataclass(frozen=True)
class DocumentResponse:
    """Results for one request, in the same order as its documents."""

    request_id: str
    results: tuple[DocumentResult, ...]

    @property
    def observations(self) -> tuple[PredictionObservation, ...]:
        return tuple(observation for result in self.results for observation in result.observations)


DocumentProgram = Generator[DocumentRequest, DocumentResponse, T]


class DocumentExecutor(Protocol):
    """Execute one pending request from each active document program."""

    def __call__(self, requests: tuple[DocumentRequest, ...]) -> tuple[DocumentResponse, ...]: ...


@dataclass(frozen=True)
class PackedPredictions:
    token_ids: np.ndarray
    logprobs: np.ndarray


@dataclass(frozen=True)
class ProgramExchange:
    request: DocumentRequest
    response: DocumentResponse


@dataclass(frozen=True)
class ProgramRun(Generic[T]):
    value: T
    exchanges: tuple[ProgramExchange, ...]


@dataclass
class _ActiveProgram(Generic[T]):
    index: int
    program: DocumentProgram[T]
    request: DocumentRequest
    exchanges: list[ProgramExchange]


@dataclass(frozen=True)
class _CompletedValue(Generic[T]):
    value: T


def run_program(program: DocumentProgram[T], executor: DocumentExecutor) -> ProgramRun[T]:
    """Run one interactive document program to completion."""
    return run_programs((program,), executor)[0]


def replay_program(
    program: DocumentProgram[T],
    exchanges: tuple[ProgramExchange, ...],
) -> ProgramRun[T]:
    """Replay recorded responses and fail at the first divergent request."""
    turn = 0

    def execute(requests: tuple[DocumentRequest, ...]) -> tuple[DocumentResponse, ...]:
        nonlocal turn
        if len(requests) != 1:
            raise AssertionError(f"Single-program replay received {len(requests)} ready requests")
        if turn >= len(exchanges):
            raise ProgramReplayError(f"Program yielded an unexpected request at turn {turn}: {requests[0].id!r}")
        exchange = exchanges[turn]
        if requests[0] != exchange.request:
            raise ProgramReplayError(
                f"Program request diverged at turn {turn} ({_request_difference(requests[0], exchange.request)}): "
                f"got {requests[0].id!r}, expected {exchange.request.id!r}"
            )
        turn += 1
        return (exchange.response,)

    run = run_program(program, execute)
    if turn != len(exchanges):
        raise ProgramReplayError(f"Program completed after {turn} turns with {len(exchanges) - turn} turns unused")
    return run


@overload
def run_programs(
    programs: tuple[DocumentProgram[T1]],
    executor: DocumentExecutor,
) -> tuple[ProgramRun[T1]]: ...


@overload
def run_programs(
    programs: tuple[DocumentProgram[T1], DocumentProgram[T2]],
    executor: DocumentExecutor,
) -> tuple[ProgramRun[T1], ProgramRun[T2]]: ...


@overload
def run_programs(
    programs: tuple[DocumentProgram[T1], DocumentProgram[T2], DocumentProgram[T3]],
    executor: DocumentExecutor,
) -> tuple[ProgramRun[T1], ProgramRun[T2], ProgramRun[T3]]: ...


@overload
def run_programs(
    programs: tuple[DocumentProgram[T], ...],
    executor: DocumentExecutor,
) -> tuple[ProgramRun[T], ...]: ...


def run_programs(
    programs: tuple[DocumentProgram[Any], ...],
    executor: DocumentExecutor,
) -> tuple[ProgramRun[Any], ...]:
    """Mix active programs while preserving each yielded request as a barrier."""
    completed: list[ProgramRun[Any] | None] = [None] * len(programs)
    active: list[_ActiveProgram[Any]] = []
    with ExitStack() as cleanup:
        for index, program in enumerate(programs):
            cleanup.callback(program.close)
            try:
                request = next(program)
            except StopIteration as stop:
                completed[index] = ProgramRun(stop.value, ())
                continue
            active.append(_ActiveProgram(index, program, request, []))

        while active:
            requests = tuple(item.request for item in active)
            responses = executor(requests)
            if len(responses) != len(requests):
                raise ValueError(f"Executor returned {len(responses)} responses for {len(requests)} requests")

            next_active = []
            for item, response in zip(active, responses, strict=True):
                _validate_response(item.request, response)
                item.exchanges.append(ProgramExchange(item.request, response))
                try:
                    item.request = item.program.send(response)
                except StopIteration as stop:
                    completed[item.index] = ProgramRun(stop.value, tuple(item.exchanges))
                    continue
                next_active.append(item)
            active = next_active
    if any(run is None for run in completed):
        raise AssertionError("Document program driver stopped without completing every program")
    return tuple(run for run in completed if run is not None)


@overload
def parallel_programs(
    programs: tuple[DocumentProgram[T1], DocumentProgram[T2]],
    *,
    request_prefix: str,
) -> DocumentProgram[tuple[T1, T2]]: ...


@overload
def parallel_programs(
    programs: tuple[DocumentProgram[T1], DocumentProgram[T2], DocumentProgram[T3]],
    *,
    request_prefix: str,
) -> DocumentProgram[tuple[T1, T2, T3]]: ...


@overload
def parallel_programs(
    programs: tuple[DocumentProgram[T], ...],
    *,
    request_prefix: str,
) -> DocumentProgram[tuple[T, ...]]: ...


def parallel_programs(
    programs: tuple[DocumentProgram[Any], ...],
    *,
    request_prefix: str,
) -> DocumentProgram[tuple[Any, ...]]:
    """Compose adaptive child programs while exposing their ready documents as one wave."""
    if not programs:
        raise ValueError("parallel_programs requires at least one child program")
    completed: list[_CompletedValue[Any] | None] = [None] * len(programs)
    active: list[_ActiveProgram[Any]] = []
    wave = 0
    with ExitStack() as cleanup:
        for index, program in enumerate(programs):
            cleanup.callback(program.close)
            try:
                request = next(program)
            except StopIteration as stop:
                completed[index] = _CompletedValue(stop.value)
                continue
            active.append(_ActiveProgram(index, program, request, []))

        while active:
            document_counts = tuple(len(item.request.documents) for item in active)
            request = DocumentRequest(
                f"{request_prefix}/wave{wave}",
                tuple(document for item in active for document in item.request.documents),
                frozenset(origin for item in active for origin in item.request.accepted_origins),
            )
            response = yield request

            child_responses = []
            offset = 0
            for item, document_count in zip(active, document_counts, strict=True):
                child_response = DocumentResponse(
                    item.request.id,
                    response.results[offset : offset + document_count],
                )
                _validate_response(item.request, child_response)
                child_responses.append(child_response)
                offset += document_count

            next_active = []
            for item, child_response in zip(active, child_responses, strict=True):
                try:
                    item.request = item.program.send(child_response)
                except StopIteration as stop:
                    completed[item.index] = _CompletedValue(stop.value)
                    continue
                next_active.append(item)
            active = next_active
            wave += 1
    if any(value is None for value in completed):
        raise AssertionError("Parallel document programs stopped without completing every child")
    return tuple(value.value for value in completed if value is not None)


def disjoint_prediction_observations(response: DocumentResponse) -> tuple[PredictionObservation, ...]:
    """Select observations only when each logical slot occurs once."""
    slots = [observation.slot for observation in response.observations]
    if len(slots) != len(set(slots)):
        raise ValueError("Cannot commit overlapping observations without an explicit selection policy")
    return response.observations


def highest_logprob_observations(response: DocumentResponse) -> tuple[PredictionObservation, ...]:
    """Select the highest-logprob observation for every logical output slot."""
    best: dict[OutputSlot, PredictionObservation] = {}
    for observation in response.observations:
        current = best.get(observation.slot)
        if current is None or observation.logprob > current.logprob:
            best[observation.slot] = observation
    return tuple(best[slot] for slot in sorted(best))


def prediction_values(observations: tuple[PredictionObservation, ...]) -> tuple[PredictionValue, ...]:
    """Discard observation metadata when committing selected predictions."""
    return tuple(observation.prediction_value() for observation in observations)


def supervised_executor(requests: tuple[DocumentRequest, ...]) -> tuple[DocumentResponse, ...]:
    """Return document supervision as explicit teacher feedback."""
    responses = []
    for request in requests:
        results = []
        for document in request.documents:
            observations = []
            for record in document.records:
                if record.output is None:
                    continue
                if record.output.supervision is None:
                    raise ValueError(f"Document {document.id!r} output {record.output.slot} has no supervised target")
                observations.append(
                    PredictionObservation(
                        record.output.slot,
                        record.output.supervision.target_id,
                        logprob=0.0,
                    )
                )
            results.append(DocumentResult(document.id, tuple(observations), FeedbackOrigin.SUPERVISED))
        responses.append(DocumentResponse(request.id, tuple(results)))
    return tuple(responses)


def mapped_executor(
    predict: Callable[[Document], tuple[PredictionObservation, ...]],
    *,
    origin: FeedbackOrigin = FeedbackOrigin.SAMPLED,
) -> DocumentExecutor:
    """Build a synchronous executor from a per-document prediction function."""

    def execute(requests: tuple[DocumentRequest, ...]) -> tuple[DocumentResponse, ...]:
        return tuple(
            DocumentResponse(
                request.id,
                tuple(DocumentResult(document.id, predict(document), origin) for document in request.documents),
            )
            for request in requests
        )

    return execute


def packed_executor(
    predict: Callable[[PackedDocuments], PackedPredictions],
    *,
    max_seq_len: int,
    origin: FeedbackOrigin = FeedbackOrigin.SAMPLED,
) -> DocumentExecutor:
    """Pack ready documents by attention layout and route batch predictions back to requests."""

    def execute(requests: tuple[DocumentRequest, ...]) -> tuple[DocumentResponse, ...]:
        flat_documents = tuple(document for request in requests for document in request.documents)
        grouped: dict[AttentionLayout, list[tuple[int, Document]]] = {}
        for document_index, document in enumerate(flat_documents):
            grouped.setdefault(document.attention_layout, []).append((document_index, document))

        observations: list[list[PredictionObservation]] = [[] for _ in flat_documents]
        for items in grouped.values():
            batch = pack_documents(tuple(document for _, document in items), max_seq_len=max_seq_len)
            predictions = predict(batch)
            if predictions.token_ids.shape != batch.token_ids.shape:
                raise ValueError(
                    f"Packed prediction tokens must have shape {batch.token_ids.shape}, "
                    f"got {predictions.token_ids.shape}"
                )
            if predictions.logprobs.shape != batch.token_ids.shape:
                raise ValueError(
                    f"Packed prediction logprobs must have shape {batch.token_ids.shape}, "
                    f"got {predictions.logprobs.shape}"
                )
            for output in batch.outputs:
                flat_index = items[output.document_index][0]
                observations[flat_index].append(
                    PredictionObservation(
                        output.slot,
                        int(predictions.token_ids[output.row, output.position]),
                        float(predictions.logprobs[output.row, output.position]),
                    )
                )

        responses = []
        offset = 0
        for request in requests:
            results = tuple(
                DocumentResult(document.id, tuple(observations[offset + document_index]), origin)
                for document_index, document in enumerate(request.documents)
            )
            responses.append(DocumentResponse(request.id, results))
            offset += len(request.documents)
        return tuple(responses)

    return execute


def _validate_response(request: DocumentRequest, response: DocumentResponse) -> None:
    if response.request_id != request.id:
        raise ValueError(f"Response for {response.request_id!r} cannot satisfy request {request.id!r}")
    if len(response.results) != len(request.documents):
        raise ValueError(
            f"Response for {request.id!r} has {len(response.results)} document results, "
            f"expected {len(request.documents)}"
        )
    for document, result in zip(request.documents, response.results, strict=True):
        if result.document_id != document.id:
            raise ValueError(f"Result for document {result.document_id!r} cannot satisfy document {document.id!r}")
        if result.origin not in request.accepted_origins:
            raise ValueError(
                f"Result for document {document.id!r} has feedback origin {result.origin.value!r}; "
                f"accepted origins are {sorted(request.accepted_origins)}"
            )
        expected_slots = Counter(slot for slot in document.output_slots if slot is not None)
        observed_slots = Counter(observation.slot for observation in result.observations)
        if observed_slots != expected_slots:
            raise ValueError(
                f"Result for document {document.id!r} returned slots {observed_slots}, expected {expected_slots}"
            )


def _request_difference(actual: DocumentRequest, expected: DocumentRequest) -> str:
    if actual.id != expected.id:
        return "request id differs"
    if actual.accepted_origins != expected.accepted_origins:
        return "accepted feedback origins differ"
    if len(actual.documents) != len(expected.documents):
        return "document count differs"
    for index, (actual_document, expected_document) in enumerate(zip(actual.documents, expected.documents, strict=True)):
        if actual_document != expected_document:
            return f"document {index} differs"
    return "request contents differ"
