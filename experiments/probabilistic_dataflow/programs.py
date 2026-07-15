# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Generic, Protocol, TypeVar, overload

import numpy as np

from experiments.probabilistic_dataflow.documents import AttentionLayout, Document, PackedBatch, pack

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class Origin(StrEnum):
    SAMPLED = "sampled"
    SUPERVISED = "supervised"
    CORRUPTED = "corrupted"


SAMPLED_ORIGINS = frozenset({Origin.SAMPLED})
GENERATED_ORIGINS = frozenset({Origin.SAMPLED, Origin.CORRUPTED})


class ProgramReplayError(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    token_id: int
    logprob: float

    def __post_init__(self) -> None:
        if not isfinite(self.logprob) or self.logprob > 0:
            raise ValueError(f"Prediction logprob must be finite and non-positive, got {self.logprob}")


@dataclass(frozen=True)
class Result:
    """Predictions for one document, in query-token order."""

    predictions: tuple[Prediction, ...]
    origin: Origin


Program = Generator[tuple[Document, ...], tuple[Result, ...], T]


class Executor(Protocol):
    """Execute ready documents and return one positional result per document."""

    def __call__(self, documents: tuple[Document, ...]) -> tuple[Result, ...]: ...


@dataclass(frozen=True)
class PackedSamples:
    token_ids: np.ndarray
    logprobs: np.ndarray


@dataclass(frozen=True)
class Exchange:
    documents: tuple[Document, ...]
    results: tuple[Result, ...]


@dataclass(frozen=True)
class Run(Generic[T]):
    value: T
    exchanges: tuple[Exchange, ...]


@dataclass
class _ActiveProgram(Generic[T]):
    index: int
    program: Program[T]
    documents: tuple[Document, ...]
    exchanges: list[Exchange]


_MISSING = object()


def run(
    program: Program[T],
    executor: Executor,
    *,
    accepted_origins: frozenset[Origin] = SAMPLED_ORIGINS,
) -> Run[T]:
    """Run one interactive document program to completion."""
    return run_many((program,), executor, accepted_origins=accepted_origins)[0]


def replay(program: Program[T], exchanges: tuple[Exchange, ...]) -> Run[T]:
    """Replay recorded results and fail at the first divergent document wave."""
    turn = 0

    def execute(documents: tuple[Document, ...]) -> tuple[Result, ...]:
        nonlocal turn
        if turn >= len(exchanges):
            raise ProgramReplayError(f"Program yielded unexpected documents at turn {turn}")
        exchange = exchanges[turn]
        if documents != exchange.documents:
            raise ProgramReplayError(
                f"Program diverged at turn {turn} ({_documents_difference(documents, exchange.documents)})"
            )
        turn += 1
        return exchange.results

    run_result = run(program, execute, accepted_origins=frozenset(Origin))
    if turn != len(exchanges):
        raise ProgramReplayError(f"Program completed after {turn} turns with {len(exchanges) - turn} turns unused")
    return run_result


@overload
def run_many(
    programs: tuple[Program[T1]],
    executor: Executor,
    *,
    accepted_origins: frozenset[Origin] = SAMPLED_ORIGINS,
) -> tuple[Run[T1]]: ...


@overload
def run_many(
    programs: tuple[Program[T1], Program[T2]],
    executor: Executor,
    *,
    accepted_origins: frozenset[Origin] = SAMPLED_ORIGINS,
) -> tuple[Run[T1], Run[T2]]: ...


@overload
def run_many(
    programs: tuple[Program[T1], Program[T2], Program[T3]],
    executor: Executor,
    *,
    accepted_origins: frozenset[Origin] = SAMPLED_ORIGINS,
) -> tuple[Run[T1], Run[T2], Run[T3]]: ...


@overload
def run_many(
    programs: tuple[Program[T], ...],
    executor: Executor,
    *,
    accepted_origins: frozenset[Origin] = SAMPLED_ORIGINS,
) -> tuple[Run[T], ...]: ...


def run_many(
    programs: tuple[Program[Any], ...],
    executor: Executor,
    *,
    accepted_origins: frozenset[Origin] = SAMPLED_ORIGINS,
) -> tuple[Run[Any], ...]:
    """Advance independent programs together without crossing yield barriers."""
    if not accepted_origins:
        raise ValueError("A program run requires at least one accepted result origin")
    completed: list[Run[Any] | None] = [None] * len(programs)
    active: list[_ActiveProgram[Any]] = []
    with ExitStack() as cleanup:
        for index, program in enumerate(programs):
            cleanup.callback(program.close)
            try:
                documents = next(program)
            except StopIteration as stop:
                completed[index] = Run(stop.value, ())
                continue
            _validate_document_wave(documents)
            active.append(_ActiveProgram(index, program, documents, []))

        while active:
            document_counts = tuple(len(item.documents) for item in active)
            ready_documents = tuple(document for item in active for document in item.documents)
            ready_results = executor(ready_documents)
            if len(ready_results) != len(ready_documents):
                raise ValueError(f"Executor returned {len(ready_results)} results for {len(ready_documents)} documents")

            next_active = []
            offset = 0
            for item, document_count in zip(active, document_counts, strict=True):
                results = ready_results[offset : offset + document_count]
                offset += document_count
                _validate_results(item.documents, results, accepted_origins)
                item.exchanges.append(Exchange(item.documents, results))
                try:
                    item.documents = item.program.send(results)
                except StopIteration as stop:
                    completed[item.index] = Run(stop.value, tuple(item.exchanges))
                    continue
                _validate_document_wave(item.documents)
                next_active.append(item)
            active = next_active
    if any(run_result is None for run_result in completed):
        raise AssertionError("Program driver stopped without completing every program")
    return tuple(run_result for run_result in completed if run_result is not None)


@overload
def parallel(programs: tuple[Program[T1], Program[T2]]) -> Program[tuple[T1, T2]]: ...


@overload
def parallel(programs: tuple[Program[T1], Program[T2], Program[T3]]) -> Program[tuple[T1, T2, T3]]: ...


@overload
def parallel(programs: tuple[Program[T], ...]) -> Program[tuple[T, ...]]: ...


def parallel(programs: tuple[Program[Any], ...]) -> Program[tuple[Any, ...]]:
    """Advance adaptive child programs while exposing their ready documents together."""
    if not programs:
        raise ValueError("parallel requires at least one child program")
    completed: list[Any] = [_MISSING] * len(programs)
    active: list[_ActiveProgram[Any]] = []
    with ExitStack() as cleanup:
        for index, program in enumerate(programs):
            cleanup.callback(program.close)
            try:
                documents = next(program)
            except StopIteration as stop:
                completed[index] = stop.value
                continue
            _validate_document_wave(documents)
            active.append(_ActiveProgram(index, program, documents, []))

        while active:
            document_counts = tuple(len(item.documents) for item in active)
            results = yield tuple(document for item in active for document in item.documents)
            if len(results) != sum(document_counts):
                raise ValueError(
                    f"Parallel program received {len(results)} results for {sum(document_counts)} documents"
                )

            next_active = []
            offset = 0
            for item, document_count in zip(active, document_counts, strict=True):
                child_results = results[offset : offset + document_count]
                offset += document_count
                try:
                    item.documents = item.program.send(child_results)
                except StopIteration as stop:
                    completed[item.index] = stop.value
                    continue
                _validate_document_wave(item.documents)
                next_active.append(item)
            active = next_active
    if any(value is _MISSING for value in completed):
        raise AssertionError("Parallel programs stopped without completing every child")
    return tuple(value for value in completed if value is not _MISSING)


def mapped_executor(
    predict: Callable[[Document], tuple[Prediction, ...]],
    *,
    origin: Origin = Origin.SAMPLED,
) -> Executor:
    """Build a synchronous executor from a per-document prediction function."""

    def execute(documents: tuple[Document, ...]) -> tuple[Result, ...]:
        return tuple(Result(predict(document), origin) for document in documents)

    return execute


def packed_executor(
    predict: Callable[[PackedBatch], PackedSamples],
    *,
    max_seq_len: int,
    origin: Origin = Origin.SAMPLED,
) -> Executor:
    """Pack documents by attention layout and restore positional document results."""

    def execute(documents: tuple[Document, ...]) -> tuple[Result, ...]:
        grouped: dict[AttentionLayout, list[tuple[int, Document]]] = {}
        for document_index, document in enumerate(documents):
            grouped.setdefault(document.attention_layout, []).append((document_index, document))

        predictions: list[list[Prediction]] = [[] for _ in documents]
        for items in grouped.values():
            batch = pack(tuple(document for _, document in items), max_seq_len=max_seq_len)
            samples = predict(batch)
            if samples.token_ids.shape != batch.token_ids.shape:
                raise ValueError(
                    f"Packed sample tokens must have shape {batch.token_ids.shape}, got {samples.token_ids.shape}"
                )
            if samples.logprobs.shape != batch.token_ids.shape:
                raise ValueError(
                    f"Packed sample logprobs must have shape {batch.token_ids.shape}, got {samples.logprobs.shape}"
                )
            for row, position in np.argwhere(batch.query_mask):
                local_document_index = int(batch.document_indices[row, position])
                document_index = items[local_document_index][0]
                predictions[document_index].append(
                    Prediction(
                        int(samples.token_ids[row, position]),
                        float(samples.logprobs[row, position]),
                    )
                )

        return tuple(Result(tuple(document_predictions), origin) for document_predictions in predictions)

    return execute


def _validate_document_wave(documents: tuple[Document, ...]) -> None:
    if not documents:
        raise ValueError("A program must yield at least one document")


def _validate_results(
    documents: tuple[Document, ...],
    results: tuple[Result, ...],
    accepted_origins: frozenset[Origin],
) -> None:
    for document, result in zip(documents, results, strict=True):
        if result.origin not in accepted_origins:
            raise ValueError(
                f"Result for document {document.name!r} has origin {result.origin.value!r}; "
                f"accepted origins are {sorted(accepted_origins)}"
            )
        expected_predictions = len(document.query_positions)
        if len(result.predictions) != expected_predictions:
            raise ValueError(
                f"Result for document {document.name!r} returned {len(result.predictions)} predictions, "
                f"expected {expected_predictions}"
            )


def _documents_difference(actual: tuple[Document, ...], expected: tuple[Document, ...]) -> str:
    if len(actual) != len(expected):
        return "document count differs"
    for index, (actual_document, expected_document) in enumerate(zip(actual, expected, strict=True)):
        if actual_document != expected_document:
            return f"document {index} differs"
    return "document contents differ"
