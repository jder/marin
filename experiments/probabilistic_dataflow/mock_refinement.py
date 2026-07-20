# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from itertools import batched
from typing import Any

from experiments.probabilistic_dataflow.documents import (
    QUERY,
    TARGET_IDS,
    AttentionLayout,
    Coordinate,
    Document,
)
from experiments.probabilistic_dataflow.programs import Prediction, Program, Result

QUERY_TOKEN_ID = 1
FIELD = Coordinate("field")
COORDINATE = Coordinate("coordinate")
OBSERVED_FIELD_ID = 0
PREDICTED_FIELD_ID = 1


@dataclass(frozen=True)
class RefinementResult:
    token_ids: tuple[int, ...]
    refinement_rounds: int


def refine(
    *,
    observed_token_ids: tuple[int, ...],
    num_outputs: int,
    outputs_per_document: int,
    minimum_logprob: float,
    max_refinement_rounds: int,
    target_token_ids: tuple[int, ...] = (),
) -> Program[RefinementResult]:
    """Predict a field and adaptively replace its low-confidence coordinates."""
    if not observed_token_ids:
        raise ValueError("Refinement requires at least one observed value")
    if num_outputs <= 0:
        raise ValueError("Refinement requires at least one output")
    if target_token_ids and len(target_token_ids) != num_outputs:
        raise ValueError("Refinement targets must match the number of outputs")
    if outputs_per_document <= 0:
        raise ValueError("outputs_per_document must be positive")
    if max_refinement_rounds < 0:
        raise ValueError("max_refinement_rounds cannot be negative")
    return _refinement_program(
        observed_token_ids=observed_token_ids,
        num_outputs=num_outputs,
        target_token_ids=target_token_ids,
        outputs_per_document=outputs_per_document,
        minimum_logprob=minimum_logprob,
        max_refinement_rounds=max_refinement_rounds,
    )


def _refinement_program(
    *,
    observed_token_ids: tuple[int, ...],
    num_outputs: int,
    target_token_ids: tuple[int, ...],
    outputs_per_document: int,
    minimum_logprob: float,
    max_refinement_rounds: int,
) -> Program[RefinementResult]:
    proposal_shards = tuple(batched(range(num_outputs), outputs_per_document))
    results = yield _proposal_documents(
        observed_token_ids=observed_token_ids,
        shards=proposal_shards,
        target_token_ids=target_token_ids,
    )
    predictions = _predictions_by_index(proposal_shards, results)
    values = {index: prediction.token_id for index, prediction in predictions.items()}
    logprobs = {index: prediction.logprob for index, prediction in predictions.items()}

    refinement_rounds = 0
    while refinement_rounds < max_refinement_rounds:
        selected = tuple(index for index in range(num_outputs) if logprobs[index] < minimum_logprob)
        if not selected:
            break

        refinement_shards = tuple(batched(selected, outputs_per_document))
        results = yield _replacement_documents(
            observed_token_ids=observed_token_ids,
            values=values,
            shards=refinement_shards,
            target_token_ids=target_token_ids,
        )
        replacements = _predictions_by_index(refinement_shards, results)
        if not replacements.keys() <= values.keys():
            raise AssertionError("Refinement produced an unknown coordinate")
        values.update((index, prediction.token_id) for index, prediction in replacements.items())
        logprobs.update((index, prediction.logprob) for index, prediction in replacements.items())
        refinement_rounds += 1

    return RefinementResult(tuple(values[index] for index in range(num_outputs)), refinement_rounds)


def _proposal_documents(
    *,
    observed_token_ids: tuple[int, ...],
    shards: tuple[tuple[int, ...], ...],
    target_token_ids: tuple[int, ...],
) -> tuple[Document, ...]:
    context = _field_document(observed_token_ids, OBSERVED_FIELD_ID, tuple(range(len(observed_token_ids))))
    return tuple(context + _query_document(shard, target_token_ids) for shard in shards)


def _replacement_documents(
    *,
    observed_token_ids: tuple[int, ...],
    values: dict[int, int],
    shards: tuple[tuple[int, ...], ...],
    target_token_ids: tuple[int, ...],
) -> tuple[Document, ...]:
    value_indices = tuple(sorted(values))
    context = _field_document(
        observed_token_ids,
        OBSERVED_FIELD_ID,
        tuple(range(len(observed_token_ids))),
    ) + _field_document(
        tuple(values[index] for index in value_indices),
        PREDICTED_FIELD_ID,
        value_indices,
    )
    return tuple(context + _query_document(shard, target_token_ids) for shard in shards)


def _predictions_by_index(
    shards: tuple[tuple[int, ...], ...],
    results: tuple[Result, ...],
) -> dict[int, Prediction]:
    return {
        index: prediction
        for shard, result in zip(shards, results, strict=True)
        for index, prediction in zip(shard, result.predictions, strict=True)
    }


def _field_document(token_ids: tuple[int, ...], field_id: int, coordinates: tuple[int, ...]) -> Document:
    return Document(
        token_ids,
        {
            FIELD: (field_id,) * len(token_ids),
            COORDINATE: coordinates,
        },
        attention=AttentionLayout.FULL,
    )


def _query_document(indices: tuple[int, ...], target_token_ids: tuple[int, ...]) -> Document:
    coordinates: dict[Coordinate, tuple[Any, ...]] = {
        FIELD: (PREDICTED_FIELD_ID,) * len(indices),
        COORDINATE: indices,
        QUERY: (True,) * len(indices),
    }
    if target_token_ids:
        coordinates[TARGET_IDS] = tuple(target_token_ids[index] for index in indices)
    return Document(
        (QUERY_TOKEN_ID,) * len(indices),
        coordinates,
        attention=AttentionLayout.FULL,
    )
