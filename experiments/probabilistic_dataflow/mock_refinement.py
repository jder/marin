# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from itertools import batched

from experiments.probabilistic_dataflow.documents import AttentionLayout, Document, Token
from experiments.probabilistic_dataflow.programs import Prediction, Program, Result

QUERY_TOKEN_ID = 1
FIELD_CHANNEL = "field"
COORDINATE_CHANNEL = "coordinate"
OBSERVED_FIELD_ID = 0
PREDICTED_FIELD_ID = 1


@dataclass(frozen=True)
class Refinement:
    """Inputs and policy for an adaptive refinement document program."""

    example_id: str
    observed_token_ids: tuple[int, ...]
    num_outputs: int
    target_token_ids: tuple[int, ...]
    outputs_per_document: int
    minimum_logprob: float
    max_refinement_rounds: int

    def documents(self) -> Program[RefinementResult]:
        return _refinement_program(self)


@dataclass(frozen=True)
class RefinementResult:
    token_ids: tuple[int, ...]
    refinement_rounds: int


def refine(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    num_outputs: int,
    outputs_per_document: int,
    minimum_logprob: float,
    max_refinement_rounds: int,
    target_token_ids: tuple[int, ...] = (),
) -> Refinement:
    """Describe a field proposal and its adaptive refinement policy."""
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
    return Refinement(
        example_id=example_id,
        observed_token_ids=observed_token_ids,
        num_outputs=num_outputs,
        target_token_ids=target_token_ids,
        outputs_per_document=outputs_per_document,
        minimum_logprob=minimum_logprob,
        max_refinement_rounds=max_refinement_rounds,
    )


def _refinement_program(refinement: Refinement) -> Program[RefinementResult]:
    proposal_shards = tuple(batched(range(refinement.num_outputs), refinement.outputs_per_document))
    results = yield _proposal_documents(
        example_id=refinement.example_id,
        observed_token_ids=refinement.observed_token_ids,
        shards=proposal_shards,
        target_token_ids=refinement.target_token_ids,
    )
    predictions = _predictions_by_index(proposal_shards, results)
    values = {index: prediction.token_id for index, prediction in predictions.items()}
    logprobs = {index: prediction.logprob for index, prediction in predictions.items()}

    refinement_rounds = 0
    while refinement_rounds < refinement.max_refinement_rounds:
        selected = tuple(
            index for index in range(refinement.num_outputs) if logprobs[index] < refinement.minimum_logprob
        )
        if not selected:
            break

        refinement_shards = tuple(batched(selected, refinement.outputs_per_document))
        results = yield _replacement_documents(
            example_id=refinement.example_id,
            observed_token_ids=refinement.observed_token_ids,
            values=values,
            shards=refinement_shards,
            target_token_ids=refinement.target_token_ids,
            round_index=refinement_rounds,
        )
        replacements = _predictions_by_index(refinement_shards, results)
        if not replacements.keys() <= values.keys():
            raise AssertionError("Refinement produced an unknown coordinate")
        values.update((index, prediction.token_id) for index, prediction in replacements.items())
        logprobs.update((index, prediction.logprob) for index, prediction in replacements.items())
        refinement_rounds += 1

    return RefinementResult(tuple(values[index] for index in range(refinement.num_outputs)), refinement_rounds)


def _proposal_documents(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    shards: tuple[tuple[int, ...], ...],
    target_token_ids: tuple[int, ...],
) -> tuple[Document, ...]:
    context = _observed_tokens(observed_token_ids)
    return tuple(
        Document(
            f"{example_id}/proposal/{shard_index}",
            context + tuple(_query_token(index, target_token_ids) for index in shard),
            AttentionLayout.FULL,
        )
        for shard_index, shard in enumerate(shards)
    )


def _replacement_documents(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    values: dict[int, int],
    shards: tuple[tuple[int, ...], ...],
    target_token_ids: tuple[int, ...],
    round_index: int,
) -> tuple[Document, ...]:
    context = _observed_tokens(observed_token_ids) + tuple(
        Token(
            values[index],
            features=_field_features(PREDICTED_FIELD_ID, index),
        )
        for index in sorted(values)
    )
    return tuple(
        Document(
            f"{example_id}/refine/{round_index}/{shard_index}",
            context + tuple(_query_token(index, target_token_ids) for index in shard),
            AttentionLayout.FULL,
        )
        for shard_index, shard in enumerate(shards)
    )


def _predictions_by_index(
    shards: tuple[tuple[int, ...], ...],
    results: tuple[Result, ...],
) -> dict[int, Prediction]:
    return {
        index: prediction
        for shard, result in zip(shards, results, strict=True)
        for index, prediction in zip(shard, result.predictions, strict=True)
    }


def _observed_tokens(token_ids: tuple[int, ...]) -> tuple[Token, ...]:
    return tuple(
        Token(token_id, features=_field_features(OBSERVED_FIELD_ID, index)) for index, token_id in enumerate(token_ids)
    )


def _query_token(index: int, target_token_ids: tuple[int, ...]) -> Token:
    return Token(
        QUERY_TOKEN_ID,
        features=_field_features(PREDICTED_FIELD_ID, index),
        query=True,
        target_id=target_token_ids[index] if target_token_ids else None,
    )


def _field_features(field_id: int, coordinate: int) -> tuple[tuple[str, int], ...]:
    return ((FIELD_CHANNEL, field_id), (COORDINATE_CHANNEL, coordinate))
