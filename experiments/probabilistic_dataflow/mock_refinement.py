# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from itertools import batched

from experiments.probabilistic_dataflow.documents import (
    AttentionLayout,
    Document,
    FeatureId,
    Output,
    OutputSlot,
    PredictionState,
    PredictionUpdateMode,
    Record,
    Supervision,
    prediction_input_record,
)
from experiments.probabilistic_dataflow.programs import (
    DocumentProgram,
    DocumentRequest,
    FeedbackOrigin,
    disjoint_prediction_observations,
    prediction_values,
)

QUERY_TOKEN_ID = 1
FIELD_CHANNEL = "field"
COORDINATE_CHANNEL = "coordinate"
OBSERVED_FIELD_ID = 0
PREDICTED_FIELD_ID = 1


@dataclass(frozen=True)
class RefinementResult:
    """Final prediction from an adaptive field-refinement program."""

    slots: tuple[OutputSlot, ...]
    predictions: PredictionState
    refinement_rounds: int

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self.predictions.value(slot) for slot in self.slots)


def iterative_refinement_program(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    num_outputs: int,
    outputs_per_document: int,
    minimum_logprob: float,
    max_refinement_rounds: int,
    accepted_feedback: frozenset[FeedbackOrigin],
) -> DocumentProgram[RefinementResult]:
    """Predict an unlabeled field, then replace its low-confidence coordinates."""
    return (
        yield from _iterative_refinement_program(
            example_id=example_id,
            observed_token_ids=observed_token_ids,
            num_outputs=num_outputs,
            supervision=(),
            outputs_per_document=outputs_per_document,
            minimum_logprob=minimum_logprob,
            max_refinement_rounds=max_refinement_rounds,
            accepted_feedback=accepted_feedback,
        )
    )


def supervised_iterative_refinement_program(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    target_token_ids: tuple[int, ...],
    outputs_per_document: int,
    minimum_logprob: float,
    max_refinement_rounds: int,
    accepted_feedback: frozenset[FeedbackOrigin],
) -> DocumentProgram[RefinementResult]:
    """Build the same adaptive program with labels attached to its queries."""
    return (
        yield from _iterative_refinement_program(
            example_id=example_id,
            observed_token_ids=observed_token_ids,
            num_outputs=len(target_token_ids),
            supervision=tuple(Supervision(target_id) for target_id in target_token_ids),
            outputs_per_document=outputs_per_document,
            minimum_logprob=minimum_logprob,
            max_refinement_rounds=max_refinement_rounds,
            accepted_feedback=accepted_feedback,
        )
    )


def _iterative_refinement_program(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    num_outputs: int,
    supervision: tuple[Supervision, ...],
    outputs_per_document: int,
    minimum_logprob: float,
    max_refinement_rounds: int,
    accepted_feedback: frozenset[FeedbackOrigin],
) -> DocumentProgram[RefinementResult]:
    if not observed_token_ids:
        raise ValueError("Refinement requires at least one observed value")
    if num_outputs <= 0:
        raise ValueError("Refinement requires at least one output")
    if supervision and len(supervision) != num_outputs:
        raise ValueError("Refinement supervision must match the number of outputs")
    if outputs_per_document <= 0:
        raise ValueError("outputs_per_document must be positive")
    if max_refinement_rounds < 0:
        raise ValueError("max_refinement_rounds cannot be negative")

    slots = tuple(OutputSlot(example_id, "predicted_field", index) for index in range(num_outputs))
    response = yield DocumentRequest(
        f"{example_id}/proposal",
        _proposal_documents(
            example_id=example_id,
            observed_token_ids=observed_token_ids,
            slots=slots,
            supervision=supervision,
            outputs_per_document=outputs_per_document,
        ),
        accepted_feedback,
    )
    observations = disjoint_prediction_observations(response)
    state = PredictionState().updated(
        prediction_values(observations),
        mode=PredictionUpdateMode.REQUIRE_EMPTY,
    )
    logprobs = {observation.slot: observation.logprob for observation in observations}

    refinement_rounds = 0
    while refinement_rounds < max_refinement_rounds:
        selected = tuple(slot for slot in slots if logprobs[slot] < minimum_logprob)
        if not selected:
            break

        response = yield DocumentRequest(
            f"{example_id}/refine/{refinement_rounds}",
            _refinement_documents(
                example_id=example_id,
                observed_token_ids=observed_token_ids,
                slots=slots,
                supervision=supervision,
                state=state,
                selected=selected,
                outputs_per_document=outputs_per_document,
                round_index=refinement_rounds,
            ),
            accepted_feedback,
        )
        observations = disjoint_prediction_observations(response)
        state = state.updated(
            prediction_values(observations),
            mode=PredictionUpdateMode.REPLACE,
        )
        logprobs.update((observation.slot, observation.logprob) for observation in observations)
        refinement_rounds += 1

    return RefinementResult(slots, state, refinement_rounds)


def _proposal_documents(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    slots: tuple[OutputSlot, ...],
    supervision: tuple[Supervision, ...],
    outputs_per_document: int,
) -> tuple[Document, ...]:
    context = _observed_records(observed_token_ids)
    documents = []
    for shard_index, shard in enumerate(batched(slots, outputs_per_document)):
        outputs = tuple(_query_record(slot, supervision) for slot in shard)
        documents.append(
            Document(
                f"{example_id}/proposal/{shard_index}",
                context + outputs,
                AttentionLayout.FULL,
            )
        )
    return tuple(documents)


def _refinement_documents(
    *,
    example_id: str,
    observed_token_ids: tuple[int, ...],
    slots: tuple[OutputSlot, ...],
    supervision: tuple[Supervision, ...],
    state: PredictionState,
    selected: tuple[OutputSlot, ...],
    outputs_per_document: int,
    round_index: int,
) -> tuple[Document, ...]:
    context = _observed_records(observed_token_ids) + tuple(
        prediction_input_record(
            state,
            slot,
            position_id=0,
            features=_field_features(PREDICTED_FIELD_ID, slot.index),
        )
        for slot in slots
    )
    documents = []
    for shard_index, shard in enumerate(batched(selected, outputs_per_document)):
        outputs = tuple(_query_record(slot, supervision) for slot in shard)
        documents.append(
            Document(
                f"{example_id}/refine/{round_index}/{shard_index}",
                context + outputs,
                AttentionLayout.FULL,
            )
        )
    return tuple(documents)


def _observed_records(token_ids: tuple[int, ...]) -> tuple[Record, ...]:
    return tuple(
        Record(token_id, position_id=0, features=_field_features(OBSERVED_FIELD_ID, index))
        for index, token_id in enumerate(token_ids)
    )


def _query_record(slot: OutputSlot, supervision: tuple[Supervision, ...]) -> Record:
    return Record(
        QUERY_TOKEN_ID,
        position_id=0,
        features=_field_features(PREDICTED_FIELD_ID, slot.index),
        output=Output(slot, supervision[slot.index] if supervision else None),
    )


def _field_features(field_id: int, coordinate: int) -> tuple[FeatureId, ...]:
    return (FeatureId(FIELD_CHANNEL, field_id), FeatureId(COORDINATE_CHANNEL, coordinate))
