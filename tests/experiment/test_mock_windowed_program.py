# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

from experiments.probabilistic_dataflow.documents import Document, OutputSlot, PredictionState
from experiments.probabilistic_dataflow.mock_windowed import (
    ContextWindow,
    forecast_slot,
    windowed_forecast_program,
)
from experiments.probabilistic_dataflow.programs import PredictionObservation, mapped_executor, run_program


def _predict_with(
    token_and_logprob: Callable[[Document, OutputSlot], tuple[int, float]],
) -> Callable[[Document], tuple[PredictionObservation, ...]]:
    def predict(document: Document) -> tuple[PredictionObservation, ...]:
        observations = []
        for slot in document.output_slots:
            if slot is None:
                continue
            token_id, logprob = token_and_logprob(document, slot)
            observations.append(PredictionObservation(slot, token_id, logprob))
        return tuple(observations)

    return predict


def _state_tokens(state: PredictionState, example_id: str, indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(state.value(forecast_slot(example_id, index)) for index in indices)


def test_windowed_forecast_assembles_disjoint_context_shards() -> None:
    example_id = "forecast-disjoint"
    program = windowed_forecast_program(
        example_id,
        (
            ContextWindow("left", (10, 11), (0, 1)),
            ContextWindow("right", (20, 21), (2, 3)),
        ),
        continuation_output_indices=(),
    )
    executor = mapped_executor(_predict_with(lambda document, slot: (100 + slot.index, -0.1)))

    result = run_program(program, executor)

    assert _state_tokens(result.value, example_id, (0, 1, 2, 3)) == (100, 101, 102, 103)
    assert tuple(document.token_ids for document in result.exchanges[0].request.documents) == (
        (10, 11, 1, 1),
        (20, 21, 1, 1),
    )


def test_windowed_forecast_materializes_prior_predictions_in_dependent_continuation() -> None:
    example_id = "forecast-continuation"
    program = windowed_forecast_program(
        example_id,
        (ContextWindow("prefix", (10,), (0, 1)),),
        continuation_output_indices=(2, 3),
    )
    executor = mapped_executor(_predict_with(lambda document, slot: (200 + slot.index, -0.1)))

    result = run_program(program, executor)

    assert _state_tokens(result.value, example_id, (0, 1, 2, 3)) == (200, 201, 202, 203)
    continuation = result.exchanges[1].request.documents[0]
    assert continuation.token_ids == (200, 201, 1, 1)
    assert continuation.output_slots == (None, None, forecast_slot(example_id, 2), forecast_slot(example_id, 3))


def test_overlapping_windows_commit_highest_logprob_observation() -> None:
    example_id = "forecast-overlap"
    program = windowed_forecast_program(
        example_id,
        (
            ContextWindow("left", (10,), (0, 1)),
            ContextWindow("right", (20,), (1, 2)),
        ),
        continuation_output_indices=(),
    )

    def token_and_logprob(document: Document, slot: OutputSlot) -> tuple[int, float]:
        if slot.index == 1 and document.id.endswith("/left"):
            return 301, -2.0
        if slot.index == 1:
            return 401, -0.2
        return 300 + slot.index, -0.1

    result = run_program(program, mapped_executor(_predict_with(token_and_logprob)))

    assert _state_tokens(result.value, example_id, (0, 1, 2)) == (300, 401, 302)
    overlapping = tuple(
        observation.token_id
        for observation in result.exchanges[0].response.observations
        if observation.slot == forecast_slot(example_id, 1)
    )
    assert overlapping == (301, 401)


def test_windowed_forecast_routes_slots_independently_of_document_order() -> None:
    example_id = "forecast-order"
    left = ContextWindow("left", (10,), (2, 0))
    right = ContextWindow("right", (20,), (3, 1))

    def token_and_logprob(document: Document, slot: OutputSlot) -> tuple[int, float]:
        context_token = document.token_ids[0]
        return context_token * 10 + slot.index, -0.1

    executor = mapped_executor(_predict_with(token_and_logprob))
    forward = run_program(
        windowed_forecast_program(example_id, (left, right), continuation_output_indices=()),
        executor,
    )
    reversed_order = run_program(
        windowed_forecast_program(example_id, (right, left), continuation_output_indices=()),
        executor,
    )

    expected = (100, 201, 102, 203)
    assert _state_tokens(forward.value, example_id, (0, 1, 2, 3)) == expected
    assert _state_tokens(reversed_order.value, example_id, (0, 1, 2, 3)) == expected
