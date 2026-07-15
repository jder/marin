# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

from experiments.probabilistic_dataflow.documents import Document
from experiments.probabilistic_dataflow.mock_windowed import (
    COORDINATE_CHANNEL,
    split_windows,
    windowed_forecast_program,
)
from experiments.probabilistic_dataflow.programs import Prediction, mapped_executor, run


def _predict_with(
    token_and_logprob: Callable[[Document, int], tuple[int, float]],
) -> Callable[[Document], tuple[Prediction, ...]]:
    def predict(document: Document) -> tuple[Prediction, ...]:
        predictions = []
        for position in document.query_positions:
            index = dict(document.tokens[position].features)[COORDINATE_CHANNEL]
            token_id, logprob = token_and_logprob(document, index)
            predictions.append(Prediction(token_id, logprob))
        return tuple(predictions)

    return predict


def test_windowed_forecast_assembles_disjoint_context_shards() -> None:
    split = split_windows(
        "forecast-disjoint",
        ((10, 11), (20, 21)),
        ((0, 1), (2, 3)),
    )
    program = windowed_forecast_program(
        split,
        continuation_output_indices=(),
    )
    executor = mapped_executor(_predict_with(lambda _document, index: (100 + index, -0.1)))

    result = run(program, executor)

    assert result.value == {0: 100, 1: 101, 2: 102, 3: 103}
    assert tuple(document.token_ids for document in result.exchanges[0].documents) == (
        (10, 11, 1, 1),
        (20, 21, 1, 1),
    )


def test_windowed_forecast_materializes_predictions_in_continuation() -> None:
    split = split_windows(
        "forecast-continuation",
        ((10,),),
        ((0, 1),),
    )
    program = windowed_forecast_program(
        split,
        continuation_output_indices=(2, 3),
    )
    executor = mapped_executor(_predict_with(lambda _document, index: (200 + index, -0.1)))

    result = run(program, executor)

    assert result.value == {0: 200, 1: 201, 2: 202, 3: 203}
    continuation = result.exchanges[1].documents[0]
    assert continuation.token_ids == (200, 201, 1, 1)
    assert tuple(
        dict(continuation.tokens[position].features)[COORDINATE_CHANNEL] for position in continuation.query_positions
    ) == (2, 3)


def test_overlapping_windows_select_highest_logprob_prediction() -> None:
    split = split_windows(
        "forecast-overlap",
        ((10,), (20,)),
        ((0, 1), (1, 2)),
    )
    program = windowed_forecast_program(
        split,
        continuation_output_indices=(),
    )

    def token_and_logprob(document: Document, index: int) -> tuple[int, float]:
        if index == 1 and document.token_ids[0] == 10:
            return 301, -2.0
        if index == 1:
            return 401, -0.2
        return 300 + index, -0.1

    result = run(program, mapped_executor(_predict_with(token_and_logprob)))

    assert result.value == {0: 300, 1: 401, 2: 302}
    left_result, right_result = result.exchanges[0].results
    assert (left_result.predictions[1].token_id, right_result.predictions[0].token_id) == (301, 401)


def test_windowed_forecast_uses_generator_indices_independently_of_document_order() -> None:
    def token_and_logprob(document: Document, index: int) -> tuple[int, float]:
        return document.token_ids[0] * 10 + index, -0.1

    executor = mapped_executor(_predict_with(token_and_logprob))
    forward_split = split_windows(
        "forecast-order",
        ((10,), (20,)),
        ((2, 0), (3, 1)),
    )
    forward = run(
        windowed_forecast_program(forward_split, continuation_output_indices=()),
        executor,
    )
    reversed_split = split_windows(
        "forecast-order",
        ((20,), (10,)),
        ((3, 1), (2, 0)),
    )
    reversed_order = run(
        windowed_forecast_program(reversed_split, continuation_output_indices=()),
        executor,
    )

    expected = {0: 100, 1: 201, 2: 102, 3: 203}
    assert forward.value == expected
    assert reversed_order.value == expected
