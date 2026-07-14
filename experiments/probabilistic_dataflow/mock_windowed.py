# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from experiments.probabilistic_dataflow.documents import (
    AttentionLayout,
    Document,
    Output,
    OutputSlot,
    PredictionState,
    PredictionUpdateMode,
    Record,
    prediction_input_record,
)
from experiments.probabilistic_dataflow.programs import (
    DocumentProgram,
    DocumentRequest,
    SAMPLED_FEEDBACK,
    highest_logprob_observations,
    prediction_values,
)

QUERY_TOKEN_ID = 1
FORECAST_VALUE_NAME = "future"


@dataclass(frozen=True)
class ContextWindow:
    """One context view and the logical forecast indices it predicts."""

    name: str
    context_token_ids: tuple[int, ...]
    output_indices: tuple[int, ...]


def forecast_slot(example_id: str, index: int) -> OutputSlot:
    return OutputSlot(example_id, FORECAST_VALUE_NAME, index)


def windowed_forecast_program(
    example_id: str,
    initial_windows: tuple[ContextWindow, ...],
    *,
    continuation_output_indices: tuple[int, ...],
) -> DocumentProgram[PredictionState]:
    """Predict a field from context windows, then optionally predict a dependent continuation."""
    initial_response = yield DocumentRequest(
        f"{example_id}/initial",
        tuple(_window_document(example_id, window) for window in initial_windows),
        SAMPLED_FEEDBACK,
    )
    selected = highest_logprob_observations(initial_response)
    state = PredictionState().updated(
        prediction_values(selected),
        mode=PredictionUpdateMode.REQUIRE_EMPTY,
    )

    if not continuation_output_indices:
        return state

    continuation = Document(
        f"{example_id}/continuation",
        (
            *(prediction_input_record(state, prediction.slot, position_id=0) for prediction in state.values),
            *(
                Record(QUERY_TOKEN_ID, position_id=0, output=Output(forecast_slot(example_id, index)))
                for index in continuation_output_indices
            ),
        ),
        AttentionLayout.FULL,
    )
    continuation_response = yield DocumentRequest(
        f"{example_id}/continuation",
        (continuation,),
        SAMPLED_FEEDBACK,
    )
    return state.updated(
        prediction_values(highest_logprob_observations(continuation_response)),
        mode=PredictionUpdateMode.REQUIRE_EMPTY,
    )


def _window_document(example_id: str, window: ContextWindow) -> Document:
    records = (
        *(Record(token_id, position_id=0) for token_id in window.context_token_ids),
        *(
            Record(QUERY_TOKEN_ID, position_id=0, output=Output(forecast_slot(example_id, index)))
            for index in window.output_indices
        ),
    )
    return Document(f"{example_id}/{window.name}", records, AttentionLayout.FULL)
