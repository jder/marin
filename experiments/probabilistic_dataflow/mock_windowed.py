# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from experiments.probabilistic_dataflow.documents import AttentionLayout, Document, Token
from experiments.probabilistic_dataflow.programs import Prediction, Program, Result

QUERY_TOKEN_ID = 1
COORDINATE_CHANNEL = "forecast_coordinate"


@dataclass(frozen=True)
class WindowSplit:
    """Window documents and the coordinates used to join their results."""

    example_id: str
    documents: tuple[Document, ...]
    output_indices: tuple[tuple[int, ...], ...]

    def join_results(self, results: tuple[Result, ...]) -> dict[int, int]:
        if len(results) != len(self.documents):
            raise ValueError(f"Received {len(results)} results for {len(self.documents)} window documents")
        candidates: dict[int, list[Prediction]] = {}
        for indices, result in zip(self.output_indices, results, strict=True):
            for index, prediction in zip(indices, result.predictions, strict=True):
                candidates.setdefault(index, []).append(prediction)
        return {
            index: max(predictions, key=lambda prediction: prediction.logprob).token_id
            for index, predictions in candidates.items()
        }


def split_windows(
    example_id: str,
    context_token_ids: tuple[tuple[int, ...], ...],
    output_indices: tuple[tuple[int, ...], ...],
) -> WindowSplit:
    """Build one document per context window and retain its output coordinates."""
    if not context_token_ids:
        raise ValueError("A window split requires at least one context window")
    if len(context_token_ids) != len(output_indices):
        raise ValueError("Window contexts and output coordinates must have the same length")
    documents = tuple(
        _window_document(example_id, window_index, context, indices)
        for window_index, (context, indices) in enumerate(zip(context_token_ids, output_indices, strict=True))
    )
    return WindowSplit(example_id, documents, output_indices)


def windowed_forecast_program(
    initial: WindowSplit,
    *,
    continuation_output_indices: tuple[int, ...],
) -> Program[dict[int, int]]:
    """Predict from overlapping context windows, then optionally continue the field."""
    results = yield initial.documents
    values = initial.join_results(results)

    if not continuation_output_indices:
        return values

    continuation = Document(
        f"{initial.example_id}/continuation",
        (
            *(Token(token_id, features=((COORDINATE_CHANNEL, index),)) for index, token_id in sorted(values.items())),
            *(
                Token(
                    QUERY_TOKEN_ID,
                    features=((COORDINATE_CHANNEL, index),),
                    query=True,
                )
                for index in continuation_output_indices
            ),
        ),
        AttentionLayout.FULL,
    )
    (result,) = yield (continuation,)
    values.update(
        (index, prediction.token_id)
        for index, prediction in zip(continuation_output_indices, result.predictions, strict=True)
    )
    return values


def _window_document(
    example_id: str,
    window_index: int,
    context_token_ids: tuple[int, ...],
    output_indices: tuple[int, ...],
) -> Document:
    tokens = (
        *(Token(token_id) for token_id in context_token_ids),
        *(
            Token(
                QUERY_TOKEN_ID,
                features=((COORDINATE_CHANNEL, index),),
                query=True,
            )
            for index in output_indices
        ),
    )
    return Document(f"{example_id}/window-{window_index}", tokens, AttentionLayout.FULL)
