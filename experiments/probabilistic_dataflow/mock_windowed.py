# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from experiments.probabilistic_dataflow.documents import (
    QUERY,
    AttentionLayout,
    Coordinate,
    Document,
)
from experiments.probabilistic_dataflow.programs import Prediction, Program, Result

QUERY_TOKEN_ID = 1
COORDINATE = Coordinate("coordinate")


def predict_windows(
    context_token_ids: tuple[tuple[int, ...], ...],
    output_indices: tuple[tuple[int, ...], ...],
) -> Program[dict[int, int]]:
    """Predict and join overlapping windows."""
    documents = _window_documents(context_token_ids, output_indices)
    results = yield documents
    return _join_window_results(output_indices, results)


def continue_forecast(
    context: dict[int, int],
    query_indices: tuple[int, ...],
) -> Program[dict[int, int]]:
    """Predict additional coordinates from materialized forecast context."""
    context_items = tuple(sorted(context.items()))
    document = Document(
        tuple(token_id for _index, token_id in context_items),
        {COORDINATE: tuple(index for index, _token_id in context_items)},
        attention=AttentionLayout.FULL,
    ) + Document(
        (QUERY_TOKEN_ID,) * len(query_indices),
        {COORDINATE: query_indices, QUERY: (True,) * len(query_indices)},
        attention=AttentionLayout.FULL,
    )
    (result,) = yield (document,)
    return {index: prediction.token_id for index, prediction in zip(query_indices, result.predictions, strict=True)}


def _window_documents(
    context_token_ids: tuple[tuple[int, ...], ...],
    output_indices: tuple[tuple[int, ...], ...],
) -> tuple[Document, ...]:
    if not context_token_ids:
        raise ValueError("Window prediction requires at least one context window")
    if len(context_token_ids) != len(output_indices):
        raise ValueError("Window contexts and output coordinates must have the same length")
    return tuple(
        _window_document(context, indices) for context, indices in zip(context_token_ids, output_indices, strict=True)
    )


def _join_window_results(
    output_indices: tuple[tuple[int, ...], ...],
    results: tuple[Result, ...],
) -> dict[int, int]:
    if len(results) != len(output_indices):
        raise ValueError(f"Received {len(results)} results for {len(output_indices)} window coordinate sets")
    candidates: dict[int, list[Prediction]] = {}
    for indices, result in zip(output_indices, results, strict=True):
        for index, prediction in zip(indices, result.predictions, strict=True):
            candidates.setdefault(index, []).append(prediction)
    return {
        index: max(predictions, key=lambda prediction: prediction.logprob).token_id
        for index, predictions in candidates.items()
    }


def _window_document(
    context_token_ids: tuple[int, ...],
    output_indices: tuple[int, ...],
) -> Document:
    return Document(context_token_ids, attention=AttentionLayout.FULL) + Document(
        (QUERY_TOKEN_ID,) * len(output_indices),
        {COORDINATE: output_indices, QUERY: (True,) * len(output_indices)},
        attention=AttentionLayout.FULL,
    )
