# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.documents import AttentionLayout, Document, Token, pack
from experiments.probabilistic_dataflow.training import (
    ADVECTION_RECORDS,
    SCIENTIFIC_POSITION_CHANNEL,
    SyntheticTokenCodec,
    build_synthetic_advection_batch,
    build_synthetic_text_batch,
    record_order_equivariance_error,
    synthetic_advection_document,
    train_cross_domain_smoke,
)


def test_synthetic_advection_keeps_targets_out_of_model_inputs() -> None:
    codec = SyntheticTokenCodec()
    document = synthetic_advection_document(codec, seed=0)
    supervised = np.asarray(document.loss_weights) > 0

    assert len(document.tokens) == ADVECTION_RECORDS
    assert np.all(np.asarray(document.token_ids)[supervised] == codec.QUERY_ID)
    assert np.all(np.asarray(document.target_ids)[supervised] >= codec.DATA_OFFSET)
    assert document.feature_ids(SCIENTIFIC_POSITION_CHANNEL) == tuple(range(ADVECTION_RECORDS))
    assert document.rotary_position_ids == (0,) * ADVECTION_RECORDS


def test_text_and_science_share_vocabulary_with_data_dependent_execution() -> None:
    codec = SyntheticTokenCodec()
    text = build_synthetic_text_batch(codec, repetitions=1)
    science = build_synthetic_advection_batch(codec, examples=1, max_seq_len=32)

    assert text.attention_layout == AttentionLayout.CAUSAL
    assert np.all(text.feature_ids(SCIENTIFIC_POSITION_CHANNEL) == -1)
    assert np.array_equal(text.rotary_position_ids[0], np.arange(text.token_ids.shape[1]))
    assert np.array_equal(text.target_ids[:, :-1], text.token_ids[:, 1:])

    assert science.attention_layout == AttentionLayout.FULL
    assert np.all(science.rotary_position_ids == 0)
    assert np.all(science.feature_ids(SCIENTIFIC_POSITION_CHANNEL)[science.segment_ids >= 0] >= 0)
    assert text.token_ids.max() < codec.DATA_OFFSET
    science_values = science.token_ids[(science.segment_ids >= 0) & ~science.query_mask]
    assert science_values.min() >= codec.DATA_OFFSET


def test_scientific_token_logits_are_equivariant_to_serialization_order() -> None:
    assert record_order_equivariance_error() < 1e-5


def test_packing_tracks_query_occurrences_without_logical_output_keys() -> None:
    logical_document = Document(
        "forecast/full",
        (
            Token(10),
            Token(11),
            *(Token(1, features=(("coordinate", index),), query=True, target_id=20 + index) for index in range(4)),
        ),
        AttentionLayout.FULL,
    )
    left = logical_document.selected("forecast/left", (0, 2, 3))
    right = logical_document.selected("forecast/right", (1, 4, 5))
    batch = pack((left, right), max_seq_len=6)

    query_rows, query_positions = np.nonzero(batch.query_mask)
    assert tuple(batch.document_indices[query_rows, query_positions]) == (0, 0, 1, 1)
    assert tuple(batch.feature_ids("coordinate")[query_rows, query_positions]) == (0, 1, 2, 3)
    assert tuple(batch.target_ids[query_rows, query_positions]) == (20, 21, 22, 23)


@pytest.mark.slow
def test_one_grug_model_learns_causal_text_and_full_attention_science() -> None:
    result = train_cross_domain_smoke(steps=10, examples_per_task=4)

    assert result.final_loss < result.initial_loss
    assert result.text.final_accuracy > result.text.initial_accuracy
    assert result.science.final_accuracy > result.science.initial_accuracy
    assert result.task_families == ("synthetic_text", "synthetic_advection")
