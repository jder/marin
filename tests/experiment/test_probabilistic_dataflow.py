# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.documents import (
    AttentionLayout,
    Document,
    Output,
    OutputSlot,
    PredictionState,
    PredictionUpdateMode,
    PredictionValue,
    Record,
    Supervision,
    pack_documents,
)
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


def test_synthetic_advection_document_keeps_targets_out_of_model_inputs() -> None:
    codec = SyntheticTokenCodec()
    document = synthetic_advection_document(codec, seed=0)
    supervised = np.asarray(document.loss_weights) > 0

    assert len(document.records) == ADVECTION_RECORDS
    assert np.all(np.asarray(document.token_ids)[supervised] == codec.QUERY_ID)
    assert np.all(np.asarray(document.target_ids)[supervised] >= codec.DATA_OFFSET)
    assert document.feature_ids(SCIENTIFIC_POSITION_CHANNEL) == tuple(range(ADVECTION_RECORDS))
    assert document.rotary_position_ids == (0,) * ADVECTION_RECORDS


def test_text_and_science_share_vocabulary_with_data_dependent_execution() -> None:
    codec = SyntheticTokenCodec()
    text = build_synthetic_text_batch(codec, repetitions=1)
    science = build_synthetic_advection_batch(codec, examples=1, max_seq_len=32)

    assert text.attention_layout == AttentionLayout.CAUSAL
    assert np.all(text.scientific_position_ids == -1)
    assert np.array_equal(text.rotary_position_ids[0], np.arange(text.token_ids.shape[1]))
    assert np.array_equal(text.target_ids[:, :-1], text.token_ids[:, 1:])

    assert science.attention_layout == AttentionLayout.FULL
    assert np.all(science.rotary_position_ids == 0)
    assert np.all(science.scientific_position_ids[science.segment_ids >= 0] >= 0)
    assert text.token_ids.max() < codec.DATA_OFFSET
    science_values = science.token_ids[(science.segment_ids >= 0) & (science.token_ids != codec.QUERY_ID)]
    assert science_values.min() >= codec.DATA_OFFSET


def test_scientific_record_logits_are_equivariant_to_serialization_order() -> None:
    assert record_order_equivariance_error() < 1e-5


def test_one_logical_prediction_assembles_from_different_context_views() -> None:
    slots = tuple(OutputSlot("forecast-0", "future", index) for index in range(4))
    logical_document = Document(
        "forecast-0/full",
        (
            Record(10, position_id=0),
            Record(11, position_id=0),
            *(Record(1, position_id=0, output=Output(slot, Supervision(20 + slot.index))) for slot in slots),
        ),
        AttentionLayout.FULL,
    )
    left = logical_document.selected("forecast-0/left", (0, 2, 3))
    right = logical_document.selected("forecast-0/right", (1, 4, 5))
    packed = pack_documents((left, right), max_seq_len=6)
    sampled = np.zeros_like(packed.token_ids)
    for output in packed.outputs:
        sampled[output.row, output.position] = 30 + output.slot.index
    state = PredictionState().updated(
        packed.prediction_values(sampled),
        mode=PredictionUpdateMode.REQUIRE_EMPTY,
    )

    assert left.token_ids == (10, 1, 1)
    assert right.token_ids == (11, 1, 1)
    assert tuple(state.value(slot) for slot in slots) == (30, 31, 32, 33)


def test_prediction_state_replacement_requires_existing_unique_slots() -> None:
    existing = OutputSlot("forecast-0", "future", 0)
    missing = OutputSlot("forecast-0", "future", 1)
    state = PredictionState((PredictionValue(existing, 10),))

    with pytest.raises(ValueError, match="requires existing slots"):
        state.updated((PredictionValue(missing, 20),), mode=PredictionUpdateMode.REPLACE)
    with pytest.raises(ValueError, match="same output slot"):
        PredictionState((PredictionValue(existing, 10), PredictionValue(existing, 11)))


@pytest.mark.slow
def test_one_grug_model_learns_causal_text_and_full_attention_science() -> None:
    result = train_cross_domain_smoke(steps=10, examples_per_task=4)

    assert result.final_loss < result.initial_loss
    assert result.text.final_accuracy > result.text.initial_accuracy
    assert result.science.final_accuracy > result.science.initial_accuracy
    assert result.task_families == ("synthetic_text", "synthetic_advection")
