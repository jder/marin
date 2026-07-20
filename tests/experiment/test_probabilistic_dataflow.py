# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.documents import (
    POSITION_IDS,
    QUERY,
    TARGET_IDS,
    TARGET_WEIGHTS,
    AttentionLayout,
    Coordinate,
    Document,
    pack,
)
from experiments.probabilistic_dataflow.training import (
    ADVECTION_RECORDS,
    SCIENTIFIC_POSITION,
    SyntheticTokenCodec,
    build_synthetic_advection_batch,
    build_synthetic_text_batch,
    record_order_equivariance_error,
    synthetic_advection_document,
    train_cross_domain_smoke,
)

COORDINATE = Coordinate("coordinate")


def test_document_coordinates_use_identity_and_unique_name_shortcuts() -> None:
    left = Coordinate("bar")
    right = Coordinate("bar")
    document = Document((10, 11), {left: (1, 2), right: (3, 4)}, attention=AttentionLayout.FULL)

    assert tuple(document[left]) == (1, 2)
    assert tuple(document[right]) == (3, 4)
    with pytest.raises(AttributeError, match="ambiguous"):
        _ = document.bar


def test_document_add_concatenates_shared_coordinates_and_fills_disjoint_coordinates() -> None:
    context_only = Coordinate("context_only")
    query_only = Coordinate("query_only")
    context = Document(
        (10, 11),
        {COORDINATE: (0, 1), context_only: (5, 6)},
        attention=AttentionLayout.FULL,
    )
    query = Document(
        (1,),
        {COORDINATE: (2,), query_only: (7,), QUERY: (True,)},
        attention=AttentionLayout.FULL,
    )

    document = context + query

    assert tuple(document.token_ids) == (10, 11, 1)
    assert tuple(document[COORDINATE]) == (0, 1, 2)
    assert tuple(document[context_only]) == (5, 6, context_only.missing)
    assert tuple(document[query_only]) == (query_only.missing, query_only.missing, 7)
    assert tuple(document.coordinate) == (0, 1, 2)
    assert document.query_positions == (2,)


def test_synthetic_advection_keeps_targets_out_of_model_inputs() -> None:
    codec = SyntheticTokenCodec()
    document = synthetic_advection_document(codec, seed=0)
    supervised = document[TARGET_WEIGHTS] > 0

    assert len(document) == ADVECTION_RECORDS
    assert np.all(document.token_ids[supervised] == codec.QUERY_ID)
    assert np.all(document[TARGET_IDS][supervised] >= codec.DATA_OFFSET)
    assert tuple(document[SCIENTIFIC_POSITION]) == tuple(range(ADVECTION_RECORDS))
    assert np.all(document[POSITION_IDS] == 0)


def test_text_and_science_share_vocabulary_with_data_dependent_execution() -> None:
    codec = SyntheticTokenCodec()
    text = build_synthetic_text_batch(codec, repetitions=1)
    science = build_synthetic_advection_batch(codec, examples=1, max_seq_len=32)

    assert text.attention == AttentionLayout.CAUSAL
    assert np.all(text[SCIENTIFIC_POSITION] == SCIENTIFIC_POSITION.missing)
    assert np.array_equal(text[POSITION_IDS][0], np.arange(text.token_ids.shape[1]))
    assert np.array_equal(text[TARGET_IDS][:, :-1], text.token_ids[:, 1:])

    assert science.attention == AttentionLayout.FULL
    assert np.all(science[POSITION_IDS] == 0)
    assert np.all(science[SCIENTIFIC_POSITION][science.segment_ids >= 0] >= 0)
    assert text.token_ids.max() < codec.DATA_OFFSET
    science_values = science.token_ids[(science.segment_ids >= 0) & ~science[QUERY]]
    assert science_values.min() >= codec.DATA_OFFSET


def test_scientific_token_logits_are_equivariant_to_serialization_order() -> None:
    assert record_order_equivariance_error() < 1e-5


def test_packing_tracks_query_occurrences_without_logical_output_keys() -> None:
    logical_document = Document(
        (10, 11, 1, 1, 1, 1),
        {
            COORDINATE: (COORDINATE.missing, COORDINATE.missing, 0, 1, 2, 3),
            QUERY: (False, False, True, True, True, True),
            TARGET_IDS: (TARGET_IDS.missing, TARGET_IDS.missing, 20, 21, 22, 23),
        },
        attention=AttentionLayout.FULL,
    )
    left = logical_document.take((0, 2, 3))
    right = logical_document.take((1, 4, 5))
    batch = pack((left, right), max_seq_len=6)

    query_rows, query_positions = np.nonzero(batch[QUERY])
    assert tuple(batch.document_indices[query_rows, query_positions]) == (0, 0, 1, 1)
    assert tuple(batch[COORDINATE][query_rows, query_positions]) == (0, 1, 2, 3)
    assert tuple(batch[TARGET_IDS][query_rows, query_positions]) == (20, 21, 22, 23)


@pytest.mark.slow
def test_one_grug_model_learns_causal_text_and_full_attention_science() -> None:
    result = train_cross_domain_smoke(steps=10, examples_per_task=4)

    assert result.final_loss < result.initial_loss
    assert result.text.final_accuracy > result.text.initial_accuracy
    assert result.science.final_accuracy > result.science.initial_accuracy
    assert result.task_families == ("synthetic_text", "synthetic_advection")
