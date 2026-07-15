# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.probabilistic_dataflow.documents import Document
from experiments.probabilistic_dataflow.mock_refinement import (
    COORDINATE_CHANNEL,
    refine,
)
from experiments.probabilistic_dataflow.programs import (
    GENERATED_ORIGINS,
    Origin,
    Prediction,
    mapped_executor,
    run,
)


def _query_coordinates(document: Document) -> tuple[int, ...]:
    return tuple(dict(document.tokens[position].features)[COORDINATE_CHANNEL] for position in document.query_positions)


def test_unlabeled_refinement_replaces_low_confidence_values_across_documents() -> None:
    def predict(document: Document) -> tuple[Prediction, ...]:
        proposal = "/proposal/" in document.name
        return tuple(
            Prediction(
                (10 if proposal else 20) + coordinate,
                -0.1 if not proposal or coordinate == 0 else -2.0,
            )
            for coordinate in _query_coordinates(document)
        )

    refinement = refine(
        example_id="field-0",
        observed_token_ids=(5, 6),
        num_outputs=4,
        outputs_per_document=2,
        minimum_logprob=-0.5,
        max_refinement_rounds=3,
    )

    def parent_program():
        return (yield from refinement.documents())

    result = run(parent_program(), mapped_executor(predict))

    assert result.value.token_ids == (10, 21, 22, 23)
    assert result.value.refinement_rounds == 1
    assert len(result.exchanges[0].documents) == 2
    assert all(
        token.target_id is None
        for exchange in result.exchanges
        for document in exchange.documents
        for token in document.tokens
    )

    refinement_documents = result.exchanges[1].documents
    assert len(refinement_documents) == 2
    assert {coordinate for document in refinement_documents for coordinate in _query_coordinates(document)} == {
        1,
        2,
        3,
    }
    for document in refinement_documents:
        context_token_ids = {token.input_id for token in document.tokens if not token.query}
        assert {10, 11, 12, 13} <= context_token_ids


def test_supervised_queries_keep_labels_out_of_sampled_feedback() -> None:
    targets = (100, 101)
    result = run(
        refine(
            example_id="field-labels",
            observed_token_ids=(5,),
            num_outputs=len(targets),
            target_token_ids=targets,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=0,
        ).documents(),
        mapped_executor(
            lambda document: tuple(Prediction(10 + coordinate, -0.1) for coordinate in _query_coordinates(document))
        ),
    )

    assert result.value.token_ids == (10, 11)
    assert result.exchanges[0].documents[0].target_ids[-2:] == targets


def test_refinement_stops_when_replacement_becomes_confident() -> None:
    round_index = 0

    def execute(documents):
        nonlocal round_index
        current_round = round_index
        round_index += 1
        return mapped_executor(
            lambda document: tuple(
                Prediction(
                    token_id=10 * (current_round + 1) + coordinate,
                    logprob=-0.1 if coordinate == 1 or current_round == 2 else -2.0,
                )
                for coordinate in _query_coordinates(document)
            )
        )(documents)

    result = run(
        refine(
            example_id="field-1",
            observed_token_ids=(5,),
            num_outputs=2,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=5,
        ).documents(),
        execute,
    )

    assert result.value.token_ids == (30, 11)
    assert result.value.refinement_rounds == 2
    assert len(result.exchanges) == 3


def test_run_rejects_supervision_as_program_feedback() -> None:
    program = refine(
        example_id="field-2",
        observed_token_ids=(5,),
        num_outputs=2,
        target_token_ids=(100, 101),
        outputs_per_document=2,
        minimum_logprob=-0.5,
        max_refinement_rounds=2,
    )
    executor = mapped_executor(
        lambda document: tuple(
            Prediction(document.tokens[position].target_id or 0, 0.0) for position in document.query_positions
        ),
        origin=Origin.SUPERVISED,
    )

    with pytest.raises(ValueError, match="origin 'supervised'"):
        run(program.documents(), executor)


def test_corrupted_proposal_can_drive_refinement_context() -> None:
    origins = iter((Origin.CORRUPTED, Origin.SAMPLED))

    def execute(documents):
        origin = next(origins)
        token_base = 70 if origin == Origin.CORRUPTED else 80
        return mapped_executor(
            lambda document: tuple(
                Prediction(token_base + coordinate, -2.0 if token_base == 70 else -0.1)
                for coordinate in _query_coordinates(document)
            ),
            origin=origin,
        )(documents)

    result = run(
        refine(
            example_id="field-corrupted",
            observed_token_ids=(5,),
            num_outputs=2,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=1,
        ).documents(),
        execute,
        accepted_origins=GENERATED_ORIGINS,
    )

    refinement_context = tuple(token.input_id for token in result.exchanges[1].documents[0].tokens if not token.query)
    assert refinement_context == (5, 70, 71)
    assert result.value.token_ids == (80, 81)
