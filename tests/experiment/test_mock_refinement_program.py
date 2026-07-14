# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.probabilistic_dataflow.documents import Document
from experiments.probabilistic_dataflow.mock_refinement import (
    iterative_refinement_program,
    supervised_iterative_refinement_program,
)
from experiments.probabilistic_dataflow.programs import (
    GENERATED_FEEDBACK,
    SAMPLED_FEEDBACK,
    FeedbackOrigin,
    PredictionObservation,
    mapped_executor,
    run_program,
    supervised_executor,
)


def test_unlabeled_refinement_replaces_low_confidence_values_across_documents() -> None:
    def predict(document: Document) -> tuple[PredictionObservation, ...]:
        is_proposal = sum(record.output is None for record in document.records) == 2
        return tuple(
            PredictionObservation(
                slot,
                (10 if is_proposal else 20) + slot.index,
                -0.1 if not is_proposal or slot.index == 0 else -2.0,
            )
            for slot in document.output_slots
            if slot is not None
        )

    run = run_program(
        iterative_refinement_program(
            example_id="field-0",
            observed_token_ids=(5, 6),
            num_outputs=4,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=3,
            accepted_feedback=SAMPLED_FEEDBACK,
        ),
        mapped_executor(predict),
    )

    assert run.value.token_ids == (10, 21, 22, 23)
    assert run.value.refinement_rounds == 1
    assert len(run.exchanges[0].request.documents) == 2
    assert all(
        record.output is None or record.output.supervision is None
        for exchange in run.exchanges
        for document in exchange.request.documents
        for record in document.records
    )

    refinement_documents = run.exchanges[1].request.documents
    assert len(refinement_documents) == 2
    refined_slots = {slot for document in refinement_documents for slot in document.output_slots if slot is not None}
    assert {slot.index for slot in refined_slots} == {1, 2, 3}
    for document in refinement_documents:
        context_token_ids = {record.input_id for record in document.records if record.output is None}
        assert {10, 11, 12, 13} <= context_token_ids


def test_supervised_documents_keep_labels_out_of_sampled_feedback() -> None:
    targets = (100, 101)
    run = run_program(
        supervised_iterative_refinement_program(
            example_id="field-labels",
            observed_token_ids=(5,),
            target_token_ids=targets,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=0,
            accepted_feedback=SAMPLED_FEEDBACK,
        ),
        mapped_executor(
            lambda document: tuple(
                PredictionObservation(slot, 10 + slot.index, -0.1)
                for slot in document.output_slots
                if slot is not None
            )
        ),
    )

    assert run.value.token_ids == (10, 11)
    assert run.exchanges[0].request.documents[0].target_ids[-2:] == targets


def test_refinement_stops_when_replacement_becomes_confident() -> None:
    round_index = 0

    def execute(requests):
        nonlocal round_index
        current_round = round_index
        round_index += 1
        return mapped_executor(
            lambda document: tuple(
                PredictionObservation(
                    slot,
                    token_id=10 * (current_round + 1) + slot.index,
                    logprob=-0.1 if slot.index == 1 or current_round == 2 else -2.0,
                )
                for slot in document.output_slots
                if slot is not None
            )
        )(requests)

    run = run_program(
        iterative_refinement_program(
            example_id="field-1",
            observed_token_ids=(5,),
            num_outputs=2,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=5,
            accepted_feedback=SAMPLED_FEEDBACK,
        ),
        execute,
    )

    assert run.value.token_ids == (30, 11)
    assert run.value.refinement_rounds == 2
    assert len(run.exchanges) == 3


def test_feedback_contract_rejects_supervision_before_resuming_program() -> None:
    program = supervised_iterative_refinement_program(
        example_id="field-2",
        observed_token_ids=(5,),
        target_token_ids=(100, 101),
        outputs_per_document=2,
        minimum_logprob=-0.5,
        max_refinement_rounds=2,
        accepted_feedback=SAMPLED_FEEDBACK,
    )

    with pytest.raises(ValueError, match="feedback origin 'supervised'"):
        run_program(program, supervised_executor)


def test_corrupted_proposal_can_drive_refinement_context() -> None:
    origins = iter((FeedbackOrigin.CORRUPTED, FeedbackOrigin.SAMPLED))

    def execute(requests):
        origin = next(origins)
        token_base = 70 if origin == FeedbackOrigin.CORRUPTED else 80
        return mapped_executor(
            lambda document: tuple(
                PredictionObservation(slot, token_base + slot.index, -2.0 if token_base == 70 else -0.1)
                for slot in document.output_slots
                if slot is not None
            ),
            origin=origin,
        )(requests)

    run = run_program(
        iterative_refinement_program(
            example_id="field-corrupted",
            observed_token_ids=(5,),
            num_outputs=2,
            outputs_per_document=2,
            minimum_logprob=-0.5,
            max_refinement_rounds=1,
            accepted_feedback=GENERATED_FEEDBACK,
        ),
        execute,
    )

    refinement_context = tuple(
        record.input_id
        for record in run.exchanges[1].request.documents[0].records
        if record.output is None
    )
    assert refinement_context == (5, 70, 71)
    assert run.value.token_ids == (80, 81)
