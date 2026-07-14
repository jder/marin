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
    Record,
    prediction_input_record,
)
from experiments.probabilistic_dataflow.programs import (
    SAMPLED_FEEDBACK,
    DocumentRequest,
    FeedbackOrigin,
    PackedPredictions,
    PredictionObservation,
    ProgramReplayError,
    disjoint_prediction_observations,
    highest_logprob_observations,
    mapped_executor,
    packed_executor,
    parallel_programs,
    replay_program,
    run_program,
    run_programs,
)


def _query_document(document_id: str, slot: OutputSlot) -> Document:
    return Document(
        document_id,
        (Record(1, position_id=0, output=Output(slot)),),
        AttentionLayout.FULL,
    )


def test_runner_mixes_ready_programs_without_crossing_yield_barriers() -> None:
    first_slot = OutputSlot("chain", "first", 0)
    second_slot = OutputSlot("chain", "second", 0)
    independent_slot = OutputSlot("independent", "value", 0)

    def chained_program():
        first = yield DocumentRequest(
            "chain/proposal",
            (_query_document("chain/proposal", first_slot),),
            SAMPLED_FEEDBACK,
        )
        state = PredictionState().updated(
            tuple(observation.prediction_value() for observation in disjoint_prediction_observations(first)),
            mode=PredictionUpdateMode.REQUIRE_EMPTY,
        )
        followup = Document(
            "chain/followup",
            (
                prediction_input_record(state, first_slot, position_id=0),
                Record(1, position_id=0, output=Output(second_slot)),
            ),
            AttentionLayout.FULL,
        )
        second = yield DocumentRequest("chain/followup", (followup,), SAMPLED_FEEDBACK)
        return state.value(first_slot), disjoint_prediction_observations(second)[0].token_id

    def independent_program():
        response = yield DocumentRequest(
            "independent/proposal",
            (_query_document("independent/proposal", independent_slot),),
            SAMPLED_FEEDBACK,
        )
        return disjoint_prediction_observations(response)[0].token_id

    token_by_document = {
        "chain/proposal": 10,
        "chain/followup": 11,
        "independent/proposal": 20,
    }
    ready_waves = []
    predict_documents = mapped_executor(
        lambda document: tuple(
            PredictionObservation(slot, token_by_document[document.id], logprob=-0.1)
            for slot in document.output_slots
            if slot is not None
        )
    )

    def execute(requests):
        ready_waves.append(tuple(request.id for request in requests))
        return predict_documents(requests)

    chained, independent = run_programs((chained_program(), independent_program()), execute)

    assert chained.value == (10, 11)
    assert independent.value == 20
    assert ready_waves == [
        ("chain/proposal", "independent/proposal"),
        ("chain/followup",),
    ]
    assert chained.exchanges[1].request.documents[0].token_ids == (10, 1)


def test_runner_closes_suspended_programs_when_execution_fails() -> None:
    slot = OutputSlot("failure", "value", 0)
    closed = []

    def program():
        try:
            yield DocumentRequest(
                "failure/request",
                (_query_document("failure/document", slot),),
                SAMPLED_FEEDBACK,
            )
        finally:
            closed.append(True)

    def fail(_requests):
        raise RuntimeError("model backend failed")

    with pytest.raises(RuntimeError, match="model backend failed"):
        run_program(program(), fail)

    assert closed == [True]


def test_runner_closes_every_program_when_one_cleanup_raises() -> None:
    slot = OutputSlot("cleanup", "value", 0)
    closed = []

    def program(name: str, *, fail_cleanup: bool):
        try:
            yield DocumentRequest(
                f"cleanup/{name}",
                (_query_document(f"cleanup/{name}", slot),),
                SAMPLED_FEEDBACK,
            )
        finally:
            closed.append(name)
            if fail_cleanup:
                raise RuntimeError(f"{name} cleanup failed")

    with pytest.raises(RuntimeError, match="second cleanup failed"):
        run_programs(
            (
                program("first", fail_cleanup=False),
                program("second", fail_cleanup=True),
            ),
            lambda requests: (_ for _ in ()).throw(RuntimeError("executor failed")),
        )

    assert set(closed) == {"first", "second"}


def test_parallel_programs_return_heterogeneous_values() -> None:
    word_slot = OutputSlot("heterogeneous", "word", 0)
    count_slot = OutputSlot("heterogeneous", "count", 0)

    def word_program():
        response = yield DocumentRequest(
            "heterogeneous/word",
            (_query_document("heterogeneous/word", word_slot),),
            SAMPLED_FEEDBACK,
        )
        return ("word", disjoint_prediction_observations(response)[0].token_id)

    def count_program():
        response = yield DocumentRequest(
            "heterogeneous/count",
            (_query_document("heterogeneous/count", count_slot),),
            SAMPLED_FEEDBACK,
        )
        return disjoint_prediction_observations(response)[0].token_id

    executor = mapped_executor(
        lambda document: tuple(
            PredictionObservation(slot, 7 if slot == word_slot else 3, -0.1)
            for slot in document.output_slots
            if slot is not None
        )
    )
    run = run_program(
        parallel_programs(
            (word_program(), count_program()),
            request_prefix="heterogeneous",
        ),
        executor,
    )

    assert run.value == (("word", 7), 3)


def test_parallel_programs_close_sibling_when_child_resume_fails() -> None:
    first_slot = OutputSlot("resume-failure", "first", 0)
    second_slot = OutputSlot("resume-failure", "second", 0)
    second_closed = []

    def failing_program():
        yield DocumentRequest(
            "resume-failure/first",
            (_query_document("resume-failure/first", first_slot),),
            SAMPLED_FEEDBACK,
        )
        raise RuntimeError("child resume failed")

    def suspended_program():
        try:
            yield DocumentRequest(
                "resume-failure/second",
                (_query_document("resume-failure/second", second_slot),),
                SAMPLED_FEEDBACK,
            )
        finally:
            second_closed.append(True)

    executor = mapped_executor(
        lambda document: tuple(
            PredictionObservation(slot, 1, -0.1) for slot in document.output_slots if slot is not None
        )
    )
    combined = parallel_programs(
        (failing_program(), suspended_program()),
        request_prefix="resume-failure",
    )

    with pytest.raises(RuntimeError, match="child resume failed"):
        run_program(combined, executor)
    assert second_closed == [True]


def test_parallel_programs_slice_multi_document_results_by_occurrence() -> None:
    slot = OutputSlot("parallel-routing", "shared", 0)

    def child(name: str, context_tokens: tuple[int, int]):
        documents = tuple(
            Document(
                "repeated-document-id",
                (
                    Record(context_token, position_id=0),
                    Record(1, position_id=0, output=Output(slot)),
                ),
                AttentionLayout.FULL,
            )
            for context_token in context_tokens
        )
        response = yield DocumentRequest(name, documents, SAMPLED_FEEDBACK)
        return tuple(result.observations[0].token_id for result in response.results)

    executor = mapped_executor(lambda document: (PredictionObservation(slot, document.token_ids[0], -0.1),))
    run = run_program(
        parallel_programs(
            (
                child("parallel-routing/first", (11, 12)),
                child("parallel-routing/second", (21, 22)),
            ),
            request_prefix="parallel-routing",
        ),
        executor,
    )

    assert run.value == ((11, 12), (21, 22))


def test_request_can_explicitly_accept_corrupted_feedback() -> None:
    slot = OutputSlot("origin", "value", 0)

    def program():
        response = yield DocumentRequest(
            "origin/request",
            (_query_document("origin/document", slot),),
            frozenset({FeedbackOrigin.CORRUPTED}),
        )
        return response.results[0].origin

    executor = mapped_executor(
        lambda document: (PredictionObservation(slot, token_id=5, logprob=-0.2),),
        origin=FeedbackOrigin.CORRUPTED,
    )

    assert run_program(program(), executor).value == FeedbackOrigin.CORRUPTED


def test_runner_rejects_unaccepted_feedback_before_resuming_program() -> None:
    slot = OutputSlot("origin-rejection", "value", 0)
    resumed = []

    def program():
        yield DocumentRequest(
            "origin-rejection/request",
            (_query_document("origin-rejection/document", slot),),
            SAMPLED_FEEDBACK,
        )
        resumed.append(True)

    executor = mapped_executor(
        lambda document: (PredictionObservation(slot, token_id=5, logprob=-0.2),),
        origin=FeedbackOrigin.CORRUPTED,
    )

    with pytest.raises(ValueError, match="feedback origin 'corrupted'"):
        run_program(program(), executor)
    assert resumed == []


def test_packed_executor_partitions_layouts_and_routes_repeated_document_ids() -> None:
    full_slot = OutputSlot("full", "value", 0)
    causal_slot = OutputSlot("causal", "value", 0)

    def program(request_id: str, slot: OutputSlot, layout: AttentionLayout):
        document = Document(
            "shared-document-name",
            (Record(1, position_id=0, output=Output(slot)),),
            layout,
        )
        response = yield DocumentRequest(request_id, (document,), SAMPLED_FEEDBACK)
        return disjoint_prediction_observations(response)[0].token_id

    observed_layouts = []

    def predict(batch):
        observed_layouts.append(batch.attention_layout)
        token_ids = np.zeros_like(batch.token_ids)
        logprobs = np.zeros_like(batch.loss_weights)
        for output in batch.outputs:
            token_ids[output.row, output.position] = 30 if output.slot == full_slot else 40
            logprobs[output.row, output.position] = -0.1
        return PackedPredictions(token_ids, logprobs)

    executor = packed_executor(predict, max_seq_len=4)
    full, causal = run_programs(
        (
            program("full/request", full_slot, AttentionLayout.FULL),
            program("causal/request", causal_slot, AttentionLayout.CAUSAL),
        ),
        executor,
    )

    assert full.value == 30
    assert causal.value == 40
    assert set(observed_layouts) == {AttentionLayout.FULL, AttentionLayout.CAUSAL}


def test_packed_executor_preserves_overlapping_document_occurrences() -> None:
    slot = OutputSlot("overlap", "value", 0)

    def program():
        documents = (
            Document(
                "repeated-name",
                (Record(1, position_id=0, output=Output(slot)),),
                AttentionLayout.FULL,
            ),
            Document(
                "repeated-name",
                (Record(1, position_id=0, output=Output(slot)),),
                AttentionLayout.CAUSAL,
            ),
        )
        response = yield DocumentRequest("overlap/request", documents, SAMPLED_FEEDBACK)
        selected = highest_logprob_observations(response)
        return response, selected[0]

    def predict(batch):
        token_ids = np.zeros_like(batch.token_ids)
        logprobs = np.zeros_like(batch.loss_weights)
        token_id, logprob = {
            AttentionLayout.FULL: (30, -0.5),
            AttentionLayout.CAUSAL: (40, -0.1),
        }[batch.attention_layout]
        for output in batch.outputs:
            token_ids[output.row, output.position] = token_id
            logprobs[output.row, output.position] = logprob
        return PackedPredictions(token_ids, logprobs)

    run = run_program(program(), packed_executor(predict, max_seq_len=4))
    response, selected = run.value

    assert tuple(result.document_id for result in response.results) == ("repeated-name", "repeated-name")
    assert tuple(result.observations[0].token_id for result in response.results) == (30, 40)
    assert (selected.token_id, selected.logprob) == (40, pytest.approx(-0.1))


def test_program_transcript_replays_branch_and_detects_divergence() -> None:
    plan_slot = OutputSlot("replay", "plan", 0)
    detail_slot = OutputSlot("replay", "detail", 0)

    def program(*, changed_detail: bool = False, changed_content: bool = False):
        plan = yield DocumentRequest(
            "replay/plan",
            (_query_document("replay/plan", plan_slot),),
            SAMPLED_FEEDBACK,
        )
        plan_token = disjoint_prediction_observations(plan)[0].token_id
        if plan_token == 1:
            detail_id = "replay/changed-detail" if changed_detail else "replay/detail"
            detail_document = _query_document(detail_id, detail_slot)
            if changed_content:
                detail_document = Document(
                    detail_id,
                    (Record(99, position_id=0), *detail_document.records),
                    AttentionLayout.FULL,
                )
            detail = yield DocumentRequest(detail_id, (detail_document,), SAMPLED_FEEDBACK)
            return disjoint_prediction_observations(detail)[0].token_id
        return plan_token

    token_by_document = {"replay/plan": 1, "replay/detail": 9}
    executor = mapped_executor(
        lambda document: tuple(
            PredictionObservation(slot, token_by_document[document.id], logprob=-0.1)
            for slot in document.output_slots
            if slot is not None
        )
    )
    original = run_program(program(), executor)
    replayed = replay_program(program(), original.exchanges)

    assert replayed.value == original.value == 9
    assert replayed.exchanges == original.exchanges
    with pytest.raises(ProgramReplayError, match="diverged at turn 1"):
        replay_program(program(changed_detail=True), original.exchanges)
    with pytest.raises(ProgramReplayError, match=r"turn 1 \(document 0 differs\)"):
        replay_program(program(changed_content=True), original.exchanges)
