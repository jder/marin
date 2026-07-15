# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.documents import AttentionLayout, Document, Token
from experiments.probabilistic_dataflow.programs import (
    GENERATED_ORIGINS,
    Origin,
    PackedSamples,
    Prediction,
    ProgramReplayError,
    mapped_executor,
    packed_executor,
    parallel,
    replay,
    run,
    run_many,
)


def _query_document(name: str, *, layout: AttentionLayout = AttentionLayout.FULL) -> Document:
    return Document(name, (Token(1, query=True),), layout)


def test_runner_mixes_ready_programs_without_crossing_yield_barriers() -> None:
    def chained_program():
        (first,) = yield (_query_document("chain/proposal"),)
        first_token = first.predictions[0].token_id
        followup = Document(
            "chain/followup",
            (Token(first_token), Token(1, query=True)),
            AttentionLayout.FULL,
        )
        (second,) = yield (followup,)
        return first_token, second.predictions[0].token_id

    def independent_program():
        (result,) = yield (_query_document("independent/proposal"),)
        return result.predictions[0].token_id

    token_by_document = {
        "chain/proposal": 10,
        "chain/followup": 11,
        "independent/proposal": 20,
    }
    ready_waves = []
    predict_documents = mapped_executor(
        lambda document: tuple(Prediction(token_by_document[document.name], -0.1) for _ in document.query_positions)
    )

    def execute(documents):
        ready_waves.append(tuple(document.name for document in documents))
        return predict_documents(documents)

    chained, independent = run_many((chained_program(), independent_program()), execute)

    assert chained.value == (10, 11)
    assert independent.value == 20
    assert ready_waves == [
        ("chain/proposal", "independent/proposal"),
        ("chain/followup",),
    ]
    assert chained.exchanges[1].documents[0].token_ids == (10, 1)


def test_runner_closes_suspended_programs_when_execution_fails() -> None:
    closed = []

    def program(name: str):
        try:
            yield (_query_document(name),)
        finally:
            closed.append(name)

    def fail(_documents):
        raise RuntimeError("model backend failed")

    with pytest.raises(RuntimeError, match="model backend failed"):
        run_many((program("first"), program("second")), fail)

    assert set(closed) == {"first", "second"}


def test_parallel_slices_multi_document_results_by_child_occurrence() -> None:
    def child(context_tokens: tuple[int, int]):
        documents = tuple(
            Document(
                "repeated-document-name",
                (Token(context_token), Token(1, query=True)),
                AttentionLayout.FULL,
            )
            for context_token in context_tokens
        )
        results = yield documents
        return tuple(result.predictions[0].token_id for result in results)

    executor = mapped_executor(lambda document: (Prediction(document.token_ids[0], -0.1),))
    result = run(parallel((child((11, 12)), child((21, 22)))), executor)

    assert result.value == ((11, 12), (21, 22))


def test_parallel_closes_sibling_when_child_resume_fails() -> None:
    sibling_closed = []

    def failing_program():
        yield (_query_document("failing"),)
        raise RuntimeError("child resume failed")

    def suspended_program():
        try:
            yield (_query_document("suspended"),)
        finally:
            sibling_closed.append(True)

    executor = mapped_executor(lambda _document: (Prediction(1, -0.1),))
    with pytest.raises(RuntimeError, match="child resume failed"):
        run(parallel((failing_program(), suspended_program())), executor)

    assert sibling_closed == [True]


def test_run_boundary_controls_accepted_result_origins() -> None:
    resumed = []

    def program():
        (result,) = yield (_query_document("origin"),)
        resumed.append(True)
        return result.origin

    executor = mapped_executor(
        lambda _document: (Prediction(5, -0.2),),
        origin=Origin.CORRUPTED,
    )

    with pytest.raises(ValueError, match="origin 'corrupted'"):
        run(program(), executor)
    assert resumed == []
    assert run(program(), executor, accepted_origins=GENERATED_ORIGINS).value == Origin.CORRUPTED


def test_runner_rejects_wrong_prediction_count_before_resuming_program() -> None:
    resumed = []

    def program():
        yield (_query_document("missing-prediction"),)
        resumed.append(True)

    with pytest.raises(ValueError, match="returned 0 predictions, expected 1"):
        run(program(), mapped_executor(lambda _document: ()))
    assert resumed == []


def test_packed_executor_partitions_layouts_and_routes_by_document_occurrence() -> None:
    def program(layout: AttentionLayout):
        (result,) = yield (_query_document("shared-document-name", layout=layout),)
        return result.predictions[0].token_id

    observed_layouts = []

    def predict(batch):
        observed_layouts.append(batch.attention_layout)
        token_ids = np.zeros_like(batch.token_ids)
        logprobs = np.zeros_like(batch.loss_weights)
        token_ids[batch.query_mask] = 30 if batch.attention_layout == AttentionLayout.FULL else 40
        logprobs[batch.query_mask] = -0.1
        return PackedSamples(token_ids, logprobs)

    full, causal = run_many(
        (
            program(AttentionLayout.FULL),
            program(AttentionLayout.CAUSAL),
        ),
        packed_executor(predict, max_seq_len=4),
    )

    assert full.value == 30
    assert causal.value == 40
    assert set(observed_layouts) == {AttentionLayout.FULL, AttentionLayout.CAUSAL}


def test_program_transcript_replays_branch_and_detects_divergence() -> None:
    def program(*, changed_detail: bool = False, changed_content: bool = False):
        (plan,) = yield (_query_document("replay/plan"),)
        plan_token = plan.predictions[0].token_id
        if plan_token != 1:
            return plan_token

        detail_name = "replay/changed-detail" if changed_detail else "replay/detail"
        detail_document = _query_document(detail_name)
        if changed_content:
            detail_document = Document(
                detail_name,
                (Token(99), *detail_document.tokens),
                AttentionLayout.FULL,
            )
        (detail,) = yield (detail_document,)
        return detail.predictions[0].token_id

    token_by_document = {"replay/plan": 1, "replay/detail": 9}
    executor = mapped_executor(
        lambda document: tuple(Prediction(token_by_document[document.name], -0.1) for _ in document.query_positions)
    )
    original = run(program(), executor)
    replayed = replay(program(), original.exchanges)

    assert replayed.value == original.value == 9
    assert replayed.exchanges == original.exchanges
    with pytest.raises(ProgramReplayError, match=r"turn 1 \(document 0 differs\)"):
        replay(program(changed_detail=True), original.exchanges)
    with pytest.raises(ProgramReplayError, match=r"turn 1 \(document 0 differs\)"):
        replay(program(changed_content=True), original.exchanges)
