# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.documents import (
    QUERY,
    TARGET_WEIGHTS,
    AttentionLayout,
    Coordinate,
    Document,
)
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

ROUTE = Coordinate("route")
DETAIL = Coordinate("detail")
CHAIN_PROPOSAL = 0
CHAIN_FOLLOWUP = 1
INDEPENDENT_PROPOSAL = 2
REPLAY_PLAN = 3
REPLAY_DETAIL = 4


def _query_document(route: int, *, layout: AttentionLayout = AttentionLayout.FULL) -> Document:
    return Document((1,), {QUERY: (True,), ROUTE: (route,)}, attention=layout)


def test_runner_mixes_ready_programs_without_crossing_yield_barriers() -> None:
    def chained_program():
        (first,) = yield (_query_document(CHAIN_PROPOSAL),)
        first_token = first.predictions[0].token_id
        followup = Document(
            (first_token, 1),
            {QUERY: (False, True), ROUTE: (ROUTE.missing, CHAIN_FOLLOWUP)},
            attention=AttentionLayout.FULL,
        )
        (second,) = yield (followup,)
        return first_token, second.predictions[0].token_id

    def independent_program():
        (result,) = yield (_query_document(INDEPENDENT_PROPOSAL),)
        return result.predictions[0].token_id

    token_by_route = {
        CHAIN_PROPOSAL: 10,
        CHAIN_FOLLOWUP: 11,
        INDEPENDENT_PROPOSAL: 20,
    }
    ready_waves = []
    predict_documents = mapped_executor(
        lambda document: tuple(
            Prediction(token_by_route[int(document[ROUTE][position])], -0.1) for position in document.query_positions
        )
    )

    def execute(documents):
        ready_waves.append(tuple(int(document[ROUTE][document.query_positions[0]]) for document in documents))
        return predict_documents(documents)

    chained, independent = run_many((chained_program(), independent_program()), execute)

    assert chained.value == (10, 11)
    assert independent.value == 20
    assert ready_waves == [
        (CHAIN_PROPOSAL, INDEPENDENT_PROPOSAL),
        (CHAIN_FOLLOWUP,),
    ]
    assert tuple(chained.exchanges[1].documents[0].token_ids) == (10, 1)


def test_runner_closes_suspended_programs_when_execution_fails() -> None:
    closed = []

    def program(name: str):
        try:
            yield (_query_document(len(name)),)
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
                (context_token, 1),
                {QUERY: (False, True)},
                attention=AttentionLayout.FULL,
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
        yield (_query_document(0),)
        raise RuntimeError("child resume failed")

    def suspended_program():
        try:
            yield (_query_document(1),)
        finally:
            sibling_closed.append(True)

    executor = mapped_executor(lambda _document: (Prediction(1, -0.1),))
    with pytest.raises(RuntimeError, match="child resume failed"):
        run(parallel((failing_program(), suspended_program())), executor)

    assert sibling_closed == [True]


def test_run_boundary_controls_accepted_result_origins() -> None:
    resumed = []

    def program():
        (result,) = yield (_query_document(0),)
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
        yield (_query_document(0),)
        resumed.append(True)

    with pytest.raises(ValueError, match="returned 0 predictions, expected 1"):
        run(program(), mapped_executor(lambda _document: ()))
    assert resumed == []


def test_packed_executor_partitions_layouts_and_routes_by_document_occurrence() -> None:
    def program(layout: AttentionLayout):
        (result,) = yield (_query_document(0, layout=layout),)
        return result.predictions[0].token_id

    observed_layouts = []

    def predict(batch):
        observed_layouts.append(batch.attention)
        token_ids = np.zeros_like(batch.token_ids)
        logprobs = np.zeros_like(batch[TARGET_WEIGHTS])
        token_ids[batch.query] = 30 if batch.attention == AttentionLayout.FULL else 40
        logprobs[batch.query] = -0.1
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
    def program(*, changed_coordinate: bool = False, changed_content: bool = False):
        (plan,) = yield (_query_document(REPLAY_PLAN),)
        plan_token = plan.predictions[0].token_id
        if plan_token != 1:
            return plan_token

        detail_document = _query_document(REPLAY_DETAIL)
        if changed_coordinate:
            detail_document = Document(
                detail_document.token_ids,
                {coordinate: detail_document[coordinate] for coordinate in detail_document.coordinates} | {DETAIL: (1,)},
                attention=detail_document.attention,
            )
        if changed_content:
            detail_document = Document(
                (99, *detail_document.token_ids),
                {
                    coordinate: (coordinate.missing, *detail_document[coordinate])
                    for coordinate in detail_document.coordinates
                },
                attention=AttentionLayout.FULL,
            )
        (detail,) = yield (detail_document,)
        return detail.predictions[0].token_id

    token_by_route = {REPLAY_PLAN: 1, REPLAY_DETAIL: 9}
    executor = mapped_executor(
        lambda document: tuple(
            Prediction(token_by_route[int(document[ROUTE][position])], -0.1) for position in document.query_positions
        )
    )
    original = run(program(), executor)
    replayed = replay(program(), original.exchanges)

    assert replayed.value == original.value == 9
    assert replayed.exchanges == original.exchanges
    with pytest.raises(ProgramReplayError, match=r"turn 1 \(document 0 differs\)"):
        replay(program(changed_coordinate=True), original.exchanges)
    with pytest.raises(ProgramReplayError, match=r"turn 1 \(document 0 differs\)"):
        replay(program(changed_content=True), original.exchanges)
