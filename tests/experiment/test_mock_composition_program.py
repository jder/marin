# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from collections.abc import Iterator

import pytest

from experiments.probabilistic_dataflow.documents import Document
from experiments.probabilistic_dataflow.mock_composition import (
    ACCEPT_TOKEN,
    CHEMISTRY_TASK,
    DETAILED_PLAN_TOKEN,
    GEOMETRY_TASK,
    PLAN_TASK,
    QUERY_TOKEN,
    REJECT_TOKEN,
    TASK_CHANNEL,
    VERIFICATION_TASK,
    SpecialistResources,
    composition_program,
)
from experiments.probabilistic_dataflow.programs import Exchange, Executor, Prediction, mapped_executor, run

COARSE_GEOMETRY_TOKEN = 30
REFINED_GEOMETRY_TOKEN = 31
RETRIED_GEOMETRY_TOKEN = 32
CHEMISTRY_TOKEN = 40


def test_parallel_adaptive_specialists_complete_unequal_steps_before_verification() -> None:
    resources = SpecialistResources()
    result = run(
        composition_program("sample", resources),
        _scripted_executor(iter((ACCEPT_TOKEN,))),
    )

    assert result.value.geometry_token == REFINED_GEOMETRY_TOKEN
    assert result.value.chemistry_token == CHEMISTRY_TOKEN
    assert result.value.accepted
    assert result.value.retries == 0
    assert _query_tasks(result.exchanges) == (
        (PLAN_TASK,),
        (GEOMETRY_TASK, CHEMISTRY_TASK),
        (GEOMETRY_TASK,),
        (VERIFICATION_TASK,),
    )
    assert resources.active == set()


def test_composition_retries_geometry_with_previous_predictions() -> None:
    resources = SpecialistResources()
    result = run(
        composition_program("sample", resources),
        _scripted_executor(iter((REJECT_TOKEN, ACCEPT_TOKEN))),
    )

    assert result.value.geometry_token == RETRIED_GEOMETRY_TOKEN
    assert result.value.chemistry_token == CHEMISTRY_TOKEN
    assert result.value.accepted
    assert result.value.retries == 1

    verification_documents = [
        document
        for exchange in result.exchanges
        for document in exchange.documents
        if _query_task(document) == VERIFICATION_TASK
    ]
    assert [document.token_ids for document in verification_documents] == [
        (REFINED_GEOMETRY_TOKEN, CHEMISTRY_TOKEN, QUERY_TOKEN),
        (RETRIED_GEOMETRY_TOKEN, CHEMISTRY_TOKEN, QUERY_TOKEN),
    ]
    assert resources.active == set()


def test_parallel_specialist_resources_are_released_when_execution_fails() -> None:
    resources = SpecialistResources()
    working_executor = _scripted_executor(iter((ACCEPT_TOKEN,)))

    def fail_during_specialists(documents: tuple[Document, ...]):
        if any(_query_task(document) == GEOMETRY_TASK for document in documents):
            raise RuntimeError("specialist backend failed")
        return working_executor(documents)

    with pytest.raises(RuntimeError, match="specialist backend failed"):
        run(composition_program("sample", resources), fail_during_specialists)

    assert resources.active == set()
    assert Counter(resources.acquired) == Counter({"geometry": 1, "chemistry": 1})
    assert Counter(resources.released) == Counter(resources.acquired)


def _scripted_executor(verification_tokens: Iterator[int]) -> Executor:
    def predict(document: Document) -> tuple[Prediction, ...]:
        task = _query_task(document)
        if task == PLAN_TASK:
            token = DETAILED_PLAN_TOKEN
        elif task == CHEMISTRY_TASK:
            token = CHEMISTRY_TOKEN
        elif task == VERIFICATION_TASK:
            token = next(verification_tokens)
        elif document.name.endswith("/coarse"):
            token = COARSE_GEOMETRY_TOKEN
        elif document.name.endswith("/refine"):
            token = REFINED_GEOMETRY_TOKEN
        elif document.name.endswith("/retry"):
            token = RETRIED_GEOMETRY_TOKEN
        else:
            raise AssertionError(f"Unexpected document {document.name}")
        return (Prediction(token, logprob=0.0),)

    return mapped_executor(predict)


def _query_tasks(exchanges: tuple[Exchange, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(_query_task(document) for document in exchange.documents) for exchange in exchanges)


def _query_task(document: Document) -> int:
    (position,) = document.query_positions
    return dict(document.tokens[position].features)[TASK_CHANNEL]
