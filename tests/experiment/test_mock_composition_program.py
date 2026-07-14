# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from collections.abc import Iterator

import pytest

from experiments.probabilistic_dataflow.documents import Document
from experiments.probabilistic_dataflow.mock_composition import (
    ACCEPT_TOKEN,
    DETAILED_PLAN_TOKEN,
    QUERY_TOKEN,
    REJECT_TOKEN,
    SpecialistResources,
    composition_program,
)
from experiments.probabilistic_dataflow.programs import (
    DocumentExecutor,
    DocumentRequest,
    DocumentResponse,
    PredictionObservation,
    ProgramExchange,
    mapped_executor,
    run_program,
)

COARSE_GEOMETRY_TOKEN = 30
REFINED_GEOMETRY_TOKEN = 31
RETRIED_GEOMETRY_TOKEN = 32
CHEMISTRY_TOKEN = 40


def test_parallel_adaptive_specialists_complete_unequal_steps_before_verification() -> None:
    resources = SpecialistResources()
    run = run_program(
        composition_program("sample", resources),
        _scripted_executor(iter((ACCEPT_TOKEN,))),
    )

    assert run.value.geometry_token == REFINED_GEOMETRY_TOKEN
    assert run.value.chemistry_token == CHEMISTRY_TOKEN
    assert run.value.accepted
    assert run.value.retries == 0
    assert _output_names(run.exchanges) == (
        ("plan",),
        ("geometry", "chemistry"),
        ("geometry",),
        ("accepted",),
    )
    assert resources.active == set()


def test_composition_program_retries_rejected_geometry_with_previous_predictions() -> None:
    resources = SpecialistResources()
    run = run_program(
        composition_program("sample", resources),
        _scripted_executor(iter((REJECT_TOKEN, ACCEPT_TOKEN))),
    )

    assert run.value.geometry_token == RETRIED_GEOMETRY_TOKEN
    assert run.value.chemistry_token == CHEMISTRY_TOKEN
    assert run.value.accepted
    assert run.value.retries == 1

    verification_documents = [
        exchange.request.documents[0]
        for exchange in run.exchanges
        if "accepted" in _document_output_names(exchange.request.documents[0])
    ]
    assert [document.token_ids for document in verification_documents] == [
        (REFINED_GEOMETRY_TOKEN, CHEMISTRY_TOKEN, QUERY_TOKEN),
        (RETRIED_GEOMETRY_TOKEN, CHEMISTRY_TOKEN, QUERY_TOKEN),
    ]
    assert resources.active == set()


def test_parallel_specialist_resources_are_released_when_execution_fails() -> None:
    resources = SpecialistResources()
    working_executor = _scripted_executor(iter((ACCEPT_TOKEN,)))

    def fail_during_specialists(requests: tuple[DocumentRequest, ...]) -> tuple[DocumentResponse, ...]:
        if any(
            output.value_name == "geometry"
            for request in requests
            for document in request.documents
            for output in document.output_slots
            if output is not None
        ):
            raise RuntimeError("specialist backend failed")
        return working_executor(requests)

    with pytest.raises(RuntimeError, match="specialist backend failed"):
        run_program(composition_program("sample", resources), fail_during_specialists)

    assert resources.active == set()
    assert Counter(resources.acquired) == Counter({"geometry": 1, "chemistry": 1})
    assert Counter(resources.released) == Counter(resources.acquired)


def _scripted_executor(verification_tokens: Iterator[int]) -> DocumentExecutor:
    def predict(document: Document) -> tuple[PredictionObservation, ...]:
        output_slots = tuple(slot for slot in document.output_slots if slot is not None)
        assert len(output_slots) == 1
        slot = output_slots[0]
        if slot.value_name == "plan":
            token = DETAILED_PLAN_TOKEN
        elif slot.value_name == "chemistry":
            token = CHEMISTRY_TOKEN
        elif slot.value_name == "accepted":
            token = next(verification_tokens)
        elif document.id.endswith("/coarse"):
            token = COARSE_GEOMETRY_TOKEN
        elif document.id.endswith("/refine"):
            token = REFINED_GEOMETRY_TOKEN
        elif document.id.endswith("/retry"):
            token = RETRIED_GEOMETRY_TOKEN
        else:
            raise AssertionError(f"Unexpected document {document.id}")
        return (PredictionObservation(slot, token, logprob=0.0),)

    return mapped_executor(predict)


def _output_names(exchanges: tuple[ProgramExchange, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(name for document in exchange.request.documents for name in _document_output_names(document))
        for exchange in exchanges
    )


def _document_output_names(document: Document) -> tuple[str, ...]:
    return tuple(slot.value_name for slot in document.output_slots if slot is not None)
