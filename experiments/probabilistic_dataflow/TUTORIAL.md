# Tutorial: from one document to adaptive scientific inference

## Orientation

The reusable API in this experiment has two data structures and one control-flow
convention:

- a `Document` contains the records for one isolated transformer call;
- an `OutputSlot` identifies a logical prediction independently of the document
  that requests it;
- a Python generator yields a `DocumentRequest` and receives the corresponding
  `DocumentResponse`.

The generator owns scientific control flow. It can split one prediction over
several documents, select among overlapping observations, feed predictions into
later documents, loop until a stopping condition, and delegate to subprograms.
The executor owns model batching and result routing.

This tutorial builds that path in five steps:

1. encode one scalar prediction as a document;
2. yield that document from a program and run it;
3. split and overlap a logical field across context windows;
4. add adaptive refinement and subprogram composition;
5. pack the same document representation for training.

The code is an experiment under `experiments/probabilistic_dataflow`, not a
production Marin API.

## 1. Encode one scalar prediction

Suppose the discretized current value is token `35`, and the training target is
token `37`. The model input has two records: the observed value and a query for
the future value.

```python
from experiments.probabilistic_dataflow.documents import (
    AttentionLayout,
    Document,
    FeatureId,
    Output,
    OutputSlot,
    Record,
    Supervision,
)

QUERY_ID = 1
CURRENT_ID = 35
TARGET_ID = 37

future_slot = OutputSlot("scalar-0", "future", 0)
current_feature = (FeatureId("scientific_position", 0),)
future_feature = (FeatureId("scientific_position", 1),)

inference_document = Document(
    "scalar-0/inference",
    (
        Record(CURRENT_ID, position_id=0, features=current_feature),
        Record(
            QUERY_ID,
            position_id=0,
            features=future_feature,
            output=Output(future_slot),
        ),
    ),
    AttentionLayout.FULL,
)

training_document = Document(
    "scalar-0/training",
    (
        Record(CURRENT_ID, position_id=0, features=current_feature),
        Record(
            QUERY_ID,
            position_id=0,
            features=future_feature,
            output=Output(future_slot, Supervision(TARGET_ID)),
        ),
    ),
    AttentionLayout.FULL,
)
```

The inference and training documents have the same model inputs:

```python
assert inference_document.token_ids == training_document.token_ids == (35, 1)
assert inference_document.target_ids == (-1, -1)
assert training_document.target_ids == (-1, 37)
```

`Supervision` is label metadata on an output record. Token `37` is absent from
`training_document.token_ids`, so full attention cannot leak the target into the
model input. Both records use rotary position `0`; their `scientific_position`
features carry the scientific identity used by the existing scientific Grug
wrapper.

`future_slot` is the identity of the requested value. The document ID is only a
description of this occurrence. Another document can request the same slot with
a different context view.

## 2. Yield documents from a program

A document program is a normal Python generator. One `yield` submits a barriered
wave of documents and evaluates to the routed response when the program resumes.

```python
from experiments.probabilistic_dataflow.programs import (
    GENERATED_FEEDBACK,
    SAMPLED_FEEDBACK,
    DocumentProgram,
    DocumentRequest,
    PredictionObservation,
    disjoint_prediction_observations,
    mapped_executor,
    run_program,
)


def scalar_program(document: Document) -> DocumentProgram[int]:
    response = yield DocumentRequest(
        "scalar-0/predict",
        (document,),
        SAMPLED_FEEDBACK,
    )
    observations = disjoint_prediction_observations(response)
    return observations[0].token_id


def fake_predict(document: Document) -> tuple[PredictionObservation, ...]:
    return tuple(
        PredictionObservation(slot, token_id=38, logprob=-0.2)
        for slot in document.output_slots
        if slot is not None
    )


run = run_program(
    scalar_program(inference_document),
    mapped_executor(fake_predict),
)
assert run.value == 38
```

`mapped_executor` adapts a per-document function for small local programs and
tests. A model-backed executor receives all ready requests, packs compatible
documents, runs the model, and reconstructs the same `DocumentResponse` shape.
The generator does not change.

Each `DocumentResult` corresponds positionally to one requested document.
`results[i]` satisfies `request.documents[i]`. This remains unambiguous when
document IDs repeat.

`run_programs` advances several independent generators together. It collects
one ready request from each program, executes that wave, and resumes each
generator with only its response. A later request from one program cannot cross
that program's preceding yield barrier.

## 3. Split a prediction across context windows

One logical field can span several documents. `ContextWindow` names the context
tokens and logical field indices requested by each document:

```python
from experiments.probabilistic_dataflow.mock_windowed import (
    ContextWindow,
    forecast_slot,
    windowed_forecast_program,
)

windows = (
    ContextWindow("left", context_token_ids=(10, 11), output_indices=(0, 1)),
    ContextWindow("right", context_token_ids=(20, 21), output_indices=(1, 2)),
)
```

Both windows request index `1`. The response keeps those two observations
separate until the program applies `highest_logprob_observations`.

```python
def window_predict(document: Document) -> tuple[PredictionObservation, ...]:
    context_token = document.token_ids[0]
    observations = []
    for slot in document.output_slots:
        if slot is None:
            continue
        if slot.index == 1 and context_token == 10:
            observations.append(PredictionObservation(slot, token_id=301, logprob=-2.0))
        elif slot.index == 1:
            observations.append(PredictionObservation(slot, token_id=401, logprob=-0.2))
        else:
            observations.append(PredictionObservation(slot, token_id=300 + slot.index, logprob=-0.1))
    return tuple(observations)


run = run_program(
    windowed_forecast_program(
        "forecast-0",
        windows,
        continuation_output_indices=(3,),
    ),
    mapped_executor(window_predict),
)

state = run.value
assert tuple(state.value(forecast_slot("forecast-0", index)) for index in range(4)) == (
    300,
    401,
    302,
    303,
)
```

The right window wins index `1` because `-0.2` is greater than `-2.0`. The
program commits one value per slot to `PredictionState`, materializes indices
`0`, `1`, and `2` as context records, then yields a continuation document for
index `3`.

`PredictionState.updated` makes write intent explicit:

- `REQUIRE_EMPTY` inserts predictions and rejects an existing slot;
- `REPLACE` updates predictions and rejects an unknown slot.

This catches two common routing errors: accidentally committing overlapping
observations without a selection policy, and refining a newly constructed slot
instead of the original prediction.

## 4. Refine and compose programs

### Adaptive partial refinement

`iterative_refinement_program` proposes a field, finds coordinates below a
log-probability threshold, and yields new documents for only those coordinates.
It does not require target values at inference time.

```python
from experiments.probabilistic_dataflow.mock_refinement import iterative_refinement_program

refinement = iterative_refinement_program(
    example_id="field-0",
    observed_token_ids=(5, 6),
    num_outputs=4,
    outputs_per_document=2,
    minimum_logprob=-0.5,
    max_refinement_rounds=3,
    accepted_feedback=SAMPLED_FEEDBACK,
)
```

The first request contains two documents, each requesting two disjoint output
slots. After sampling, the generator stores the four observations and their
log-probabilities. Each refinement document uses tokens from that sampled state
as context. Unselected coordinates remain unchanged.

Training uses the same control flow through an explicit labeled constructor:

```python
from experiments.probabilistic_dataflow.mock_refinement import (
    supervised_iterative_refinement_program,
)

training_refinement = supervised_iterative_refinement_program(
    example_id="field-0",
    observed_token_ids=(5, 6),
    target_token_ids=(100, 101, 102, 103),
    outputs_per_document=2,
    minimum_logprob=-0.5,
    max_refinement_rounds=3,
    accepted_feedback=SAMPLED_FEEDBACK,
)
```

The labels supervise each query record. They do not become refinement context.
For refinement training from deliberately perturbed proposals, use
`GENERATED_FEEDBACK`, which accepts sampled and corrupted observations while
still rejecting supervised feedback as runtime state.

### Sequential and parallel subprograms

Normal generator delegation handles sequential composition:

```python
from experiments.probabilistic_dataflow.mock_composition import (
    SpecialistResources,
    chemistry_program,
    geometry_program,
    planning_program,
    verification_program,
)
from experiments.probabilistic_dataflow.programs import parallel_programs


def specialist_program(
    example_id: str,
    resources: SpecialistResources,
) -> DocumentProgram[tuple[int, int, bool]]:
    plan = yield from planning_program(example_id)
    geometry_token, chemistry_token = yield from parallel_programs(
        (
            geometry_program(example_id, plan, resources),
            chemistry_program(example_id, resources),
        ),
        request_prefix=f"{example_id}/specialists",
    )
    accepted = yield from verification_program(
        example_id,
        geometry_token,
        chemistry_token,
        attempt=0,
    )
    return geometry_token, chemistry_token, accepted
```

`yield from` handles the planning and verification calls sequentially.
`parallel_programs` is the small extra combinator needed for adaptive children
whose ready documents should share execution waves.

The geometry child may yield a coarse document and then a refinement document.
The chemistry child may finish after one document. `parallel_programs` exposes
both initial documents together, resumes each child with its positional slice,
then exposes only geometry's second request. The parent proceeds to verification
after both children return.

See [`mock_composition.py`](mock_composition.py) for planning, unequal specialist
workloads, verification, conditional retry, and cleanup of suspended children
after an executor failure.

## 5. Pack the same documents for training

`pack_documents` converts documents with one attention layout into dense arrays
while retaining document and output locations:

```python
from experiments.probabilistic_dataflow.documents import pack_documents

batch = pack_documents((training_document,), max_seq_len=8)

assert batch.token_ids.shape == (1, 8)
assert batch.target_ids[0, 1] == TARGET_ID
assert batch.loss_weights[0, 1] == 1.0
assert batch.outputs[0].slot == future_slot
```

The packed batch carries:

- input token, feature, and rotary-position IDs;
- segment IDs that isolate documents in attention;
- aligned target IDs and loss weights;
- the physical row and position of each logical `OutputSlot`.

`packed_executor` uses those output locations to route sampled tokens and
log-probabilities back to document occurrences. It groups ready documents by
`AttentionLayout`, so full-attention scientific records and causal text use the
same model parameters in separate dense calls.

Causal text uses the same document representation. The record containing token
`i` predicts the slot for token `i + 1`:

```python
from experiments.probabilistic_dataflow.documents import causal_training_document

text_document = causal_training_document(
    "text-0",
    token_ids=(2, 3, 4, 5),
    sequence_name="text",
)
assert text_document.attention_layout == AttentionLayout.CAUSAL
assert text_document.target_ids == (3, 4, 5, -1)
```

The cross-domain smoke test trains one Grug parameter set on packed causal text
and full-attention scientific documents:

```python
from experiments.probabilistic_dataflow.training import train_cross_domain_smoke

result = train_cross_domain_smoke(steps=80, examples_per_task=8, seed=0)
```

The checked-in CPU smoke reduced combined training loss from `4.1909` to
`0.2022`. This demonstrates model and optimization compatibility on a memorized
synthetic workload. It does not measure held-out scientific prediction,
language quality, or refinement quality.

Run the document-program behavior tests from the repository root:

```bash
uv run pytest -q \
  tests/experiment/test_document_programs.py \
  tests/experiment/test_mock_refinement_program.py \
  tests/experiment/test_mock_windowed_program.py \
  tests/experiment/test_mock_composition_program.py
```
