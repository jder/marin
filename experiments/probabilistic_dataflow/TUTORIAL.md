# Tutorial: direct document generators

## Orientation

A document program has one model-facing data structure and one control-flow
rule:

- a `Document` contains token IDs and aligned coordinates for one isolated
  attention domain;
- a Python generator yields a tuple of documents and receives a tuple of
  positional results.

The generator owns field coordinates, candidate selection, refinement state,
branching, and loops. The executor owns batching and model execution. This
tutorial builds a scalar call, overlapping windows, refinement, parallel
subprograms, and training documents.

The code lives under `experiments/probabilistic_dataflow`; it is not a
production Marin API.

## 1. Encode one prediction

Suppose the current value is token `35` and the training target is token `37`.
The document contains the observed value and a query token for the next value:

```python
from experiments.probabilistic_dataflow.documents import (
    POSITION_IDS,
    QUERY,
    TARGET_IDS,
    AttentionLayout,
    Coordinate,
    Document,
)

QUERY_ID = 1
CURRENT_ID = 35
TARGET_ID = 37
SCIENTIFIC_POSITION = Coordinate("scientific_position")

coordinates = {
    POSITION_IDS: (0, 0),
    SCIENTIFIC_POSITION: (0, 1),
    QUERY: (False, True),
}

inference_document = Document(
    (CURRENT_ID, QUERY_ID),
    coordinates,
    attention=AttentionLayout.FULL,
)
training_document = Document(
    (CURRENT_ID, QUERY_ID),
    coordinates | {TARGET_IDS: (TARGET_IDS.missing, TARGET_ID)},
    attention=AttentionLayout.FULL,
)
```

Inference and training have identical model inputs:

```python
assert tuple(inference_document.token_ids) == tuple(training_document.token_ids) == (35, 1)
assert tuple(inference_document[TARGET_IDS]) == (-1, -1)
assert tuple(training_document[TARGET_IDS]) == (-1, 37)
```

`TARGET_IDS` is metadata for the loss. It is absent from `token_ids`, so the
full-attention query cannot read token `37`. Both tokens use rotary position
`0`; `SCIENTIFIC_POSITION` tells the model which field position each token
represents.

Coordinates are keyed by object identity. A domain may define another
coordinate named `scientific_position` without colliding with this one. Use
`document[SCIENTIFIC_POSITION]` for canonical access. The shortcut
`document.scientific_position` works when the document has only one coordinate
with that name.

## 2. Yield the document

A `Program[T]` is a normal Python generator. One `yield` submits a tuple of
documents and evaluates to one `Result` per document when execution finishes:

```python
from experiments.probabilistic_dataflow.programs import (
    Prediction,
    Program,
    mapped_executor,
    run,
)


def scalar_program(document: Document) -> Program[int]:
    (result,) = yield (document,)
    return result.predictions[0].token_id


def fake_predict(document: Document) -> tuple[Prediction, ...]:
    return tuple(Prediction(token_id=38, logprob=-0.2) for _ in document.query_positions)


result = run(
    scalar_program(inference_document),
    mapped_executor(fake_predict),
)
assert result.value == 38
```

The outer tuple follows document occurrence order. `result.predictions` follows
the document's `query_positions`.

`run_many` advances several generators together. It collects their ready
documents, calls the executor once, and returns each generator only its own
result slice. A generator's later yield cannot cross its preceding barrier.

## 3. Split and overlap a field

`predict_windows` is a subprogram that constructs the window documents and
joins their positional results:

```python
from experiments.probabilistic_dataflow.mock_windowed import continue_forecast, predict_windows

values = yield from predict_windows(
    ((10, 11), (20, 21)),
    ((0, 1), (1, 2)),
)
```

Both documents query coordinate `1`. Each query position carries that
model-visible coordinate. The subprogram keeps the two predictions separate
until it selects the higher-log-probability candidate. No logical output key
crosses the executor boundary.

Compose a second subprogram when later predictions depend on the joined values:

```python
continuation = yield from continue_forecast(
    values,
    query_indices=(3,),
)
values.update(continuation)
```

The executable example is
[`mock_windowed.py`](mock_windowed.py). `predict_windows` owns only the split and
join. `continue_forecast` builds a new document from the materialized values,
so the parent generator makes the dependency between waves explicit.

## 4. Refine selected coordinates

`refine` is an adaptive subprogram. Its arguments specify the observed field,
output shape, sharding, confidence threshold, and round limit:

```python
from experiments.probabilistic_dataflow.mock_refinement import refine

result = yield from refine(
    observed_token_ids=(5, 6),
    num_outputs=4,
    outputs_per_document=2,
    minimum_logprob=-0.5,
    max_refinement_rounds=3,
)
```

`refine` yields the initial proposal documents, stores their positional
predictions in dictionaries, and yields replacement documents until every
coordinate is confident or the round limit is reached.

At a top-level execution boundary, pass the same generator to `run`:

```python
completed = run(
    refine(
        observed_token_ids=(5, 6),
        num_outputs=4,
        outputs_per_document=2,
        minimum_logprob=-0.5,
        max_refinement_rounds=3,
    ),
    executor,
)
final_token_ids = completed.value.token_ids
```

Training labels are an optional subprogram argument:

```python
labeled_result = yield from refine(
    observed_token_ids=(5, 6),
    num_outputs=4,
    outputs_per_document=2,
    minimum_logprob=-0.5,
    max_refinement_rounds=3,
    target_token_ids=(100, 101, 102, 103),
)
```

Labels stay on query tokens and never become refinement context.

`run` accepts sampled results by default. Deliberately corrupted proposals use:

```python
from experiments.probabilistic_dataflow.programs import GENERATED_ORIGINS

result = run(program, executor, accepted_origins=GENERATED_ORIGINS)
```

## 5. Compose adaptive programs

Sequential composition uses Python's `yield from`:

```python
refine_geometry = yield from planning_program()
accepted = yield from verification_program(
    geometry_token,
    chemistry_token,
    attempt=0,
)
```

`parallel` advances adaptive children together:

```python
from experiments.probabilistic_dataflow.programs import parallel

geometry_token, chemistry_token = yield from parallel(
    (
        geometry_program(refine_geometry, resources),
        chemistry_program(resources),
    )
)
```

Geometry may yield a coarse document and a refinement document. Chemistry may
finish after one document. `parallel` exposes both initial documents together,
returns each positional result slice to its child, and then exposes only the
geometry refinement. The parent resumes after both children finish.

[`mock_composition.py`](mock_composition.py) includes planning, unequal child
workloads, verification, conditional retry, and cleanup of suspended children
after executor failure.

## 6. Pack documents for training and inference

`pack` converts documents with one attention layout into dense arrays:

```python
from experiments.probabilistic_dataflow.documents import QUERY, TARGET_IDS, pack

batch = pack((training_document,), max_seq_len=8)

assert batch.token_ids.shape == (1, 8)
assert batch[TARGET_IDS][0, 1] == TARGET_ID
assert batch[QUERY][0, 1]
```

`PackedBatch` carries token IDs, aligned coordinates, segments, and document
indices. `packed_executor` samples positions marked by `QUERY` and reconstructs
one `Result` per document occurrence.

Causal text uses the same representation. The token at position `i` is a query
for token `i + 1`:

```python
from experiments.probabilistic_dataflow.documents import causal_training_document

text_document = causal_training_document((5, 6, 7, 8))

assert text_document.attention == AttentionLayout.CAUSAL
assert tuple(text_document.token_ids) == (5, 6, 7, 8)
assert tuple(text_document[TARGET_IDS]) == (6, 7, 8, -1)
```

[`training.py`](training.py) packs causal text and full-attention advection
documents for one Grug model. The coordinates and attention layout change;
the token embedding table, transformer parameters, output projection, and loss
remain shared.
