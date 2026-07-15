# Tutorial: direct document generators

## Orientation

A document program has one model-facing data structure and one control-flow
rule:

- a `Document` contains `Token` values for one isolated attention domain;
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
    AttentionLayout,
    Document,
    Token,
)

QUERY_ID = 1
CURRENT_ID = 35
TARGET_ID = 37

current = Token(
    CURRENT_ID,
    features=(("scientific_position", 0),),
)
inference_query = Token(
    QUERY_ID,
    features=(("scientific_position", 1),),
    query=True,
)
training_query = Token(
    QUERY_ID,
    features=(("scientific_position", 1),),
    query=True,
    target_id=TARGET_ID,
)

inference_document = Document(
    "scalar/inference",
    (current, inference_query),
    AttentionLayout.FULL,
)
training_document = Document(
    "scalar/training",
    (current, training_query),
    AttentionLayout.FULL,
)
```

Inference and training have identical model inputs:

```python
assert inference_document.token_ids == training_document.token_ids == (35, 1)
assert inference_document.target_ids == (-1, -1)
assert training_document.target_ids == (-1, 37)
```

`target_id` is metadata for the loss. It is absent from `token_ids`, so the
full-attention query cannot read token `37`. Both tokens use rotary position
`0`; the `scientific_position` feature tells the model which field position each
token represents.

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
the document's `query_positions`. Document names are descriptions and may
repeat.

`run_many` advances several generators together. It collects their ready
documents, calls the executor once, and returns each generator only its own
result slice. A generator's later yield cannot cross its preceding barrier.

## 3. Split and overlap a field

`split_windows` returns a domain dataclass containing the documents and the
coordinates needed to join their positional results:

```python
from experiments.probabilistic_dataflow.mock_windowed import (
    split_windows,
    windowed_forecast_program,
)

split = split_windows(
    "forecast-0",
    context_token_ids=((10, 11), (20, 21)),
    output_indices=((0, 1), (1, 2)),
)
```

Both documents query coordinate `1`. Each query token carries that coordinate
as a model-visible feature. `WindowSplit.join_results` keeps the two predictions
separate until it selects the higher-log-probability candidate:

```python
results = yield split.documents
values = split.join_results(results)
```

No logical output key crosses the executor boundary. The generator already has
the `WindowSplit` needed to assemble the field. It can feed `values` into later
documents as ordinary context tokens. The complete program is:

```python
program = windowed_forecast_program(
    split,
    continuation_output_indices=(3,),
)
```

The executable example is
[`mock_windowed.py`](mock_windowed.py). It also builds a continuation document
whose query coordinates follow the materialized initial predictions.

## 4. Refine selected coordinates

`refine` returns a `Refinement` dataclass containing the observed field, output
shape, sharding parameters, confidence threshold, and round limit:

```python
from experiments.probabilistic_dataflow.mock_refinement import refine

refinement = refine(
    example_id="field-0",
    observed_token_ids=(5, 6),
    num_outputs=4,
    outputs_per_document=2,
    minimum_logprob=-0.5,
    max_refinement_rounds=3,
)
```

`Refinement.documents()` is a `Program[RefinementResult]`. It yields the initial
proposal documents, stores their positional predictions in dictionaries, and
yields replacement documents until every coordinate is confident or the round
limit is reached:

```python
result = yield from refinement.documents()
assert result.refinement_rounds <= 3
```

At a top-level execution boundary, pass the same generator to `run`:

```python
completed = run(refinement.documents(), executor)
final_token_ids = completed.value.token_ids
```

Training labels are another field on the domain value:

```python
labeled_refinement = refine(
    example_id="field-0",
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
refine_geometry = yield from planning_program(example_id)
accepted = yield from verification_program(
    example_id,
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
        geometry_program(example_id, refine_geometry, resources),
        chemistry_program(example_id, resources),
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
from experiments.probabilistic_dataflow.documents import pack

batch = pack((training_document,), max_seq_len=8)

assert batch.token_ids.shape == (1, 8)
assert batch.target_ids[0, 1] == TARGET_ID
assert batch.query_mask[0, 1]
```

`PackedBatch` carries token IDs, feature arrays, rotary positions, targets,
weights, segments, document indices, and a query mask. `packed_executor` samples
the query-mask positions and reconstructs one `Result` per document occurrence.

Causal text uses the same representation. The token at position `i` is a query
for token `i + 1`:

```python
from experiments.probabilistic_dataflow.documents import causal_training_document

text_document = causal_training_document("text-0", (5, 6, 7, 8))

assert text_document.attention_layout == AttentionLayout.CAUSAL
assert text_document.token_ids == (5, 6, 7, 8)
assert text_document.target_ids == (6, 7, 8, -1)
```

[`training.py`](training.py) packs causal text and full-attention advection
documents for one Grug model. The position features and attention layout change;
the token embedding table, transformer parameters, output projection, and loss
remain shared.
