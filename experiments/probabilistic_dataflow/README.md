# Probabilistic scientific document programs

This experiment provides a small library for transformer documents and adaptive
model-call sequences. Its public concepts are:

- `Coordinate` and `Document` for aligned model inputs, query positions, and
  training labels;
- Python generators that yield tuples of documents;
- `Prediction` and `Result` values returned in document and query order;
- executor adapters that batch compatible documents.

Domain identity and mutable inference state stay in the generator. The library
does not define logical output keys, refinement operators, state-update modes,
graph nodes, or scientific value types.

[`TUTORIAL.md`](TUTORIAL.md) builds a scalar prediction, overlapping context
windows, adaptive refinement, parallel subprograms, and shared
text-and-science training with this API.

## Documents

A `Document` is one attention domain with a single implicit token axis.
`token_ids` is its content. `Coordinate` instances define typed arrays aligned
with that content:

```python
from experiments.probabilistic_dataflow.documents import QUERY, TARGET_IDS, AttentionLayout, Coordinate, Document

SCIENTIFIC_POSITION = Coordinate("scientific_position")

document = Document(
    (*context_token_ids, QUERY_ID),
    {
        SCIENTIFIC_POSITION: (*context_positions, 7),
        QUERY: (*((False,) * len(context_token_ids)), True),
        TARGET_IDS: (*((TARGET_IDS.missing,) * len(context_token_ids)), target_id),
    },
    attention=AttentionLayout.FULL,
)
```

`POSITION_IDS`, `QUERY`, `TARGET_IDS`, and `TARGET_WEIGHTS` are reserved
coordinates interpreted by the runtime. Domains define coordinates such as
field, scientific position, or task identity. Coordinate instances are keys;
two instances with the same display name remain distinct. Use
`document[SCIENTIFIC_POSITION]` for unambiguous access. `document.scientific_position`
is available when exactly one attached coordinate has that name.

Missing coordinates are padded with the value declared by their `Coordinate`.
`concatenate` joins documents along the implicit axis. It concatenates values
for shared `Coordinate` instances and fills a coordinate missing from one input;
`left + right` is its two-document shorthand.
`document.take(indices)` selects or reorders positions. Targets remain absent
from `token_ids`, so full-attention documents cannot read their labels.

## Programs

A `Program[T]` is a normal two-way generator:

```python
def forecast_program(document: Document) -> Program[int]:
    (result,) = yield (document,)
    return result.predictions[0].token_id
```

One `yield` is one barrier. Every yielded document may execute in parallel, and
the generator resumes with one `Result` per document. Each result contains one
`Prediction` per query token, in query-token order.

Domain subprograms retain the metadata needed to interpret positional results.
The caller only composes them:

```python
windowed = yield from predict_windows(
    contexts,
    output_indices,
)
continuation = yield from continue_forecast(
    windowed,
    query_indices=(4, 5),
)
windowed.update(continuation)
refined = yield from refine(
    observed_token_ids=tuple(windowed[index] for index in sorted(windowed)),
    num_outputs=4,
    outputs_per_document=2,
    minimum_logprob=-0.5,
    max_refinement_rounds=3,
)
```

`predict_windows` owns one split/join wave. `continue_forecast` owns one
dependent wave built from materialized predictions. `refine` owns its proposal
and adaptive replacement waves. Their document builders and positional joins
are private domain functions.

`run_many` mixes ready documents from independent programs without crossing a
yield barrier. Sequential subprograms use `yield from`. `parallel` advances
adaptive child generators together and slices the positional results back to
each child.

## Executors and provenance

`mapped_executor` adapts a per-document prediction function for local examples
and tests. `packed_executor`:

1. groups ready documents by `AttentionLayout`;
2. packs each group into a `PackedBatch`;
3. invokes a batch sampling function;
4. reconstructs one positional `Result` per document.

`PackedBatch[QUERY]` marks sample positions. `document_indices` maps packed
positions back to document occurrences. Packing preserves user-defined
`Coordinate` instances without interpreting them.

Each result has an `Origin`: sampled, corrupted, or supervised. `run` accepts
sampled results by default. Pass `accepted_origins=GENERATED_ORIGINS` for a
program that deliberately refines corrupted proposals. A program with
phase-specific provenance rules can inspect `Result.origin` itself.

`Run.exchanges` records each yielded document tuple and returned result tuple.
`replay` drives a fresh generator from that transcript and reports the first
changed document wave. Generator frames and transcripts are not durable
checkpoints.

## Mock domains

- [`mock_windowed.py`](mock_windowed.py) combines overlapping context windows,
  selects by log-probability, and exposes a separate continuation subprogram.
- [`mock_refinement.py`](mock_refinement.py) shards a field proposal and
  repeatedly replaces low-confidence coordinates. Optional targets add labels
  without making them inference feedback.
- [`mock_composition.py`](mock_composition.py) uses sequential and parallel
  subprograms for planning, unequal specialist workloads, verification, retry,
  and cleanup after failure.

## Shared Grug training smoke

[`training.py`](training.py) constructs causal text and full-attention advection
documents from the same `Coordinate`, `Document`, and `pack` functions. Both
tasks use one Grug parameter set, token embedding table, transformer stack, and
output projection.

| Task | Position signal | Attention | Target alignment |
| --- | --- | --- | --- |
| Synthetic text | rotary index `0..S-1` | causal | next token |
| Synthetic advection | scientific coordinate; rotary position `0` | full per segment | same token position |

The current 80-step CPU smoke reduces combined training loss from `4.1909` to
`0.2022`. This is a compatibility and memorization result, not a held-out
scientific evaluation.

## Run

Run the document and generator behavior tests:

```bash
uv run pytest -q \
  tests/experiment/test_document_programs.py \
  tests/experiment/test_mock_refinement_program.py \
  tests/experiment/test_mock_windowed_program.py \
  tests/experiment/test_mock_composition_program.py \
  tests/experiment/test_probabilistic_dataflow.py -m 'not slow'
```

Run the two-layer Grug training smoke:

```bash
uv run pytest -q tests/experiment/test_probabilistic_dataflow.py -m slow
```

## Limits

- Scientific and text values use small synthetic vocabularies.
- The scientific position adapter learns one embedding per token identity;
  compositional axis and topology encoders are not implemented.
- Calls with different attention layouts use separate dense batches.
- Refinement uses hand-written confidence rules. Learned stopping policies and
  refinement-quality experiments are untested.
- KV-cache execution, production sampling, datasets, simulators, durable
  checkpoints, and external effects are out of scope.
