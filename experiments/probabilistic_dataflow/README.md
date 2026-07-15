# Probabilistic scientific document programs

This experiment provides a small library for transformer documents and adaptive
model-call sequences. Its public concepts are:

- `Token` and `Document` for model inputs, query positions, and training labels;
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

A `Document` is one attention domain. Each `Token` carries its input token ID,
rotary position, categorical features, and optional query metadata:

```python
query = Token(
    QUERY_ID,
    features=(("scientific_position", 7),),
    query=True,
    target_id=target_id,  # omit during inference
)
document = Document(
    "advection/window-1",
    (*context_tokens, query),
    AttentionLayout.FULL,
)
```

`target_id` and `target_weight` are training metadata. Targets are absent from
`Document.token_ids`, so full-attention documents cannot read their labels.
Features such as field, coordinate, and task identity are model inputs. They do
not live in routing metadata.

Document names are descriptive and may repeat. Result routing uses document
occurrence order.

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

Domain code can retain the metadata needed to interpret positional results in a
small dataclass. The windowed example returns documents and their coordinate
tuples together:

```python
split = split_windows(example_id, contexts, output_indices)
results = yield split.documents
values = split.join_results(results)
```

`WindowSplit` owns the window coordinate tuples and overlap selection. An
adaptive `Refinement` owns a different set of domain fields and exposes its
multi-turn program through `documents()`:

```python
refinement = refine(...)
refined = yield from refinement.documents()
```

Neither dataclass is part of the execution library. They are concrete domain
values shaped for their operations.

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

`PackedBatch.query_mask` marks sample positions. `document_indices` maps packed
positions back to document occurrences. Application-level keys are not part of
packing.

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
  returns a `WindowSplit`, selects by log-probability, and feeds predictions
  into a continuation.
- [`mock_refinement.py`](mock_refinement.py) shards a field proposal and
  repeatedly replaces low-confidence coordinates through
  `Refinement.documents()`. Optional targets add labels without making them
  inference feedback.
- [`mock_composition.py`](mock_composition.py) uses sequential and parallel
  subprograms for planning, unequal specialist workloads, verification, retry,
  and cleanup after failure.

## Shared Grug training smoke

[`training.py`](training.py) constructs causal text and full-attention advection
documents from the same `Token`, `Document`, and `pack` functions. Both tasks
use one Grug parameter set, token embedding table, transformer stack, and output
projection.

| Task | Position signal | Attention | Target alignment |
| --- | --- | --- | --- |
| Synthetic text | rotary index `0..S-1` | causal | next token |
| Synthetic advection | scientific feature; rotary position `0` | full per segment | same token position |

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
