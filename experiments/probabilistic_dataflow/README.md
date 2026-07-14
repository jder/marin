# Probabilistic scientific document programs

This experiment tests a small library for expressing transformer inputs and
adaptive model-call sequences. The public concepts are:

- `Record`, `Document`, and `OutputSlot` for model inputs and logical outputs;
- Python generators that yield barriered `DocumentRequest` waves;
- executor adapters that batch documents and route `DocumentResult` values back
  to the suspended generators.

Domain control flow stays in Python. The driver does not contain refinement
operators, graph nodes, strategy registries, or scientific value types.

Start with [`TUTORIAL.md`](TUTORIAL.md) for an executable path from a two-record
scalar prediction through overlapping context windows, adaptive refinement,
subprogram composition, and shared text-and-science training.

## Documents

An immutable `Document` is one attention domain. Each `Record` carries an input
token, a rotary position, categorical feature IDs, and an optional `Output`:

```python
slot = OutputSlot("advection-0", "future", index=7)
query = Record(
    input_id=QUERY_ID,
    position_id=0,
    features=(FeatureId("scientific_position", 7),),
    output=Output(slot, Supervision(target_id)),
)
document = Document("advection-0/window-1", records, AttentionLayout.FULL)
```

`OutputSlot` identifies the requested value independently of the document that
predicts it. Several documents may request disjoint subsets of one field or
produce competing observations for the same slot. Document IDs are descriptive
and may repeat; result routing uses document occurrence order.

`Supervision` is label metadata. The target token is absent from
`Document.token_ids`. Inference uses the same query record with
`Output(slot)` and no label.

`pack_documents` creates dense token, feature, rotary-position, target, loss,
and segment arrays while retaining each output's logical slot and physical
location. Documents in one packed batch share an `AttentionLayout`.

## Programs

A `DocumentProgram[T]` is a normal Python generator:

```python
def forecast_program(document):
    response = yield DocumentRequest(
        "forecast/initial",
        (document,),
        SAMPLED_FEEDBACK,
    )
    observations = disjoint_prediction_observations(response)
    return PredictionState().updated(
        prediction_values(observations),
        mode=PredictionUpdateMode.REQUIRE_EMPTY,
    )
```

Each yielded request is one barrier. Its documents may execute in parallel, and
the generator resumes after every result is available. `run_programs` mixes the
ready requests from independent programs. Sequential subprograms use
`yield from`; `parallel_programs` exposes ready waves from adaptive children
together.

Responses retain individual `PredictionObservation` values until the program
chooses a combination policy. `disjoint_prediction_observations` rejects
overlap. `highest_logprob_observations` selects one candidate per slot.
`PredictionState` then commits selected tokens with explicit insert or replace
semantics.

Each `DocumentResult` records whether its observations are sampled, supervised,
or corrupted. A request declares which origins it accepts, so labels cannot
silently become inference feedback.

The driver records yield-boundary exchanges for replay. A fresh generator can
consume the transcript and fail at the first changed request. The live generator
frame and transcript are not durable checkpoints.

## Executors

`mapped_executor` adapts a per-document prediction function for local examples
and tests. `packed_executor`:

1. collects documents from every ready request;
2. groups them by attention layout;
3. packs each group;
4. invokes a batch prediction function;
5. reconstructs results by request and document occurrence.

Packing carries an explicit document index, so repeated document IDs and
overlapping output slots do not affect routing.

## Mock domains

- [`mock_refinement.py`](mock_refinement.py) performs adaptive partial
  refinement. Inference construction requires only output shape; a separate
  constructor attaches training labels.
- [`mock_windowed.py`](mock_windowed.py) combines overlapping context windows
  and feeds selected predictions into a continuation document.
- [`mock_composition.py`](mock_composition.py) uses sequential and parallel
  subprograms for planning, unequal specialist workloads, verification, retry,
  and cleanup on failure.

## Shared Grug training smoke

[`training.py`](training.py) constructs causal text and full-attention advection
documents directly from the document primitives. Both tasks use one dense Grug
parameter set, token embedding table, transformer stack, and output projection.

| Task | Position signal | Attention | Target alignment |
| --- | --- | --- | --- |
| Synthetic text | rotary index `0..S-1` | causal | next token |
| Synthetic advection | scientific feature; rotary position `0` | full per segment | same record |

The 80-step CPU smoke reduced combined training loss from `4.1909` to `0.2022`.
This is a compatibility and memorization result. It does not measure held-out
scientific prediction, language quality, cross-task transfer, or refinement
quality.

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

## Deliberate limits

- Scientific and text values use small synthetic vocabularies.
- The scientific position adapter learns one embedding per record identity;
  compositional axis and topology encoders are not implemented.
- Calls with different attention layouts use separate dense batches.
- The refinement examples use hand-written confidence rules. Learned stopping
  policies and refinement-quality experiments are untested.
- Transcripts retain requests and responses in memory and are not a durable
  checkpoint format.
- KV-cache execution, production sampling, datasets, simulators, and external
  effects are out of scope.
