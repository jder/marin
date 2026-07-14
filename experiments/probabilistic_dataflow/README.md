# Probabilistic scientific dataflow spike

This experiment tests a small document library for building heterogeneous
transformer calls without introducing a new model stack or dataset system.
The reusable surface has three layers:

- `Record`, `Document`, and `OutputSlot` encode model inputs and logical outputs;
- Python generators yield barriered `DocumentRequest` waves and receive
  per-document prediction observations;
- executor adapters pack ready documents and route results back to the generator.

Control flow stays in Python. A program can branch, loop, split one prediction
over several context views, or feed sampled values into a later document. An
executor does not understand refinement, domain values, or call topology.

The scientist-facing `InferenceProgram` remains one optional static encoder
into the document library. It is useful when a complete plan must be inspected
before execution, but packing and interactive execution do not depend on it.

Start with [`TUTORIAL.md`](TUTORIAL.md) for a guided path from a two-record
scalar prediction through generator execution, overlapping context windows,
adaptive refinement, subprogram composition, and shared text-and-science
training.

## Document library boundary

The reusable execution boundary consists of immutable `Record`, `Document`,
`OutputSlot`, `PackedDocuments`, and `PredictionState` values. An output slot
identifies a logical value independently of the document that predicts it:

```python
slot = OutputSlot("advection-0", "future", index=7)
query = Record(
    input_id=codec.QUERY_ID,
    position_id=0,
    output=Output(slot, Supervision(codec.data(true_value))),
)
document = Document("advection-0/window-1", records, AttentionLayout.FULL)
```

Several documents can write disjoint subsets of the same logical slot set.
Each document may carry a different context view, which permits context-window
sharding without changing the identity of the requested values. Packing keeps
the output-slot locations alongside the dense arrays, and sampled values merge
into one immutable state.

Refinement writes the same slots again with an explicit replacement policy.
Feedback records are materialized from `PredictionState`, so inference code
feeds the preceding proposal back into the next document rather than silently
using a training label. Multiple hypotheses are represented by multiple states;
the document library does not choose how competing predictions are combined.

Causal text uses the same representation: the record containing token `i`
writes the logical slot for token `i + 1`. Scientific query records instead
write aligned field-value slots. Both paths use `pack_documents`.

## Interactive document programs

A document source is an ordinary two-way generator:

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

Each yielded request is one barrier: all its documents may execute in parallel,
and the generator resumes only after every document result is available.
`run_programs` mixes the ready requests from independent programs. Sequential
subprograms compose with `yield from`; `parallel_programs` is the one small
combinator needed to expose ready waves from adaptive children together.

Responses preserve document occurrences before observations are combined.
`results[i]` belongs to `request.documents[i]`; document IDs are descriptive
and may repeat. This permits overlapping context windows to predict the same
slot several times. The program must then choose a policy, such as the highest
log-probability observation, before committing values to `PredictionState`.
Each result also records whether its values were sampled, supervised, or
corrupted, and the request declares which origins it accepts.

Three mock domains exercise the boundary:

- [`mock_refinement.py`](mock_refinement.py) performs adaptive partial
  refinement with unlabeled inference documents and an explicit labeled variant;
- [`mock_windowed.py`](mock_windowed.py) assembles overlapping context windows
  and feeds the selected predictions into a continuation document;
- [`mock_composition.py`](mock_composition.py) uses sequential and parallel
  subprograms for planning, unequal specialist workloads, verification, and retry.

The driver records yield-boundary exchanges for deterministic replay. A fresh
generator can consume that transcript and fail at the first changed request;
the live generator frame is not a serialized plan or checkpoint.

## Scientist-facing surface

```python
document = DocumentSpec(
    attention=AttentionPattern.FULL,
    positions=PositionMode.SCIENTIFIC,
)
program = InferenceProgram(
    "advection",
    budget=Budget(model_calls=1, generated_tokens=12),
)

initial = program.input_value("initial", state)
forcing = program.input_value("forcing", forcing_type)
future = program.generate(
    "future",
    trajectory,
    context=(initial, forcing),
    document=document,
    factor_name="advection_transition",
)
program.finish(future)
```

The logical values contain no physical token order. Compilation creates one
record per scientific value instance:

```text
record embedding = content token embedding + scientific position embedding
```

Context records carry value tokens. Target records carry a `<query>` token and
an aligned training label; the target value is not a model input. Scientific
documents use zero rotary positions and full attention when those choices match
the scientific factor. Reordering complete records therefore only reorders the
outputs.

Sequential factorization is represented by generated values flowing between
model calls. For example:

```python
contacts = program.generate("contacts", contacts_type, context=(sequence,), ...)
distances = program.generate("distances", distance_type, context=(sequence, contacts), ...)
```

The second call depends on the first because `contacts` is generated context.
A causal mask is used only when the task itself requires sequence order.

## Shared text-and-science model

The model remains a normal dense Grug transformer with RoPE. Execution data
selects the behavior for each call:

| Task | Position signal | Attention | Target alignment |
| --- | --- | --- | --- |
| Synthetic text | rotary token index `0..S-1` | causal | next token |
| Scientific records | scientific descriptor; rotary position `0` | full within segment | same record |

Both paths use the same token embeddings, transformer blocks, and output
projection. Calls with different attention layouts are evaluated in separate
dense batches, but their losses update the same model parameters.

The scientific smoke test packs synthetic advection and contact-map examples.
A second smoke test mixes causal synthetic text with full-attention scientific
records. These are compatibility and memorization checks, not scientific
generalization or language-quality results.

## Run

Inspect compiler behavior without training:

```bash
uv run python -m experiments.probabilistic_dataflow.demo --training-steps 0
```

Run the tiny training demonstrations:

```bash
uv run python -m experiments.probabilistic_dataflow.demo --training-steps 80
```

## Debug renderings

Generate readable Markdown for the staged values, model-call plan, transformer
execution, and document layout:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render
```

The reports live in [`debug_outputs/`](debug_outputs/README.md):

- [`scalar.md`](debug_outputs/scalar.md) shows one context scalar and one target;
- [`advection.md`](debug_outputs/advection.md) shows an indexed field and refinement calls;
- [`contacts.md`](debug_outputs/contacts.md) shows unordered residue-pair targets;
- [`structure.md`](debug_outputs/structure.md) shows `sequence -> contacts -> distances`;
- [`mixed-packing.md`](debug_outputs/mixed-packing.md) shows packed segment boundaries;
- [`cross-domain.md`](debug_outputs/cross-domain.md) compares text and science calls through one model.

Verify that checked-in reports match the renderer:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render --check
```

## Implemented

- ordered, set, mesh, categorical, and unordered-pair axes;
- typed discrete fields and canonical pair coordinates;
- external inputs, deterministic map/join/select/reduce nodes, and generated values;
- provenance, split-key, and random-ancestor propagation;
- a staged model-call DAG with full or causal attention and scientific or sequence positions;
- stable logical output slots spanning multiple documents and context views;
- immutable prediction state with explicit initial-write and refinement-replacement policies;
- interactive generator programs with barriered multi-document requests;
- mixed execution of independent programs and parallel adaptive subprograms;
- per-document feedback provenance and explicit observation-selection policies;
- yield-boundary transcript replay with divergence detection;
- parallel generation and fixed-step refinement;
- factor-dependency preservation and explicit parallel-marginal approximation notes;
- inference-plan and transformer-execution IRs;
- shared value-token embeddings plus scientific position embeddings;
- aligned scientific labels, shifted text labels, and standard cross-entropy for both;
- heterogeneous scientific packing with per-document segment boundaries;
- a numerical scientific-record permutation-equivariance check;
- field RMSE and spectral-error metrics;
- tiny scientific-only and cross-domain Marin Grug training loops.

## Deliberate limits

- Values are already discretized synthetic integers, and text uses a tiny fixed vocabulary.
- An LM generating these Python inference programs is the intended workflow, but is not implemented here.
- Parallel field generation is a product-of-token-marginals approximation, recorded in the plan.
- Prediction-state assembly and replacement are implemented; model sampling is supplied through executor adapters.
- Scientific positions use one learned embedding per fully qualified coordinate; compositional axis and topology encoders are not implemented.
- Calls with different attention layouts are not packed into the same dense batch.
- The mock refinement loop has adaptive stopping, but learned stopping policies and refinement-quality experiments are out of scope.
- Transcripts currently retain requests and responses in memory and are not a durable checkpoint format.
- KV-cache execution, datasets, simulators, and external effects are out of scope.
