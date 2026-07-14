# Tutorial: from one scalar to a shared text-and-science model

## Orientation

A language model normally treats each input position as a token in a sentence.
This prototype can instead treat one position as a scientific value such as
`future[time=1, cell=0.5]`. We call that position a **record**.

The transformer is still an ordinary dense model. The DSL describes which
scientific records to construct, which records may attend to each other, and
which records have training labels.

The examples build that idea in five steps:

1. predict one scalar from another;
2. read the compiler's debug output;
3. extend the same program to an indexed advection field;
4. request lower-level control over factorization and refinement;
5. train one transformer on scientific records and ordinary text.

The code is an experiment, not a production Marin API. Generate the reports
used below from the repository root with:

```bash
uv run python -m experiments.probabilistic_dataflow.debug_render
```

## 1. Predict one scalar

Suppose the current measurement is the integer `3`, and a training example says
the future measurement is `5`. Both values are discrete IDs from `0` through
`15`. We want to model `p(future | current)`.

```python
from experiments.probabilistic_dataflow.dsl import (
    AttentionPattern,
    Budget,
    DocumentSpec,
    FieldType,
    InferenceProgram,
    PositionMode,
)

measurement = FieldType("measurement", bins=16)
scientific_document = DocumentSpec(
    attention=AttentionPattern.FULL,
    positions=PositionMode.SCIENTIFIC,
)

program = InferenceProgram(
    "scalar_forecast",
    budget=Budget(model_calls=1, generated_tokens=1),
)
current = program.input_value("current", measurement)
future = program.generate(
    "future",
    measurement,
    context=(current,),
    document=scientific_document,
    factor_name="scalar_transition",
)
program.finish(future)
```

`FieldType` describes the kind of value. Here `bins=16` means one token chosen
from 16 possible value tokens. `input_value` introduces a value supplied to a
model call. `generate` does two things together:

- it defines `future` as a scientific value modeled from `current`;
- it adds a transformer call that will generate `future` from that context.

`finish(future)` names the program output and checks the call and token budgets.
There is no separate query or lowering-strategy object.

The `DocumentSpec` is explicit because attention and position have semantic
consequences. This factor has no meaningful left-to-right order, so it uses full
attention and scientific identities rather than sequence positions.

## 2. Read the scalar debug dump

A concrete training example supplies realized values:

```python
from experiments.probabilistic_dataflow.compiler import TokenCodec, lower_to_transformer
from experiments.probabilistic_dataflow.synthetic import scalar_forecast_example

example = scalar_forecast_example(program)  # current=3, future=5
execution = lower_to_transformer(program, example, TokenCodec())
```

`future=5` is present in the example so it can become a cross-entropy label. It
is not fed to the model. The full rendering is
[`debug_outputs/scalar.md`](debug_outputs/scalar.md).

### Inference Program Values

The first graph shows scientific values and dependencies:

```text
%0 current : input measurement[scalar]
    |
    v
%1 future  : sample measurement[scalar], factor=scalar_transition
```

The `FlowInfo` column carries provenance, split keys, and random ancestors for
later analysis. None of those fields changes the document layout in this
example.

### Inference Plan IR

The second graph is the model-call schedule recorded by `generate`:

```text
call 0: generate future from current
attention: full_segment
positions: scientific
```

The plan is a mechanical, validated view of the staged Python program. It is
useful to compiler and runtime code, but users do not author it separately.

### Transformer Execution IR

The final section shows the exact records sent to the transformer:

| Role | Scientific identity | Model input | Training label |
| --- | --- | --- | --- |
| context | `scalar_forecast.current[scalar]` | `value:3` | none |
| target | `scalar_forecast.future[scalar]` | `<query>` | `value:5` |

At the target record the model sees `<query>` plus the embedding identifying
`future[scalar]`. It predicts logits there, and cross-entropy compares those
logits with `value:5`. The target value is only the label.

Every record in this document has rotary position `0`, so RoPE contributes no
serialization-order signal. The scientific identity embedding distinguishes
`current` from `future`. Full attention lets both records exchange information.
Printing `current` first is a packing choice, not part of the scientific model.

## 3. Extend the program to advection

The advection example predicts a field on four spatial cells for three future
times. It has four initial values, twelve forcing values, and twelve target
values.

```python
from experiments.probabilistic_dataflow.dsl import MeshAxis, OrderedAxis

cell = MeshAxis("cell", 4, coordinates=((0.0,), (0.25,), (0.5,), (0.75,)))
time = OrderedAxis("time", 3)

state = FieldType("state", (cell,), bins=16)
forcing_type = FieldType("forcing", (time, cell), bins=16)
trajectory = FieldType("state_trajectory", (time, cell), bins=16)

program = InferenceProgram(
    "synthetic_advection",
    budget=Budget(model_calls=1, generated_tokens=12),
)
initial = program.input_value("initial", state)
forcing = program.input_value("forcing", forcing_type)
future = program.generate(
    "future",
    trajectory,
    context=(initial, forcing),
    document=scientific_document,
    factor_name="advection_transition",
)
program.finish(future)
```

The DSL did not gain a spatial attention primitive. Named axes expand each
field into records with meaningful identities. For example:

```text
synthetic_advection.future[time=1,cell=(0.5,)]
```

The record counts changed mechanically:

| | Scalar | Advection |
| --- | ---: | ---: |
| Context records | 1 | 4 initial + 12 forcing |
| Target records | 1 | 12 future |
| Total records | 2 | 28 |

See [`debug_outputs/advection.md`](debug_outputs/advection.md). Its plan notes
that one twelve-token factor is approximated by twelve parallel token
marginals. All query records see the same context, but independently sampling
their logits cannot represent correlations among generated coordinates. The
compiler reports that approximation instead of silently calling it the original
joint distribution.

## 4. Drop down for control over model calls

Because the staged program is already an inference program, dropping down means
writing more calls and passing generated values between them.

### Preserve a scientific factorization

The structure example first predicts contacts from a sequence, then predicts
distances from the sequence and the generated contacts:

```python
contacts = program.generate(
    "contacts",
    contacts_type,
    context=(sequence,),
    document=scientific_document,
    factor_name="contact_map",
)
distances = program.generate(
    "distances",
    distance_type,
    context=(sequence, contacts),
    document=scientific_document,
    factor_name="distance_given_contacts",
)
program.finish(contacts, distances)
```

The second call consumes a value produced by the first, so the plan in
[`debug_outputs/structure.md`](debug_outputs/structure.md) contains `call 0 ->
call 1`. This is the factorization
`p(contacts | sequence) p(distances | sequence, contacts)`. It is not replaced
with one joint call.

### Add adaptive refinement

The static `InferenceProgram.refine` method can still describe a fixed call
plan. Runtime-dependent refinement is more naturally a Python generator. It
yields documents and receives structured model results at the same expression:

```python
def refine_field(proposal_documents):
    response = yield DocumentRequest(
        "advection/proposal",
        proposal_documents,
        SAMPLED_FEEDBACK,
    )
    observations = disjoint_prediction_observations(response)
    state = PredictionState().updated(
        prediction_values(observations),
        mode=PredictionUpdateMode.REQUIRE_EMPTY,
    )

    while selected := tuple(obs.slot for obs in observations if obs.logprob < -0.5):
        refinement_document = build_refinement_document(state, selected)
        response = yield DocumentRequest(
            "advection/refine",
            (refinement_document,),
            GENERATED_FEEDBACK,
        )
        observations = disjoint_prediction_observations(response)
        state = state.updated(
            prediction_values(observations),
            mode=PredictionUpdateMode.REPLACE,
        )
    return state
```

The loop, stopping rule, and choice of slots are ordinary Python. The reusable
runtime only knows that each `DocumentRequest` is a barriered wave. Feedback
tokens are materialized from `PredictionState`, so the next document consumes
the proposal produced by the preceding call rather than a training label.
`REPLACE` rejects unknown slots, which catches a refinement step that silently
writes a new identity instead of updating its proposal.

The proposal may be split across several documents with different context
views. Disjoint results assemble directly. Overlapping results remain separate
`PredictionObservation` values until the program explicitly selects one, for
example with `highest_logprob_observations`.

Inference construction does not require targets. A separate supervised builder
attaches labels to the same query records for training while sampled or
corrupted feedback still supplies the next-round context. See
[`mock_refinement.py`](mock_refinement.py) and
[`mock_windowed.py`](mock_windowed.py) for complete executable examples.

### Compose document programs

Sequential subprograms use Python's `yield from`. When independent adaptive
subprograms should expose their ready documents in the same model wave, use the
small `parallel_programs` combinator:

```python
plan = yield from planning_program(example_id)
geometry, chemistry = yield from parallel_programs(
    (
        geometry_program(example_id, plan),
        chemistry_program(example_id),
    ),
    request_prefix=f"{example_id}/specialists",
)
accepted = yield from verification_program(example_id, geometry, chemistry)
```

The geometry branch may yield twice while chemistry yields once. Verification
starts only after both return. The scheduler sees document waves, not specialist
types or the reason for the branch. The full example in
[`mock_composition.py`](mock_composition.py) also retries rejected geometry and
checks that suspended child resources close on failure.

### Keep document programs synchronous

`DocumentProgram` is a synchronous generator even when model execution uses
background work. This matches the interfaces Grug exposes today:

- checkpoint and Hugging Face model loaders are synchronous functions;
- the training loader consumes `AsyncDataset` values internally, then exposes a
  normal Python iterator backed by background prefetch;
- JAX dispatches device computation asynchronously, but a model call returns a
  `jax.Array` through a synchronous Python interface.

The implementations are in
[`levanter.model_loading`](../../lib/levanter/src/levanter/model_loading.py),
[`levanter.data.loader`](../../lib/levanter/src/levanter/data/loader.py), and the
explicit Grug synchronization point in
[`experiments/grug/base/train.py`](../grug/base/train.py).

The generator describes dependencies between model calls. The executor owns
waiting, batching, and device synchronization. A local JAX executor can call the
model and materialize predictions before returning `DocumentResponse`. A remote
HTTP or vLLM executor may need `await`, but that changes the driver rather than
the document program.

The current prototype provides the synchronous `DocumentExecutor` and
`run_programs`. An async backend should add a separate executor boundary of this
form:

```python
class AsyncDocumentExecutor(Protocol):
    async def __call__(
        self,
        requests: tuple[DocumentRequest, ...],
    ) -> tuple[DocumentResponse, ...]: ...
```

An `arun_programs` driver would prime and resume the same synchronous generators
but await this executor between ready waves. Making `DocumentProgram` itself an
async generator would remove two useful Python operations: async generators
cannot return the program's final value or delegate with `yield from`.

For a genuinely sequential task, choose
`DocumentSpec(attention=CAUSAL, positions=SEQUENCE)`. That uses ordinary rotary
indices and a causal mask. The choice is per call, so it does not require a
different transformer architecture.

## 5. Train one transformer on text and science

The cross-domain demo trains one Grug model on causal synthetic text and
full-attention scientific records:

```python
from experiments.probabilistic_dataflow.training import train_cross_domain_smoke

result = train_cross_domain_smoke(steps=80, examples_per_task=8, seed=0)
```

| | Synthetic text | Synthetic advection |
| --- | --- | --- |
| Input unit | word-like token | scientific record |
| Position signal | rotary index `0..S-1` | scientific identity; rotary position `0` |
| Attention | causal | full within one example |
| Label | next token | value aligned with the `<query>` record |

Both tasks use the same token embeddings, transformer blocks, and output
projection. Scientific records additionally use a scientific-identity embedding
table, which contributes zero to text inputs. The tasks are evaluated as
separate dense batches because their masks differ, then their losses are
averaged before one optimizer update.

[`debug_outputs/cross-domain.md`](debug_outputs/cross-domain.md) places the two
document types side by side. It shows shifted text labels such as:

```text
input <text:the> -> label <text:ocean>
```

and aligned scientific labels such as:

```text
input <query> at future[time=0,cell=(0.0,)] -> label value:5
```

The tiny run is only a compatibility and memorization check. It does not test
held-out scientific prediction, language quality, cross-task transfer, or a
complete refinement runtime.
