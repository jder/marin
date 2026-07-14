# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.base.model import GrugModelConfig
from experiments.probabilistic_dataflow.documents import (
    AttentionLayout,
    Document,
    FeatureId,
    Output,
    OutputSlot,
    PackedDocuments,
    Record,
    Supervision,
    causal_training_document,
    pack_documents,
)
from experiments.probabilistic_dataflow.scientific_model import CrossDomainTransformer

SCIENTIFIC_POSITION_CHANNEL = "scientific_position"
ADVECTION_CELLS = 4
ADVECTION_STEPS = 3
ADVECTION_CONTEXT_RECORDS = ADVECTION_CELLS + ADVECTION_CELLS * ADVECTION_STEPS
ADVECTION_OUTPUT_RECORDS = ADVECTION_CELLS * ADVECTION_STEPS
ADVECTION_RECORDS = ADVECTION_CONTEXT_RECORDS + ADVECTION_OUTPUT_RECORDS

TEXT_SENTENCES = (
    ("bos", "the", "ocean", "field", "changes", "slowly", "eos"),
    ("bos", "the", "protein", "contact", "changes", "slowly", "eos"),
    ("bos", "the", "ocean", "contact", "changes", "today", "eos"),
    ("bos", "the", "protein", "field", "changes", "today", "eos"),
)


@dataclass
class SyntheticTokenCodec:
    """Small shared vocabulary for the direct document training smoke."""

    _tokens: dict[str, int] = field(default_factory=dict)

    PAD_ID: ClassVar[int] = 0
    QUERY_ID: ClassVar[int] = 1
    TOKEN_OFFSET: ClassVar[int] = 2
    DATA_OFFSET: ClassVar[int] = 32
    DATA_BINS: ClassVar[int] = 32

    @property
    def vocab_size(self) -> int:
        return self.DATA_OFFSET + self.DATA_BINS

    def token(self, name: str) -> int:
        if name not in self._tokens:
            token_id = self.TOKEN_OFFSET + len(self._tokens)
            if token_id >= self.DATA_OFFSET:
                raise ValueError("Synthetic text vocabulary overlaps scientific value tokens")
            self._tokens[name] = token_id
        return self._tokens[name]

    def data(self, value: int) -> int:
        if value < 0 or value >= self.DATA_BINS:
            raise ValueError(f"Synthetic value {value} is outside [0, {self.DATA_BINS})")
        return self.DATA_OFFSET + value


@dataclass(frozen=True)
class TaskBatch:
    name: str
    documents: PackedDocuments

    @property
    def token_ids(self) -> np.ndarray:
        return self.documents.token_ids

    @property
    def scientific_position_ids(self) -> np.ndarray:
        return self.documents.feature_ids(SCIENTIFIC_POSITION_CHANNEL)

    @property
    def rotary_position_ids(self) -> np.ndarray:
        return self.documents.rotary_position_ids

    @property
    def target_ids(self) -> np.ndarray:
        return self.documents.target_ids

    @property
    def loss_weights(self) -> np.ndarray:
        return self.documents.loss_weights

    @property
    def segment_ids(self) -> np.ndarray:
        return self.documents.segment_ids

    @property
    def attention_layout(self) -> AttentionLayout:
        return self.documents.attention_layout


@dataclass(frozen=True)
class TaskTrainingMetrics:
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    supervised_tokens: int


@dataclass(frozen=True)
class CrossDomainTrainingResult:
    initial_loss: float
    final_loss: float
    text: TaskTrainingMetrics
    science: TaskTrainingMetrics
    task_families: tuple[str, ...]
    shared_vocab_size: int


def synthetic_advection_document(codec: SyntheticTokenCodec, *, seed: int) -> Document:
    """Build one labeled advection call directly from document primitives."""
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 16, size=ADVECTION_CELLS, dtype=np.int32)
    forcing = rng.integers(0, 4, size=(ADVECTION_STEPS, ADVECTION_CELLS), dtype=np.int32)
    future_steps = []
    current = initial
    for step_forcing in forcing:
        current = (np.roll(current, 1) + step_forcing) % 16
        future_steps.append(current)
    future = np.stack(future_steps)

    example_id = f"advection-{seed}"
    records = []
    position = 0
    for value in (*initial, *forcing.flat):
        records.append(
            Record(
                codec.data(int(value)),
                position_id=0,
                features=(FeatureId(SCIENTIFIC_POSITION_CHANNEL, position),),
            )
        )
        position += 1
    for index, value in enumerate(future.flat):
        records.append(
            Record(
                codec.QUERY_ID,
                position_id=0,
                features=(FeatureId(SCIENTIFIC_POSITION_CHANNEL, position),),
                output=Output(
                    OutputSlot(example_id, "future", index),
                    Supervision(codec.data(int(value))),
                ),
            )
        )
        position += 1
    assert position == ADVECTION_RECORDS
    return Document(example_id, tuple(records), AttentionLayout.FULL)


def record_order_equivariance_error(*, seed: int = 0) -> float:
    """Measure the maximum logit change after permuting and restoring scientific records."""
    codec = SyntheticTokenCodec()
    document = synthetic_advection_document(codec, seed=seed)
    order = tuple(int(index) for index in np.random.default_rng(seed).permutation(len(document.records)))
    permuted = document.reordered(order)
    config = GrugModelConfig(
        vocab_size=codec.vocab_size,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=len(document.records),
    )
    with jax.set_mesh(compact_grug_mesh()):
        model = CrossDomainTransformer.init(
            config,
            scientific_position_count=ADVECTION_RECORDS,
            key=jax.random.PRNGKey(seed),
        )
        segment_ids = jnp.zeros((1, len(document.records)), dtype=jnp.int32)
        mask = AttentionMask().with_segment_ids(segment_ids)
        logits = model.logits(
            jnp.asarray((document.token_ids,)),
            jnp.asarray((document.feature_ids(SCIENTIFIC_POSITION_CHANNEL),)),
            mask=mask,
            rotary_position_ids=jnp.asarray((document.rotary_position_ids,)),
        )
        permuted_logits = model.logits(
            jnp.asarray((permuted.token_ids,)),
            jnp.asarray((permuted.feature_ids(SCIENTIFIC_POSITION_CHANNEL),)),
            mask=mask,
            rotary_position_ids=jnp.asarray((permuted.rotary_position_ids,)),
        )
    inverse = np.argsort(np.asarray(order))
    restored_logits = np.asarray(permuted_logits)[:, inverse]
    return float(np.max(np.abs(np.asarray(logits) - restored_logits)))


def build_synthetic_text_batch(codec: SyntheticTokenCodec, *, repetitions: int = 4) -> TaskBatch:
    """Build a small causal next-token workload in the shared token vocabulary."""
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}")
    sentences = [sentence for _ in range(repetitions) for sentence in TEXT_SENTENCES]
    documents = tuple(
        causal_training_document(
            f"text-{index}",
            tuple(codec.token(word) for word in sentence),
            sequence_name="text",
        )
        for index, sentence in enumerate(sentences)
    )
    return TaskBatch("synthetic_text", pack_documents(documents, max_seq_len=len(TEXT_SENTENCES[0])))


def build_synthetic_advection_batch(
    codec: SyntheticTokenCodec,
    *,
    examples: int = 8,
    max_seq_len: int = 64,
) -> TaskBatch:
    """Build a batch of full-attention scientific documents."""
    if examples <= 0:
        raise ValueError(f"examples must be positive, got {examples}")
    documents = tuple(synthetic_advection_document(codec, seed=seed) for seed in range(examples))
    return TaskBatch("synthetic_advection", pack_documents(documents, max_seq_len=max_seq_len))


def train_cross_domain_smoke(
    *,
    steps: int = 100,
    examples_per_task: int = 8,
    max_seq_len: int = 64,
    seed: int = 0,
) -> CrossDomainTrainingResult:
    """Train one Grug parameter set on causal text and full-attention scientific calls."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    codec = SyntheticTokenCodec()
    text_batch = build_synthetic_text_batch(codec, repetitions=max(1, examples_per_task // len(TEXT_SENTENCES)))
    science_batch = build_synthetic_advection_batch(
        codec,
        examples=examples_per_task,
        max_seq_len=max_seq_len,
    )
    model_config = GrugModelConfig(
        vocab_size=codec.vocab_size,
        hidden_dim=48,
        intermediate_dim=96,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=max_seq_len,
    )
    optimizer = optax.adam(learning_rate=3e-3)
    text_arrays = _task_arrays(text_batch)
    science_arrays = _task_arrays(science_batch)
    text_mask = _task_attention_mask(text_batch, text_arrays[-1])
    science_mask = _task_attention_mask(science_batch, science_arrays[-1])

    with jax.set_mesh(compact_grug_mesh()):
        model = CrossDomainTransformer.init(
            model_config,
            scientific_position_count=ADVECTION_RECORDS,
            key=jax.random.PRNGKey(seed),
        )
        opt_state = optimizer.init(model)
        initial_text = _aligned_metrics(model, *text_arrays[:-1], mask=text_mask)
        initial_science = _aligned_metrics(model, *science_arrays[:-1], mask=science_mask)

        @eqx.filter_jit
        def train_step(current_model: CrossDomainTransformer, current_opt_state: optax.OptState):
            def loss_fn(candidate: CrossDomainTransformer):
                text_loss = _task_loss(candidate, text_arrays, text_mask)
                science_loss = _task_loss(candidate, science_arrays, science_mask)
                return 0.5 * (text_loss + science_loss)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
            next_model = eqx.apply_updates(current_model, updates)
            return next_model, next_opt_state, loss

        for _ in range(steps):
            model, opt_state, _ = train_step(model, opt_state)

        final_text = _aligned_metrics(model, *text_arrays[:-1], mask=text_mask)
        final_science = _aligned_metrics(model, *science_arrays[:-1], mask=science_mask)

    text_metrics = TaskTrainingMetrics(
        initial_loss=float(initial_text[0]),
        final_loss=float(final_text[0]),
        initial_accuracy=float(initial_text[1]),
        final_accuracy=float(final_text[1]),
        supervised_tokens=int(np.sum(text_batch.loss_weights)),
    )
    science_metrics = TaskTrainingMetrics(
        initial_loss=float(initial_science[0]),
        final_loss=float(final_science[0]),
        initial_accuracy=float(initial_science[1]),
        final_accuracy=float(final_science[1]),
        supervised_tokens=int(np.sum(science_batch.loss_weights)),
    )
    return CrossDomainTrainingResult(
        initial_loss=0.5 * (text_metrics.initial_loss + science_metrics.initial_loss),
        final_loss=0.5 * (text_metrics.final_loss + science_metrics.final_loss),
        text=text_metrics,
        science=science_metrics,
        task_families=(text_batch.name, science_batch.name),
        shared_vocab_size=codec.vocab_size,
    )


def _aligned_metrics(
    model: CrossDomainTransformer,
    token_ids: jax.Array,
    scientific_position_ids: jax.Array,
    rotary_position_ids: jax.Array,
    target_ids: jax.Array,
    loss_weights: jax.Array,
    *,
    mask: AttentionMask,
) -> tuple[jax.Array, jax.Array]:
    loss = model.aligned_token_loss(
        token_ids,
        scientific_position_ids,
        target_ids,
        loss_weights,
        mask=mask,
        rotary_position_ids=rotary_position_ids,
        reduction="mean",
    )
    logits = model.logits(
        token_ids,
        scientific_position_ids,
        mask=mask,
        rotary_position_ids=rotary_position_ids,
    )
    predictions = jnp.argmax(logits, axis=-1)
    supervised = loss_weights > 0
    correct = jnp.sum((predictions == target_ids) * supervised)
    accuracy = correct / jnp.maximum(jnp.sum(supervised), 1)
    return loss, accuracy


def _task_arrays(batch: TaskBatch) -> tuple[jax.Array, ...]:
    return (
        jnp.asarray(batch.token_ids),
        jnp.asarray(batch.scientific_position_ids),
        jnp.asarray(batch.rotary_position_ids),
        jnp.asarray(batch.target_ids),
        jnp.asarray(batch.loss_weights),
        jnp.asarray(batch.segment_ids),
    )


def _task_attention_mask(batch: TaskBatch, segment_ids: jax.Array) -> AttentionMask:
    if batch.attention_layout == AttentionLayout.CAUSAL:
        return AttentionMask.causal().with_segment_ids(segment_ids)
    return AttentionMask().with_segment_ids(segment_ids)


def _task_loss(
    model: CrossDomainTransformer,
    arrays: tuple[jax.Array, ...],
    mask: AttentionMask,
) -> jax.Array:
    token_ids, scientific_position_ids, rotary_position_ids, target_ids, loss_weights, _segment_ids = arrays
    return model.aligned_token_loss(
        token_ids,
        scientific_position_ids,
        target_ids,
        loss_weights,
        mask=mask,
        rotary_position_ids=rotary_position_ids,
        reduction="mean",
    )
