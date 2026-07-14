# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class AttentionLayout(StrEnum):
    FULL = "full_segment"
    CAUSAL = "causal_segment"


@dataclass(frozen=True, order=True)
class OutputSlot:
    """Stable identity for one value predicted by one or more documents."""

    example_id: str
    value_name: str
    index: int


@dataclass(frozen=True)
class FeatureId:
    """One categorical embedding feature attached to a record."""

    channel: str
    value: int


@dataclass(frozen=True)
class Supervision:
    target_id: int
    weight: float = 1.0


@dataclass(frozen=True)
class Output:
    slot: OutputSlot
    supervision: Supervision | None = None


@dataclass(frozen=True)
class Record:
    """One transformer position with input features and an optional logical output."""

    input_id: int
    position_id: int
    features: tuple[FeatureId, ...] = ()
    output: Output | None = None

    def __post_init__(self) -> None:
        channels = [feature.channel for feature in self.features]
        if len(channels) != len(set(channels)):
            raise ValueError(f"Record contains duplicate feature channels: {channels}")


@dataclass(frozen=True)
class Document:
    """An isolated attention domain containing encoded model records."""

    id: str
    records: tuple[Record, ...]
    attention_layout: AttentionLayout

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("A document requires at least one record")

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(record.input_id for record in self.records)

    @property
    def rotary_position_ids(self) -> tuple[int, ...]:
        return tuple(record.position_id for record in self.records)

    @property
    def target_ids(self) -> tuple[int, ...]:
        return tuple(
            record.output.supervision.target_id
            if record.output is not None and record.output.supervision is not None
            else -1
            for record in self.records
        )

    @property
    def loss_weights(self) -> tuple[float, ...]:
        return tuple(
            record.output.supervision.weight
            if record.output is not None and record.output.supervision is not None
            else 0.0
            for record in self.records
        )

    @property
    def output_slots(self) -> tuple[OutputSlot | None, ...]:
        return tuple(record.output.slot if record.output is not None else None for record in self.records)

    @property
    def feature_channels(self) -> tuple[str, ...]:
        return tuple(sorted({feature.channel for record in self.records for feature in record.features}))

    def feature_ids(self, channel: str) -> tuple[int, ...]:
        return tuple(
            next((feature.value for feature in record.features if feature.channel == channel), -1)
            for record in self.records
        )

    def reordered(self, order: tuple[int, ...]) -> Document:
        """Return the same records in a different physical order."""
        if tuple(sorted(order)) != tuple(range(len(self.records))):
            raise ValueError("Record order must be a permutation of all document positions")
        return Document(self.id, tuple(self.records[index] for index in order), self.attention_layout)

    def selected(self, id: str, record_indices: tuple[int, ...]) -> Document:
        """Create a context view or prediction shard over selected records."""
        if len(set(record_indices)) != len(record_indices):
            raise ValueError("A document view cannot repeat record indices")
        if any(index < 0 or index >= len(self.records) for index in record_indices):
            raise IndexError(f"Record view {record_indices} is outside document length {len(self.records)}")
        return Document(id, tuple(self.records[index] for index in record_indices), self.attention_layout)


def causal_training_document(id: str, token_ids: tuple[int, ...], *, sequence_name: str) -> Document:
    """Encode shifted next-token supervision as aligned document outputs."""
    if len(token_ids) < 2:
        raise ValueError("Causal training documents require at least two tokens")
    records = []
    for index, token_id in enumerate(token_ids):
        output = None
        if index + 1 < len(token_ids):
            slot = OutputSlot(id, sequence_name, index + 1)
            output = Output(slot, Supervision(token_ids[index + 1]))
        records.append(Record(token_id, index, output=output))
    return Document(id, tuple(records), AttentionLayout.CAUSAL)


@dataclass(frozen=True)
class PredictionValue:
    slot: OutputSlot
    token_id: int


class PredictionUpdateMode(StrEnum):
    REQUIRE_EMPTY = "require_empty"
    REPLACE = "replace"


@dataclass(frozen=True)
class PredictionState:
    """Immutable materialized values keyed by logical output slot."""

    values: tuple[PredictionValue, ...] = ()

    def value(self, slot: OutputSlot) -> int:
        for prediction in self.values:
            if prediction.slot == slot:
                return prediction.token_id
        raise KeyError(slot)

    def updated(
        self,
        predictions: tuple[PredictionValue, ...],
        *,
        mode: PredictionUpdateMode,
    ) -> PredictionState:
        prediction_slots = [prediction.slot for prediction in predictions]
        if len(prediction_slots) != len(set(prediction_slots)):
            raise ValueError("One prediction update cannot contain the same output slot more than once")

        current = {prediction.slot: prediction for prediction in self.values}
        overlap = current.keys() & prediction_slots
        if overlap and mode == PredictionUpdateMode.REQUIRE_EMPTY:
            raise ValueError(f"Prediction update would overwrite existing slots: {sorted(overlap)}")
        current.update((prediction.slot, prediction) for prediction in predictions)
        return PredictionState(tuple(current[slot] for slot in sorted(current)))


def prediction_input_record(
    state: PredictionState,
    slot: OutputSlot,
    *,
    position_id: int,
    features: tuple[FeatureId, ...] = (),
) -> Record:
    """Materialize a previously predicted value as document context."""
    return Record(state.value(slot), position_id, features=features)


@dataclass(frozen=True)
class PackedFeatureIds:
    channel: str
    ids: np.ndarray


@dataclass(frozen=True)
class PackedDocumentLocation:
    document_id: str
    row: int
    start: int
    end: int


@dataclass(frozen=True)
class PackedOutputLocation:
    slot: OutputSlot
    row: int
    position: int


@dataclass(frozen=True)
class PackedDocuments:
    token_ids: np.ndarray
    features: tuple[PackedFeatureIds, ...]
    rotary_position_ids: np.ndarray
    target_ids: np.ndarray
    loss_weights: np.ndarray
    segment_ids: np.ndarray
    attention_layout: AttentionLayout
    locations: tuple[PackedDocumentLocation, ...]
    outputs: tuple[PackedOutputLocation, ...]

    def feature_ids(self, channel: str) -> np.ndarray:
        for feature in self.features:
            if feature.channel == channel:
                return feature.ids
        return np.full_like(self.token_ids, -1)

    def prediction_values(self, sampled_token_ids: np.ndarray) -> tuple[PredictionValue, ...]:
        """Associate sampled tokens at output positions with their logical slots."""
        if sampled_token_ids.shape != self.token_ids.shape:
            raise ValueError(
                f"Sampled tokens must have packed shape {self.token_ids.shape}, got {sampled_token_ids.shape}"
            )
        return tuple(
            PredictionValue(output.slot, int(sampled_token_ids[output.row, output.position]))
            for output in self.outputs
        )


def pack_documents(documents: tuple[Document, ...], *, max_seq_len: int) -> PackedDocuments:
    """Greedily pack documents while preserving attention and output boundaries."""
    if not documents:
        raise ValueError("Cannot pack an empty document collection")
    attention_layouts = {document.attention_layout for document in documents}
    if len(attention_layouts) != 1:
        raise ValueError(f"Packed documents must share one attention layout, got {sorted(attention_layouts)}")
    attention_layout = attention_layouts.pop()
    if any(len(document.records) > max_seq_len for document in documents):
        longest = max(len(document.records) for document in documents)
        raise ValueError(f"Document length {longest} exceeds max_seq_len={max_seq_len}")

    rows: list[list[Document]] = [[]]
    row_lengths = [0]
    for document in documents:
        if row_lengths[-1] + len(document.records) > max_seq_len:
            rows.append([])
            row_lengths.append(0)
        rows[-1].append(document)
        row_lengths[-1] += len(document.records)

    shape = (len(rows), max_seq_len)
    token_ids = np.zeros(shape, dtype=np.int32)
    rotary_position_ids = np.zeros(shape, dtype=np.int32)
    target_ids = np.full(shape, -1, dtype=np.int32)
    loss_weights = np.zeros(shape, dtype=np.float32)
    segment_ids = np.full(shape, -1, dtype=np.int32)
    channels = tuple(sorted({channel for document in documents for channel in document.feature_channels}))
    feature_arrays = {channel: np.full(shape, -1, dtype=np.int32) for channel in channels}
    locations = []
    outputs = []

    for row_index, row in enumerate(rows):
        offset = 0
        for segment_id, document in enumerate(row):
            end = offset + len(document.records)
            token_ids[row_index, offset:end] = document.token_ids
            rotary_position_ids[row_index, offset:end] = document.rotary_position_ids
            target_ids[row_index, offset:end] = document.target_ids
            loss_weights[row_index, offset:end] = document.loss_weights
            segment_ids[row_index, offset:end] = segment_id
            for channel in channels:
                feature_arrays[channel][row_index, offset:end] = document.feature_ids(channel)
            locations.append(PackedDocumentLocation(document.id, row_index, offset, end))
            for position, slot in enumerate(document.output_slots):
                if slot is not None:
                    outputs.append(PackedOutputLocation(slot, row_index, offset + position))
            offset = end

    return PackedDocuments(
        token_ids=token_ids,
        features=tuple(PackedFeatureIds(channel, feature_arrays[channel]) for channel in channels),
        rotary_position_ids=rotary_position_ids,
        target_ids=target_ids,
        loss_weights=loss_weights,
        segment_ids=segment_ids,
        attention_layout=attention_layout,
        locations=tuple(locations),
        outputs=tuple(outputs),
    )
