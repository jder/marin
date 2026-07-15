# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np


class AttentionLayout(StrEnum):
    FULL = "full_segment"
    CAUSAL = "causal_segment"


@dataclass(frozen=True)
class Token:
    """One transformer position, including optional query and training metadata."""

    input_id: int
    position_id: int = 0
    features: tuple[tuple[str, int], ...] = ()
    query: bool = False
    target_id: int | None = None
    target_weight: float = 1.0

    def __post_init__(self) -> None:
        channels = [channel for channel, _value in self.features]
        if len(channels) != len(set(channels)):
            raise ValueError(f"Token contains duplicate feature channels: {channels}")
        if self.target_id is not None and not self.query:
            raise ValueError("A training target must be attached to a query token")
        if not isfinite(self.target_weight) or self.target_weight < 0:
            raise ValueError(f"Target weight must be finite and non-negative, got {self.target_weight}")


@dataclass(frozen=True)
class Document:
    """An isolated attention domain containing encoded model tokens."""

    name: str
    tokens: tuple[Token, ...]
    attention_layout: AttentionLayout

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("A document requires at least one token")

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.input_id for token in self.tokens)

    @property
    def rotary_position_ids(self) -> tuple[int, ...]:
        return tuple(token.position_id for token in self.tokens)

    @property
    def target_ids(self) -> tuple[int, ...]:
        return tuple(token.target_id if token.target_id is not None else -1 for token in self.tokens)

    @property
    def loss_weights(self) -> tuple[float, ...]:
        return tuple(token.target_weight if token.target_id is not None else 0.0 for token in self.tokens)

    @property
    def query_positions(self) -> tuple[int, ...]:
        return tuple(index for index, token in enumerate(self.tokens) if token.query)

    @property
    def feature_channels(self) -> tuple[str, ...]:
        return tuple(sorted({channel for token in self.tokens for channel, _value in token.features}))

    def feature_ids(self, channel: str) -> tuple[int, ...]:
        return tuple(
            next((value for feature_channel, value in token.features if feature_channel == channel), -1)
            for token in self.tokens
        )

    def reordered(self, order: tuple[int, ...]) -> Document:
        """Return the same tokens in a different physical order."""
        if tuple(sorted(order)) != tuple(range(len(self.tokens))):
            raise ValueError("Token order must be a permutation of all document positions")
        return Document(self.name, tuple(self.tokens[index] for index in order), self.attention_layout)

    def selected(self, name: str, token_indices: tuple[int, ...]) -> Document:
        """Create a context view or prediction shard over selected tokens."""
        if len(set(token_indices)) != len(token_indices):
            raise ValueError("A document view cannot repeat token indices")
        if any(index < 0 or index >= len(self.tokens) for index in token_indices):
            raise IndexError(f"Token view {token_indices} is outside document length {len(self.tokens)}")
        return Document(name, tuple(self.tokens[index] for index in token_indices), self.attention_layout)


def causal_training_document(name: str, token_ids: tuple[int, ...]) -> Document:
    """Encode shifted next-token supervision as aligned query tokens."""
    if len(token_ids) < 2:
        raise ValueError("Causal training documents require at least two tokens")
    tokens = tuple(
        Token(
            token_id,
            position_id=index,
            query=index + 1 < len(token_ids),
            target_id=token_ids[index + 1] if index + 1 < len(token_ids) else None,
        )
        for index, token_id in enumerate(token_ids)
    )
    return Document(name, tokens, AttentionLayout.CAUSAL)


@dataclass(frozen=True)
class PackedBatch:
    token_ids: np.ndarray
    features: dict[str, np.ndarray]
    rotary_position_ids: np.ndarray
    target_ids: np.ndarray
    loss_weights: np.ndarray
    segment_ids: np.ndarray
    document_indices: np.ndarray
    query_mask: np.ndarray
    attention_layout: AttentionLayout

    def feature_ids(self, channel: str) -> np.ndarray:
        feature_ids = self.features.get(channel)
        if feature_ids is None:
            return np.full_like(self.token_ids, -1)
        return feature_ids


def pack(documents: tuple[Document, ...], *, max_seq_len: int) -> PackedBatch:
    """Greedily pack documents while preserving attention and document boundaries."""
    if not documents:
        raise ValueError("Cannot pack an empty document collection")
    attention_layouts = {document.attention_layout for document in documents}
    if len(attention_layouts) != 1:
        raise ValueError(f"Packed documents must share one attention layout, got {sorted(attention_layouts)}")
    attention_layout = attention_layouts.pop()
    if any(len(document.tokens) > max_seq_len for document in documents):
        longest = max(len(document.tokens) for document in documents)
        raise ValueError(f"Document length {longest} exceeds max_seq_len={max_seq_len}")

    rows: list[list[tuple[int, Document]]] = [[]]
    row_lengths = [0]
    for document_index, document in enumerate(documents):
        if row_lengths[-1] + len(document.tokens) > max_seq_len:
            rows.append([])
            row_lengths.append(0)
        rows[-1].append((document_index, document))
        row_lengths[-1] += len(document.tokens)

    shape = (len(rows), max_seq_len)
    token_ids = np.zeros(shape, dtype=np.int32)
    rotary_position_ids = np.zeros(shape, dtype=np.int32)
    target_ids = np.full(shape, -1, dtype=np.int32)
    loss_weights = np.zeros(shape, dtype=np.float32)
    segment_ids = np.full(shape, -1, dtype=np.int32)
    document_indices = np.full(shape, -1, dtype=np.int32)
    query_mask = np.zeros(shape, dtype=np.bool_)
    channels = tuple(sorted({channel for document in documents for channel in document.feature_channels}))
    features = {channel: np.full(shape, -1, dtype=np.int32) for channel in channels}

    for row_index, row in enumerate(rows):
        offset = 0
        for segment_id, (document_index, document) in enumerate(row):
            end = offset + len(document.tokens)
            token_ids[row_index, offset:end] = document.token_ids
            rotary_position_ids[row_index, offset:end] = document.rotary_position_ids
            target_ids[row_index, offset:end] = document.target_ids
            loss_weights[row_index, offset:end] = document.loss_weights
            segment_ids[row_index, offset:end] = segment_id
            document_indices[row_index, offset:end] = document_index
            query_mask[row_index, offset:end] = tuple(token.query for token in document.tokens)
            for channel in channels:
                features[channel][row_index, offset:end] = document.feature_ids(channel)
            offset = end

    return PackedBatch(
        token_ids=token_ids,
        features=features,
        rotary_position_ids=rotary_position_ids,
        target_ids=target_ids,
        loss_weights=loss_weights,
        segment_ids=segment_ids,
        document_indices=document_indices,
        query_mask=query_mask,
        attention_layout=attention_layout,
    )
