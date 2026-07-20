# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np


class AttentionLayout(StrEnum):
    FULL = "full_segment"
    CAUSAL = "causal_segment"


@dataclass(frozen=True, eq=False)
class Coordinate:
    """Definition of one typed, token-aligned document coordinate."""

    name: str
    dtype: Any = np.int32
    missing: Any = -1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A coordinate requires a name")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        missing = np.asarray(self.missing, dtype=self.dtype)
        if missing.ndim != 0:
            raise ValueError("A coordinate missing value must be scalar")
        object.__setattr__(self, "missing", missing.item())


POSITION_IDS = Coordinate("position_ids", missing=0)
QUERY = Coordinate("query", dtype=np.bool_, missing=False)
TARGET_IDS = Coordinate("target_ids")
TARGET_WEIGHTS = Coordinate("target_weights", dtype=np.float32, missing=0.0)

RUNTIME_COORDINATES = (POSITION_IDS, QUERY, TARGET_IDS, TARGET_WEIGHTS)


class Document:
    """One attention domain with a single implicit token axis."""

    __slots__ = ("_coordinates", "attention", "token_ids")

    def __init__(
        self,
        token_ids: np.ndarray | tuple[int, ...] | list[int],
        coordinates: Mapping[Coordinate, np.ndarray | tuple[Any, ...] | list[Any]] | None = None,
        *,
        attention: AttentionLayout,
    ) -> None:
        self.token_ids = _vector(token_ids, np.dtype(np.int32), name="token_ids")
        if len(self.token_ids) == 0:
            raise ValueError("A document requires at least one token")
        self.attention = attention

        provided = coordinates or {}
        values = {
            coordinate: _aligned_vector(coordinate, coordinate_values, len(self.token_ids))
            for coordinate, coordinate_values in provided.items()
        }
        for coordinate in RUNTIME_COORDINATES:
            if coordinate not in values:
                values[coordinate] = _filled_coordinate(coordinate, len(self.token_ids))

        target_ids = values[TARGET_IDS]
        query = values[QUERY]
        targeted = target_ids != TARGET_IDS.missing
        if np.any(targeted & ~query):
            raise ValueError("Training targets must be attached to query positions")
        if TARGET_IDS in provided and TARGET_WEIGHTS not in provided:
            values[TARGET_WEIGHTS] = _readonly_array(targeted.astype(TARGET_WEIGHTS.dtype))
        target_weights = values[TARGET_WEIGHTS]
        if np.any(~np.isfinite(target_weights)) or np.any(target_weights < 0):
            raise ValueError("Target weights must be finite and non-negative")
        if np.any((~targeted) & (target_weights != 0)):
            raise ValueError("Target weights require training targets")

        self._coordinates = MappingProxyType(values)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __add__(self, other: Document) -> Document:
        return concatenate(self, other)

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(self._coordinates)

    @property
    def query_positions(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self[QUERY]))

    def __getitem__(self, coordinate: Coordinate) -> np.ndarray:
        values = self._coordinates.get(coordinate)
        if values is None:
            return _filled_coordinate(coordinate, len(self))
        return values

    def __getattr__(self, name: str) -> np.ndarray:
        coordinates = object.__getattribute__(self, "_coordinates")
        matches = [values for coordinate, values in coordinates.items() if coordinate.name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttributeError(f"Coordinate name {name!r} is ambiguous")
        raise AttributeError(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        if self.attention != other.attention or not np.array_equal(self.token_ids, other.token_ids):
            return False
        if self._coordinates.keys() != other._coordinates.keys():
            return False
        return all(np.array_equal(values, other[coordinate]) for coordinate, values in self._coordinates.items())

    def __repr__(self) -> str:
        coordinate_names = ", ".join(coordinate.name for coordinate in self.coordinates)
        return f"Document(tokens={len(self)}, coordinates=[{coordinate_names}], attention={self.attention.value!r})"

    def take(self, indices: np.ndarray | tuple[int, ...] | list[int]) -> Document:
        """Select or reorder token positions along the implicit axis."""
        indices_array = np.asarray(indices, dtype=np.intp)
        if indices_array.ndim != 1:
            raise ValueError(f"Document indices must be one-dimensional, got shape {indices_array.shape}")
        return Document(
            self.token_ids[indices_array],
            {coordinate: values[indices_array] for coordinate, values in self._coordinates.items()},
            attention=self.attention,
        )


def concatenate(*documents: Document) -> Document:
    """Concatenate documents along their implicit token axis."""
    if not documents:
        raise ValueError("concatenate requires at least one document")
    attentions = {document.attention for document in documents}
    if len(attentions) != 1:
        raise ValueError(f"Concatenated documents must share one attention layout, got {sorted(attentions)}")
    coordinates = tuple(dict.fromkeys(coordinate for document in documents for coordinate in document.coordinates))
    return Document(
        np.concatenate(tuple(document.token_ids for document in documents)),
        {
            coordinate: np.concatenate(tuple(document[coordinate] for document in documents))
            for coordinate in coordinates
        },
        attention=documents[0].attention,
    )


def causal_training_document(token_ids: tuple[int, ...]) -> Document:
    """Encode shifted next-token supervision as aligned query positions."""
    if len(token_ids) < 2:
        raise ValueError("Causal training documents require at least two tokens")
    length = len(token_ids)
    return Document(
        token_ids,
        {
            POSITION_IDS: np.arange(length, dtype=POSITION_IDS.dtype),
            QUERY: np.arange(length) + 1 < length,
            TARGET_IDS: (*token_ids[1:], TARGET_IDS.missing),
        },
        attention=AttentionLayout.CAUSAL,
    )


@dataclass(frozen=True)
class PackedBatch:
    token_ids: np.ndarray
    coordinates: dict[Coordinate, np.ndarray]
    segment_ids: np.ndarray
    document_indices: np.ndarray
    attention: AttentionLayout

    def __getitem__(self, coordinate: Coordinate) -> np.ndarray:
        values = self.coordinates.get(coordinate)
        if values is None:
            return np.full(self.token_ids.shape, coordinate.missing, dtype=coordinate.dtype)
        return values

    def __getattr__(self, name: str) -> np.ndarray:
        matches = [values for coordinate, values in self.coordinates.items() if coordinate.name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttributeError(f"Coordinate name {name!r} is ambiguous")
        raise AttributeError(name)


def pack(documents: tuple[Document, ...], *, max_seq_len: int) -> PackedBatch:
    """Greedily pack documents while preserving attention boundaries."""
    if not documents:
        raise ValueError("Cannot pack an empty document collection")
    attentions = {document.attention for document in documents}
    if len(attentions) != 1:
        raise ValueError(f"Packed documents must share one attention layout, got {sorted(attentions)}")
    attention = attentions.pop()
    if any(len(document) > max_seq_len for document in documents):
        longest = max(len(document) for document in documents)
        raise ValueError(f"Document length {longest} exceeds max_seq_len={max_seq_len}")

    rows: list[list[tuple[int, Document]]] = [[]]
    row_lengths = [0]
    for document_index, document in enumerate(documents):
        if row_lengths[-1] + len(document) > max_seq_len:
            rows.append([])
            row_lengths.append(0)
        rows[-1].append((document_index, document))
        row_lengths[-1] += len(document)

    shape = (len(rows), max_seq_len)
    token_ids = np.zeros(shape, dtype=np.int32)
    segment_ids = np.full(shape, -1, dtype=np.int32)
    document_indices = np.full(shape, -1, dtype=np.int32)
    coordinate_definitions = tuple(
        dict.fromkeys(coordinate for document in documents for coordinate in document.coordinates)
    )
    coordinates = {
        coordinate: np.full(shape, coordinate.missing, dtype=coordinate.dtype) for coordinate in coordinate_definitions
    }

    for row_index, row in enumerate(rows):
        offset = 0
        for segment_id, (document_index, document) in enumerate(row):
            end = offset + len(document)
            token_ids[row_index, offset:end] = document.token_ids
            segment_ids[row_index, offset:end] = segment_id
            document_indices[row_index, offset:end] = document_index
            for coordinate in coordinate_definitions:
                coordinates[coordinate][row_index, offset:end] = document[coordinate]
            offset = end

    return PackedBatch(
        token_ids=token_ids,
        coordinates=coordinates,
        segment_ids=segment_ids,
        document_indices=document_indices,
        attention=attention,
    )


def _aligned_vector(coordinate: Coordinate, values: Any, length: int) -> np.ndarray:
    array = _vector(values, coordinate.dtype, name=coordinate.name)
    if len(array) != length:
        raise ValueError(f"Coordinate {coordinate.name!r} has length {len(array)}, expected {length}")
    return array


def _filled_coordinate(coordinate: Coordinate, length: int) -> np.ndarray:
    return _readonly_array(np.full(length, coordinate.missing, dtype=coordinate.dtype))


def _vector(values: Any, dtype: np.dtype[Any], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return _readonly_array(array)


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array
