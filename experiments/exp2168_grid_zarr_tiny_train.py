# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny training experiment for grid-token Zarr data.

This follows the same shape as `experiments/tutorials/train_tiny_model_cpu.py`,
but replaces text tokenization with a cache-building step from a grid-token Zarr export.

Grid-token context for this experiment:
- A timestep is flattened over grid levels/pixels/token slots/RVQ codebooks.
- Token IDs are remapped with per-position codebook offsets so each codebook has
  its own ID range (same raw code in two codebooks becomes different IDs).
- Each timestep starts with a dedicated start token; masked/land entries are mapped
  to a shared land token ID.
- `sequence_ordering=prog_first` (default) reorders positions so prognostic tokens
  come before forcing tokens before chunking into fixed-length LM sequences.
- Base LM positional embeddings are standard 1..N positions; codebook/slot identity is
  represented in token IDs (via the offset remap), not in position indices.
- When `GRID_TRAIN_USE_SPATIAL_EMBEDDINGS=1`, the model also adds a learned
  per-position spatial embedding table derived from Zarr metadata (lat/lon trig features,
  level/slot/codebook ids, prognostic/non-land flags). This table is built in-memory at
  model init, participates in training, and is saved in checkpoints as
  `spatial_position_embeddings`.
- `GRID_TRAIN_SPATIAL_EMBED_DEMO=1` optionally writes a standalone demo `.npz` artifact
  showing the same feature projection, but this artifact is not required for training.
- Train loss uses non-land tokens across all steps in the history window, while eval
  loss is concentrated on prognostic, non-land tokens in the final step.
"""

import dataclasses
import io
import json
import os
from dataclasses import dataclass
from typing import Literal

import fsspec
import haliax as hax
import jax.numpy as jnp
import numpy as np
from fray.v2 import ResourceConfig
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray
from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from marin.execution import THIS_OUTPUT_PATH, ExecutorStep
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.tokenize.grid_zarr_cache import GridZarrTokenizeConfig, grid_zarr_to_pretokenized_cache
from marin.tokenize.grid_zarr_loader import GridTokenZarrSource, load_grid_sequence_layout


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_opt_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _source_from_env() -> GridTokenZarrSource:
    local_path = os.environ.get("GRID_TOKENS_ZARR_PATH")
    if local_path:
        return GridTokenZarrSource(path=os.path.expanduser(local_path))

    hf_repo_id = os.environ.get("GRID_TOKENS_HF_REPO_ID")
    if hf_repo_id:
        return GridTokenZarrSource(
            hf_repo_id=hf_repo_id,
            hf_revision=os.environ.get("GRID_TOKENS_HF_REVISION") or "main",
            hf_subpath=os.environ.get("GRID_TOKENS_HF_SUBPATH"),
            hf_mode=os.environ.get("GRID_TOKENS_HF_MODE", "stage"),
            hf_token=os.environ.get("HF_TOKEN"),
        )

    raise ValueError(
        "Set either GRID_TOKENS_ZARR_PATH or GRID_TOKENS_HF_REPO_ID for exp2168_grid_zarr_tiny_train."
    )


def _build_spatial_position_embedding_array(
    *,
    source: GridTokenZarrSource,
    max_levels: int | None,
    max_codebooks: int | None,
    sequence_ordering: Literal["prog_first", "storage"],
    n_history: int,
    sequence_length: int,
    embed_dim: int,
    seed: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic per-position spatial features and a projected embedding table."""
    layout, _ = load_grid_sequence_layout(
        source,
        max_levels=max_levels,
        max_codebooks=max_codebooks,
        sequence_ordering=sequence_ordering,
    )
    metadata = layout.token_metadata()

    level_denom = max(len(layout.levels) - 1, 1)
    slot_denom = max(int(np.max(metadata.slot_ids)), 1)
    codebook_denom = max(int(np.max(metadata.codebook_ids)), 1)

    token_features = np.concatenate(
        [
            metadata.latlon_features.astype(np.float32),
            (metadata.level_ids.astype(np.float32) / level_denom)[:, None],
            (metadata.slot_ids.astype(np.float32) / slot_denom)[:, None],
            (metadata.codebook_ids.astype(np.float32) / codebook_denom)[:, None],
            metadata.prognostic_mask.astype(np.float32)[:, None],
            (~metadata.land_mask).astype(np.float32)[:, None],
        ],
        axis=1,
    )

    start_token_features = np.zeros((1, token_features.shape[1]), dtype=np.float32)
    per_step_features = np.concatenate([start_token_features, token_features], axis=0)
    steps = n_history + 1
    window_features = np.tile(per_step_features, (steps, 1))

    if sequence_length > window_features.shape[0]:
        raise ValueError(
            f"sequence_length={sequence_length} exceeds one window length={window_features.shape[0]}. "
            "Increase n_history or lower sequence_length."
        )

    position_features = window_features[:sequence_length]
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((position_features.shape[1], embed_dim)).astype(np.float32)
    projection *= np.float32(scale / np.sqrt(position_features.shape[1]))
    position_embedding = position_features @ projection
    return position_embedding.astype(np.float32), position_features.astype(np.float32), projection.astype(np.float32)


@dataclass(frozen=True)
class GridSpatialPositionalEmbeddingDemoConfig:
    """Configuration for materializing a demo spatial position-embedding table."""

    source: GridTokenZarrSource
    output_path: str = THIS_OUTPUT_PATH
    max_levels: int | None = 2
    max_codebooks: int | None = 1
    sequence_ordering: Literal["prog_first", "storage"] = "prog_first"
    n_history: int = 2
    sequence_length: int = 512
    embed_dim: int = 32
    seed: int = 0


def _run_spatial_positional_embedding_demo(cfg: GridSpatialPositionalEmbeddingDemoConfig) -> dict[str, object]:
    """Build an example per-position embedding table from grid spatial metadata."""
    position_embedding, position_features, projection = _build_spatial_position_embedding_array(
        source=cfg.source,
        max_levels=cfg.max_levels,
        max_codebooks=cfg.max_codebooks,
        sequence_ordering=cfg.sequence_ordering,
        n_history=cfg.n_history,
        sequence_length=cfg.sequence_length,
        embed_dim=cfg.embed_dim,
        seed=cfg.seed,
        scale=1.0,
    )

    fs, output_root = fsspec.core.url_to_fs(cfg.output_path)
    fs.makedirs(output_root, exist_ok=True)

    npz_path = os.path.join(cfg.output_path, "grid_spatial_position_embedding_demo.npz")
    with io.BytesIO() as buffer:
        np.savez(
            buffer,
            position_embedding=position_embedding.astype(np.float32),
            position_features=position_features.astype(np.float32),
            projection=projection.astype(np.float32),
        )
        payload = buffer.getvalue()
    with fsspec.open(npz_path, "wb") as f:
        f.write(payload)

    summary = {
        "sequence_length": int(cfg.sequence_length),
        "embed_dim": int(cfg.embed_dim),
        "feature_dim": int(position_features.shape[1]),
        "feature_names": [
            "sin(lat)",
            "cos(lat)",
            "sin(lon)",
            "cos(lon)",
            "level_norm",
            "slot_norm",
            "codebook_norm",
            "is_prognostic",
            "is_non_land",
        ],
        "position_embedding_path": npz_path,
        "usage_note": (
            "Load `position_embedding` and add it to token embeddings in the model forward pass, "
            "broadcasting across batch, e.g. x = token_embed(input_ids) + spatial_pos_embed[Pos]."
        ),
    }
    summary_path = os.path.join(cfg.output_path, "grid_spatial_position_embedding_demo_summary.json")
    with fsspec.open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def grid_spatial_positional_embedding_demo_step(
    source: GridTokenZarrSource,
    *,
    max_levels: int | None,
    max_codebooks: int | None,
    sequence_ordering: Literal["prog_first", "storage"],
    n_history: int,
    sequence_length: int,
    embed_dim: int,
    seed: int,
) -> ExecutorStep[GridSpatialPositionalEmbeddingDemoConfig]:
    """Create an optional step that writes a demo spatial position-embedding artifact."""
    return ExecutorStep(
        name="artifacts/grid-spatial-positional-embedding-demo",
        description="Create a demo position-embedding table from Zarr spatial metadata.",
        fn=_run_spatial_positional_embedding_demo,
        config=GridSpatialPositionalEmbeddingDemoConfig(
            source=source,
            max_levels=max_levels,
            max_codebooks=max_codebooks,
            sequence_ordering=sequence_ordering,
            n_history=n_history,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            seed=seed,
        ),
        resources=ResourceConfig.with_cpu(cpu=2, ram="4g", disk="4g"),
    )


@LlamaConfig.register_subclass("spatial_grid_llama")
@dataclass(frozen=True)
class SpatialGridLlamaConfig(LlamaConfig):
    """LLaMA config that seeds an additive spatial position table from grid metadata."""

    spatial_source_json: str = ""
    spatial_max_levels: int | None = 2
    spatial_max_codebooks: int | None = 1
    spatial_sequence_ordering: Literal["prog_first", "storage"] = "prog_first"
    spatial_n_history: int = 2
    spatial_projection_seed: int = 0
    spatial_projection_scale: float = 0.1

    @property
    def model_type(self) -> type["SpatialGridLlamaLMHeadModel"]:
        return SpatialGridLlamaLMHeadModel

    def __post_init__(self):
        super().__post_init__()
        if not self.spatial_source_json:
            raise ValueError("SpatialGridLlamaConfig requires `spatial_source_json`.")


class SpatialGridLlamaLMHeadModel(LlamaLMHeadModel):
    """LLaMA head model with additive, trainable spatial position embeddings."""

    spatial_position_embeddings: NamedArray

    @classmethod
    def init(cls, Vocab: Axis, config: SpatialGridLlamaConfig, *, key: PRNGKeyArray) -> "SpatialGridLlamaLMHeadModel":
        base = LlamaLMHeadModel.init(Vocab, config, key=key)
        source = GridTokenZarrSource(**json.loads(config.spatial_source_json))

        spatial_table, _, _ = _build_spatial_position_embedding_array(
            source=source,
            max_levels=config.spatial_max_levels,
            max_codebooks=config.spatial_max_codebooks,
            sequence_ordering=config.spatial_sequence_ordering,
            n_history=config.spatial_n_history,
            sequence_length=config.max_seq_len,
            embed_dim=config.hidden_dim,
            seed=config.spatial_projection_seed,
            scale=config.spatial_projection_scale,
        )
        spatial_embeddings = hax.named(jnp.asarray(spatial_table, dtype=jnp.float32), (config.max_Pos, config.Embed))

        return cls(
            transformer=base.transformer,
            embeddings=base.embeddings,
            lm_head=base.lm_head,
            spatial_position_embeddings=spatial_embeddings,
        )

    def _spatial_embeddings_for_input(self, input_ids: NamedArray) -> NamedArray:
        pos = input_ids.resolve_axis("position")
        table_pos = self.spatial_position_embeddings.resolve_axis("position")
        if pos.size > table_pos.size:
            raise ValueError(
                f"Input position axis size {pos.size} exceeds spatial table size {table_pos.size}. "
                "Lower train_seq_len or increase model max_seq_len."
            )
        if pos.size == table_pos.size:
            return self.spatial_position_embeddings

        return hax.take(self.spatial_position_embeddings, axis=table_pos, index=hax.arange(pos))

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x = x + self._spatial_embeddings_for_input(input_ids).astype(x.dtype)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        x = self.activations(input_ids, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        if self.lm_head is not None:
            return self.lm_head(x, key=None)
        return self.embeddings.unembed(x)

    def resize_vocab(self, new_size: int, key=None):
        resized = super().resize_vocab(new_size, key=key)
        if not isinstance(resized, LlamaLMHeadModel):
            raise TypeError(f"Expected LlamaLMHeadModel from resize_vocab, got {type(resized)}")
        return dataclasses.replace(self, embeddings=resized.embeddings, lm_head=resized.lm_head)

    def _state_dict_key_map(self):
        mapping = dict(super()._state_dict_key_map())
        mapping["spatial_position_embeddings"] = "spatial_position_embeddings"
        return mapping


def _to_spatial_grid_llama_config(
    *,
    base: LlamaConfig,
    source: GridTokenZarrSource,
    max_levels: int | None,
    max_codebooks: int | None,
    sequence_ordering: Literal["prog_first", "storage"],
    n_history: int,
    projection_seed: int,
    projection_scale: float,
) -> SpatialGridLlamaConfig:
    base_kwargs = {field.name: getattr(base, field.name) for field in dataclasses.fields(LlamaConfig)}
    return SpatialGridLlamaConfig(
        **base_kwargs,
        spatial_source_json=json.dumps(dataclasses.asdict(source), sort_keys=True),
        spatial_max_levels=max_levels,
        spatial_max_codebooks=max_codebooks,
        spatial_sequence_ordering=sequence_ordering,
        spatial_n_history=n_history,
        spatial_projection_seed=projection_seed,
        spatial_projection_scale=projection_scale,
    )


sequence_ordering_env = os.environ.get("GRID_TRAIN_SEQUENCE_ORDERING", "prog_first")
if sequence_ordering_env not in {"prog_first", "storage"}:
    raise ValueError(
        "GRID_TRAIN_SEQUENCE_ORDERING must be one of {'prog_first', 'storage'}, "
        f"got {sequence_ordering_env!r}."
    )
sequence_ordering: Literal["prog_first", "storage"] = sequence_ordering_env
max_train_windows = _env_opt_int("GRID_TRAIN_MAX_WINDOWS")
max_validation_windows = _env_opt_int("GRID_TRAIN_MAX_VALIDATION_WINDOWS")
max_levels = _env_opt_int("GRID_TRAIN_MAX_LEVELS")
max_codebooks = _env_opt_int("GRID_TRAIN_MAX_CODEBOOKS")
history_steps = _env_int("GRID_TRAIN_HISTORY_STEPS", 2)
sequence_length = _env_int("GRID_TRAIN_SEQUENCE_LENGTH", 512)
split_seed = _env_int("GRID_TRAIN_SPLIT_SEED", 0)
enable_spatial_embeddings = _env_bool("GRID_TRAIN_USE_SPATIAL_EMBEDDINGS", default=False)
enable_spatial_demo = _env_bool("GRID_TRAIN_SPATIAL_EMBED_DEMO", default=False)
spatial_embed_dim = _env_int("GRID_TRAIN_SPATIAL_EMBED_DIM", 32)
spatial_embed_seed = _env_int("GRID_TRAIN_SPATIAL_EMBED_SEED", 0)
spatial_embed_scale = _env_float("GRID_TRAIN_SPATIAL_EMBED_SCALE", 0.1)
use_gpu = _env_bool("GRID_TRAIN_USE_GPU", default=False)
grid_source = _source_from_env()
resource_config = (
    ResourceConfig.with_gpu(
        gpu_type=os.environ.get("GRID_TRAIN_GPU_TYPE", "auto"),
        count=_env_int("GRID_TRAIN_GPU_COUNT", 1),
    )
    if use_gpu
    else ResourceConfig.with_cpu()
)

grid_tokenized = grid_zarr_to_pretokenized_cache(
    name="tokenized/grid-zarr-tiny-train-cache",
    config=GridZarrTokenizeConfig(
        source=grid_source,
        tokenizer=marin_tokenizer,
        max_levels=max_levels if max_levels is not None else 2,
        max_codebooks=max_codebooks if max_codebooks is not None else 1,
        sequence_ordering=sequence_ordering,
        n_history=history_steps,
        sequence_length=sequence_length,
        max_train_windows=max_train_windows if max_train_windows is not None else 4096,
        max_validation_windows=max_validation_windows if max_validation_windows is not None else 512,
        split_seed=split_seed,
        tags=["grid-zarr", "prebuilt"],
    ),
)


tiny_grid_train_config = SimpleTrainConfig(
    resources=resource_config,
    train_batch_size=_env_int("GRID_TRAIN_BATCH_SIZE", 4),
    num_train_steps=_env_int("GRID_TRAIN_NUM_STEPS", 100),
    train_seq_len=sequence_length,
    learning_rate=6e-4,
    weight_decay=0.1,
    max_eval_batches=_env_int("GRID_TRAIN_MAX_EVAL_BATCHES", 4),
)

model_config_for_train: LlamaConfig = llama_nano
if enable_spatial_embeddings:
    if llama_nano.max_seq_len != sequence_length:
        model_config_for_train = dataclasses.replace(llama_nano, max_seq_len=sequence_length)
    model_config_for_train = _to_spatial_grid_llama_config(
        base=model_config_for_train,
        source=grid_source,
        max_levels=max_levels if max_levels is not None else 2,
        max_codebooks=max_codebooks if max_codebooks is not None else 1,
        sequence_ordering=sequence_ordering,
        n_history=history_steps,
        projection_seed=spatial_embed_seed,
        projection_scale=spatial_embed_scale,
    )


grid_nano_model = default_train(
    name="marin-nano-grid-zarr",
    tokenized=grid_tokenized,
    model_config=versioned(model_config_for_train),
    train_config=tiny_grid_train_config,
    tags=["llama", "nano", "grid-zarr", "tutorial-like"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    steps = []
    if enable_spatial_demo:
        steps.append(
            grid_spatial_positional_embedding_demo_step(
                source=grid_source,
                max_levels=max_levels if max_levels is not None else 2,
                max_codebooks=max_codebooks if max_codebooks is not None else 1,
                sequence_ordering=sequence_ordering,
                n_history=history_steps,
                sequence_length=sequence_length,
                embed_dim=spatial_embed_dim,
                seed=spatial_embed_seed,
            )
        )
    steps.append(grid_nano_model)

    executor_main(
        steps=steps,
        description="Tiny Marin training experiment over grid-token Zarr data.",
    )
