"""Gemma-3 text tower + audio text connector on MLX."""

from __future__ import annotations

import json
import math

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from ltx_core_mlx.text_encoders.gemma.embeddings_connector import Embeddings1DConnector

GEMMA_HIDDEN = 3840
NUM_HIDDEN_STATES = 49     # embedding + 47 layer outputs + final norm
AUDIO_DIM = 2048
MAX_LEN = 1024

CONNECTOR_NUM_LAYERS = 8
CONNECTOR_NUM_REGISTERS = 128
CONNECTOR_HEADS = 32
CONNECTOR_HEAD_DIM = 64
CONNECTOR_MAX_POS = 4096

# Streamed projection chunk: part of the numerics contract (bf16 GEMM reduction
# order changes with the M dimension), not a free tuning knob. Must be >= MAX_LEN/2
# (512 of 1024) so MLX reuses the eager GEMM kernel: 64/96/128/192/256 all select a
# different kernel and drift to rel-L2 5.6e-2 on the outlier-heavy DEFAULT_NEG;
# 512 stays at 8.1e-3 for DEFAULT_NEG and 4.9e-3 for the positive prompt while still
# cutting the connector transient 3.25 -> 2.74 GB. Re-validate if MAX_LEN grows.
CONNECTOR_CHUNK_TOKENS = 512

# Connector params from the checkpoint header; present-and-different is a hard
# error, absent warns (the components file may be re-exported without a config).
_CONNECTOR_CONFIG = {
    "connector_num_layers": CONNECTOR_NUM_LAYERS,
    "connector_num_learnable_registers": CONNECTOR_NUM_REGISTERS,
    "audio_connector_num_attention_heads": CONNECTOR_HEADS,
    "audio_connector_attention_head_dim": CONNECTOR_HEAD_DIM,
    "connector_apply_gated_attention": True,
    "connector_norm_output": True,
}

NUM_CONNECTOR_TENSORS = 131
_TEXT_PREFIX_HINTS = ("text_embedding_projection.", "model.diffusion_model.audio_embeddings_connector.")


class AudioTextConnector(nn.Module):
    """Audio-only text embedding projection + 1D connector."""

    def __init__(self):
        super().__init__()
        self.audio_aggregate_embed = nn.Linear(NUM_HIDDEN_STATES * GEMMA_HIDDEN, AUDIO_DIM, bias=True)
        self.audio_embeddings_connector = Embeddings1DConnector(
            dim=AUDIO_DIM,
            num_heads=CONNECTOR_HEADS,
            head_dim=CONNECTOR_HEAD_DIM,
            num_layers=CONNECTOR_NUM_LAYERS,
            num_registers=CONNECTOR_NUM_REGISTERS,
            ff_mult=4.0,
            max_pos=CONNECTOR_MAX_POS,
            norm_output=True,
            apply_gated_attention=True,
            double_precision_rope=True,
        )

    def __call__(self, hidden_states: list[mx.array], attention_mask: mx.array) -> mx.array:
        # Stream the 49 x 3840 -> 2048 projection in chunks (512 tokens with the
        # default MAX_LEN) so the (T, 3840, 49) stack and the (T, 49*3840) input
        # never materialise for the whole sequence. Per chunk: RMS over the
        # (chunk, 3840, 49) stack, mask, project. Same op order/dtypes as the eager
        # path; only the GEMM M changes (see CONNECTOR_CHUNK_TOKENS).
        T = hidden_states[0].shape[1]
        scale = math.sqrt(AUDIO_DIM / GEMMA_HIDDEN)
        feats = []
        chunk = min(CONNECTOR_CHUNK_TOKENS, T)
        for start in range(0, T, chunk):
            sl = slice(start, min(start + chunk, T))
            encoded = mx.stack([h[:, sl] for h in hidden_states], axis=-1)   # (B, c, D, L) bf16
            variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
            normed = encoded * mx.rsqrt(variance + 1e-6)
            B, c, D, L = normed.shape
            normed = normed.reshape(B, c, D * L)
            normed = normed * attention_mask[:, sl, None].astype(normed.dtype)
            feat = self.audio_aggregate_embed(normed * scale)
            # The graph is lazy: without this the chunk stacks of all iterations
            # stay resident until the connector's first block evals (i.e. no
            # saving at all). Evaluating the chunk feature frees each chunk.
            mx.eval(feat)
            feats.append(feat)
        # The connector itself stays full length (registers, RoPE and masks unchanged).
        return self.audio_embeddings_connector(mx.concatenate(feats, axis=1), attention_mask=attention_mask)


def remap_text_key(key: str) -> str | None:
    prefix = "text_embedding_projection.audio_aggregate_embed."
    if key.startswith(prefix):
        return "audio_aggregate_embed." + key[len(prefix):]
    prefix = "model.diffusion_model.audio_embeddings_connector."
    if key.startswith(prefix):
        return "audio_embeddings_connector." + key[len(prefix):]
    return None


def _read_transformer_config(path: str) -> dict | None:
    """Return the checkpoint's transformer config, or None (advisory for components)."""
    try:
        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata() or {}
        raw = metadata.get("config")
        if raw is None:
            return None
        cfg = json.loads(raw)["transformer"]
        return cfg if isinstance(cfg, dict) else None
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[mlx] warning: {path}: unreadable checkpoint config ({exc})", flush=True)
        return None


def _validate_connector_config(cfg: dict) -> None:
    for key, expected in _CONNECTOR_CONFIG.items():
        if key not in cfg:
            print(f"[mlx] warning: checkpoint config has no {key!r}; assuming {expected!r}", flush=True)
        elif cfg[key] != expected:
            raise ValueError(f"checkpoint config {key}={cfg[key]!r} (expected {expected!r})")

    max_pos = cfg.get("connector_positional_embedding_max_pos")
    if max_pos is None:
        print(f"[mlx] warning: checkpoint config has no 'connector_positional_embedding_max_pos'; "
              f"assuming [{CONNECTOR_MAX_POS}]", flush=True)
    else:
        values = [float(max_pos)] if isinstance(max_pos, (int, float)) else [float(v) for v in max_pos]
        if values != [float(CONNECTOR_MAX_POS)]:
            raise ValueError(
                f"checkpoint config connector_positional_embedding_max_pos={max_pos!r} "
                f"(expected [{CONNECTOR_MAX_POS}])"
            )


def load_connector(checkpoint_path: str) -> AudioTextConnector:
    cfg = _read_transformer_config(checkpoint_path)
    if cfg is None:
        print(f"[mlx] warning: {checkpoint_path}: no checkpoint config, connector architecture unverified",
              flush=True)
    else:
        _validate_connector_config(cfg)

    connector = AudioTextConnector()
    raw = mx.load(checkpoint_path)
    weights = {}
    unmatched = []
    for key, value in raw.items():
        new_key = remap_text_key(key)
        if new_key is None:
            # Warn only about keys that look like they belong to the text path (the
            # components file also holds the VAE/vocoder, which is loaded elsewhere).
            if key.startswith(_TEXT_PREFIX_HINTS):
                unmatched.append(key)
            continue
        weights[new_key] = value.astype(mx.bfloat16)
    del raw

    if len(weights) != NUM_CONNECTOR_TENSORS:
        print(f"[mlx] warning: {checkpoint_path}: mapped {len(weights)} connector tensors "
              f"(expected {NUM_CONNECTOR_TENSORS})", flush=True)
    if unmatched:
        print(f"[mlx] warning: {checkpoint_path}: {len(unmatched)} text-path tensors not mapped "
              f"(e.g. {unmatched[:3]})", flush=True)

    connector.load_weights(list(weights.items()))
    del weights
    mx.clear_cache()
    return connector


class GemmaTextEncoder:
    """Wraps an mlx-lm Gemma-3 model and extracts the 49 hidden states.

    transformers 4.51.0 (the version this checkpoint targets) returns
    ``[scaled embedding, out(layer 0) .. out(layer 46), norm(out(layer 47))]``,
    which is 48 loop outputs plus the post-final-norm output. Verified against
    transformers 4.51.0 with a small Gemma3TextModel.
    """

    def __init__(self, model_dir: str):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_dir)
        self.inner = self.model.language_model.model
        self.args = self.inner.args

        # The shared tokenizer wrapper's detokenizer is unaffected; set once here.
        self.hf = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.hf.padding_side = "left"
        if self.hf.pad_token is None:
            self.hf.pad_token = self.hf.eos_token

        n_layers = len(self.inner.layers)
        if not n_layers == self.args.num_hidden_layers == NUM_HIDDEN_STATES - 1:
            raise ValueError(
                f"Gemma tower has {n_layers} layers (args.num_hidden_layers={self.args.num_hidden_layers}); "
                f"the audio connector expects {NUM_HIDDEN_STATES - 1}"
            )

    def tokenize(self, text: str) -> tuple[mx.array, mx.array]:
        enc = self.hf(
            text.strip(),
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="np",
        )
        return mx.array(enc["input_ids"]), mx.array(enc["attention_mask"])

    def hidden_states(self, input_ids: mx.array, attention_mask: mx.array) -> list[mx.array]:
        from mlx_lm.models.base import create_causal_mask

        inner = self.inner
        h = inner.embed_tokens(input_ids)
        h = h * mx.array(inner.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        layers = inner.layers
        pattern = self.args.sliding_window_pattern
        N = input_ids.shape[1]
        left_pad = N - attention_mask.sum(axis=1)  # (B,)
        global_mask = create_causal_mask(N, left_padding=left_pad)
        # HF builds the sliding window as triu(causal, diagonal=-W), i.e. j in [i-W, i].
        sliding_mask = create_causal_mask(N, window_size=self.args.sliding_window + 1, left_padding=left_pad)

        states = [h.astype(mx.bfloat16)]
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            is_global = (i % pattern) == (pattern - 1)
            h = layer(h, global_mask if is_global else sliding_mask, None)
            if i < last:
                states.append(h.astype(mx.bfloat16))
            if (i + 1) % 8 == 0:
                mx.eval(h)
        states.append(inner.norm(h).astype(mx.bfloat16))
        return states

    def encode(self, text: str, connector: AudioTextConnector) -> mx.array:
        ids, mask = self.tokenize(text)
        states = self.hidden_states(ids, mask)
        ctx = connector(states, mask)
        mx.eval(ctx)
        return ctx

    def free(self) -> None:
        self.model = None
        self.inner = None
        mx.clear_cache()
