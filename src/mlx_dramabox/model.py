"""Audio-only LTX-2.3 DiT for DramaBox, implemented on MLX.

The upstream DramaBox checkpoint stores only the audio branch of the joint
LTX-2.3 AV transformer (48 blocks, audio attention + audio FF, no video
weights). This module reconstructs exactly that branch, reusing the
attention / feed-forward / AdaLN primitives from `ltx_core_mlx`.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.feed_forward import FeedForward
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding

AUDIO_DIM = 2048
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_CROSS_DIM = 2048
AUDIO_CHANNELS = 128
CAPTION_CHANNELS = 3840
FF_MULT = 4.0
NUM_LAYERS = 48
NORM_EPS = 1e-6
TIMESTEP_DIM = 256
ROPE_MAX_POS = [20.0]
ROPE_TYPE = "split"

# The safetensors header carries the config the checkpoint was exported with.
# A mismatch is a hard error: several of these keys (rope_type,
# frequencies_precision, audio_positional_embedding_max_pos) load all 1457
# tensors fine and then silently compute different rotations/masks.
_DIT_CONFIG_REQUIRED = {
    "num_layers": NUM_LAYERS,
    "audio_num_attention_heads": AUDIO_HEADS,
    "audio_attention_head_dim": AUDIO_HEAD_DIM,
    "audio_cross_attention_dim": AUDIO_CROSS_DIM,
    "caption_channels": CAPTION_CHANNELS,
    "frequencies_precision": "float64",
    "positional_embedding_theta": 10000.0,
    "use_middle_indices_grid": True,
    "timestep_scale_multiplier": 1000,
    "norm_eps": NORM_EPS,
    "apply_gated_attention": True,
    "cross_attention_adaln": True,
    "caption_proj_before_connector": True,
    "text_encoder_norm_type": "per_token_rms",
}
# Optional keys: warn when absent, hard error when present and different.
_DIT_CONFIG_OPTIONAL = {
    "in_channels": AUDIO_CHANNELS,
    "out_channels": AUDIO_CHANNELS,
    "audio_out_channels": AUDIO_CHANNELS,
}

NUM_DIT_TENSORS = 1457
_KNOWN_DIT_FILES = ("dramabox-dit-v1.safetensors",)
_KNOWN_DIT_VERSIONS = ("2.3.0",)

# Exactly the 10 per-block projections are quantized. Everything else (the
# AdaLN subtree, patchify, the velocity head, the per-head gate logits and the
# normative QK RMSNorms) stays in bf16: `nn.quantize` raises on non-Linear
# leaves, and the small/gain-sensitive tensors are not worth 4/8-bit noise.
_QUANT_ALLOW = (
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out",
    "audio_ff.proj_in",
    "audio_ff.proj_out",
)


def _in_quant_allow(path: str) -> bool:
    """True when a module path is one of the 10 quantized per-block projections."""
    if not path.startswith("transformer_blocks."):
        return False
    parts = path.split(".", 2)
    return len(parts) == 3 and parts[2] in _QUANT_ALLOW


def _quant_pred(path: str, module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if not path.startswith("transformer_blocks."):
        return False
    return path.split(".", 2)[2] in _QUANT_ALLOW


def make_rope(mid_positions: mx.array, rope_type: str = ROPE_TYPE):
    """Exact port of upstream precompute_freqs_cis for the 1-D audio grid.

    Args:
        mid_positions: (B, N) midpoint timestamps in seconds.
        rope_type: Only the "split" layout (upstream's default for this
            checkpoint) is implemented; "interleaved" raises.
    """
    import numpy as np

    if rope_type != ROPE_TYPE:
        raise ValueError(f"unsupported rope_type {rope_type!r}: this port implements 'split' only")

    pos = np.array(mid_positions, dtype=np.float32)
    grid = (np.power(10000.0, np.linspace(0.0, 1.0, AUDIO_DIM // 2, dtype=np.float64)) * (np.pi / 2)).astype(np.float32)
    frac = pos / np.float32(ROPE_MAX_POS[0])
    freqs = (frac[..., None] * np.float32(2.0) - np.float32(1.0)) * grid
    cos = np.cos(freqs).reshape(*freqs.shape[:2], AUDIO_HEADS, -1).transpose(0, 2, 1, 3)
    sin = np.sin(freqs).reshape(*freqs.shape[:2], AUDIO_HEADS, -1).transpose(0, 2, 1, 3)
    return mx.array(cos), mx.array(sin), "split"


def _ada(table: mx.array, params: mx.array, sl: slice, dtype) -> tuple[mx.array, mx.array, mx.array]:
    """Split adaptive-norm params into (shift, scale, gate) like upstream get_ada_values."""
    values = table[sl][None, None].astype(dtype) + params[:, :, sl]
    return values[:, :, 0], values[:, :, 1], values[:, :, 2]


def apply_output_head(x: mx.array, table: mx.array, embedded: mx.array) -> mx.array:
    """Audio output head: affine-free LayerNorm then (shift, scale) from a (2, D) table.

    ``table[..., 0]`` is shift and ``table[..., 1]`` is scale; the block AdaLN
    convention is (shift, scale, gate), so this is the one place the two differ.
    """
    ss = table[None, None].astype(x.dtype) + embedded[:, :, None]
    x = mx.fast.layer_norm(x, None, None, NORM_EPS)
    return x * (1 + ss[:, :, 1]) + ss[:, :, 0]


class AudioBlock(nn.Module):
    """One audio-only LTX transformer block."""

    def __init__(self):
        super().__init__()
        self.audio_attn1 = Attention(
            query_dim=AUDIO_DIM, num_heads=AUDIO_HEADS, head_dim=AUDIO_HEAD_DIM, use_rope=True
        )
        self.audio_attn2 = Attention(
            query_dim=AUDIO_DIM,
            kv_dim=AUDIO_CROSS_DIM,
            num_heads=AUDIO_HEADS,
            head_dim=AUDIO_HEAD_DIM,
            use_rope=False,
        )
        self.audio_ff = FeedForward(AUDIO_DIM, dim_out=AUDIO_DIM, mult=FF_MULT, bias=True)
        self.audio_scale_shift_table = mx.zeros((9, AUDIO_DIM))
        self.audio_prompt_scale_shift_table = mx.zeros((2, AUDIO_DIM))

    def __call__(self, x, context, adaln, prompt_ts, rope, mask=None, skip_self_attn=False):
        dtype = x.dtype
        B, N = x.shape[:2]

        shift, scale, gate = _ada(self.audio_scale_shift_table, adaln, slice(0, 3), dtype)
        h = mx.fast.rms_norm(x, None, NORM_EPS) * (1 + scale) + shift

        if skip_self_attn:
            # STG fast path (upstream `all_perturbed`): self-attention degenerates
            # to out = v, so skip q/k/RoPE/softmax and apply gate + to_out only.
            # Upstream ignores attention_mask on this branch; so do we.
            a1 = self.audio_attn1
            v = a1.to_v(h).reshape(B, N, a1.num_heads, a1.head_dim)
            if a1.to_gate_logits is not None:
                v = v * (2.0 * mx.sigmoid(a1.to_gate_logits(h)))[:, :, :, None]
            attn = a1.to_out(v.reshape(B, N, a1.num_heads * a1.head_dim))
        else:
            attn = self.audio_attn1(h, rope_freqs=rope, attention_mask=mask)
        x = x + attn * gate

        # Text cross-attention with cross-attention AdaLN (indices 6..9).
        shift_q, scale_q, gate_q = _ada(self.audio_scale_shift_table, adaln, slice(6, 9), dtype)
        kv = self.audio_prompt_scale_shift_table[None, None].astype(dtype) + prompt_ts
        enc = context * (1 + kv[:, :, 1]) + kv[:, :, 0]
        h = mx.fast.rms_norm(x, None, NORM_EPS) * (1 + scale_q) + shift_q
        x = x + self.audio_attn2(h, encoder_hidden_states=enc) * gate_q

        shift, scale, gate = _ada(self.audio_scale_shift_table, adaln, slice(3, 6), dtype)
        h = mx.fast.rms_norm(x, None, NORM_EPS) * (1 + scale) + shift
        x = x + self.audio_ff(h) * gate
        return x


class AudioOnlyLTX(nn.Module):
    """DramaBox audio-only velocity model."""

    def __init__(self, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.audio_patchify_proj = nn.Linear(128, AUDIO_DIM)
        self.audio_proj_out = nn.Linear(AUDIO_DIM, 128)
        self.audio_scale_shift_table = mx.zeros((2, AUDIO_DIM))
        self.audio_adaln_single = AdaLayerNormSingle(AUDIO_DIM, num_params=9, timestep_dim=TIMESTEP_DIM)
        self.audio_prompt_adaln_single = AdaLayerNormSingle(AUDIO_DIM, num_params=2, timestep_dim=TIMESTEP_DIM)
        self.transformer_blocks = [AudioBlock() for _ in range(num_layers)]
        # RoPE layout, verified against the checkpoint config by load_dit().
        self.rope_type = ROPE_TYPE

    def __call__(self, latent, timesteps, sigma, context, positions, attention_mask=None,
                 skip_blocks: tuple[int, ...] = (), eval_every: int = 8, rope=None):
        """Return the velocity prediction.

        Args:
            latent: (B, N, 128) patchified audio latent.
            timesteps: (B, N, 1) per-token timestep = denoise_mask * sigma.
            sigma: (B,) scalar sigma per batch item.
            context: (B, 1024, 2048) audio text embeddings.
            positions: (B, N) midpoint RoPE positions in seconds.
            attention_mask: (B, N, N) float in [0, 1] or None.
        """
        latent = latent.astype(mx.bfloat16)
        context = context.astype(mx.bfloat16)
        B, N, _ = latent.shape

        x = self.audio_patchify_proj(latent)

        t_flat = (timesteps.reshape(-1).astype(mx.float32) * 1000)
        t_emb = get_timestep_embedding(t_flat, TIMESTEP_DIM).astype(mx.bfloat16)
        adaln, embedded = self.audio_adaln_single(t_emb)
        adaln = adaln.reshape(B, N, 9, AUDIO_DIM)
        embedded = embedded.reshape(B, N, AUDIO_DIM)

        p_emb = get_timestep_embedding(sigma.reshape(-1).astype(mx.float32) * 1000, TIMESTEP_DIM)
        prompt_ts, _ = self.audio_prompt_adaln_single(p_emb.astype(mx.bfloat16))
        prompt_ts = prompt_ts.reshape(B, 1, 2, AUDIO_DIM)

        if rope is None:
            if positions is None:
                raise ValueError("AudioOnlyLTX needs either rope=(cos, sin, rope_type) or positions=(B, N)")
            rope = make_rope(positions, rope_type=self.rope_type)

        mask = None
        if attention_mask is not None:
            info = mx.finfo(mx.bfloat16)
            pos = attention_mask > 0
            # Clamp at the dtype's smallest normal (torch's ``finfo.tiny``; MLX calls
            # it ``smallest_normal``) exactly like the reference mask preparation.
            log_mask = mx.where(pos, mx.log(mx.maximum(attention_mask, info.smallest_normal)), info.min)
            mask = log_mask.astype(mx.bfloat16)[:, None]

        skip = set(skip_blocks)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context, adaln, prompt_ts, rope, mask, skip_self_attn=i in skip)
            if eval_every and (i + 1) % eval_every == 0:
                mx.eval(x)

        return self.audio_proj_out(apply_output_head(x, self.audio_scale_shift_table, embedded))


def remap_dit_key(key: str) -> str | None:
    """Map a DramaBox DiT checkpoint key to our MLX module tree."""
    prefix = "model.diffusion_model."
    if not key.startswith(prefix):
        return None
    key = key[len(prefix):]
    key = key.replace(".linear_1.", ".linear1.").replace(".linear_2.", ".linear2.")
    key = key.replace(".to_out.0.", ".to_out.")
    key = key.replace("audio_ff.net.0.proj.", "audio_ff.proj_in.")
    key = key.replace("audio_ff.net.2.", "audio_ff.proj_out.")
    return key


def _pos_list(value) -> list[float]:
    """Normalize a checkpoint max_pos field ([20], [20.0] or 20) to floats."""
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def _read_dit_config(path: str) -> tuple[dict | None, dict]:
    """Return (transformer config or None, full safetensors metadata).

    Missing metadata is fatal unless ``DRAMABOX_ALLOW_NO_CONFIG=1``: without the
    header config nothing checks rope_type / max_pos / gated attention, and a
    silently re-exported checkpoint would then compute different math.
    """
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata() or {}

    raw = metadata.get("config")
    if raw is None:
        if os.environ.get("DRAMABOX_ALLOW_NO_CONFIG") != "1":
            raise ValueError(
                f"{path}: missing __metadata__['config']; refusing to load an architecture-unverified "
                "checkpoint (set DRAMABOX_ALLOW_NO_CONFIG=1 to override)"
            )
        print(f"[mlx] warning: {path}: no checkpoint config, using built-in defaults", flush=True)
        return None, metadata

    cfg = json.loads(raw).get("transformer")
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: __metadata__['config'] has no 'transformer' section")
    return cfg, metadata


def _validate_dit_config(cfg: dict) -> str:
    """Assert the checkpoint config against this port; return the RoPE layout."""
    missing = [k for k in _DIT_CONFIG_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"checkpoint config is missing required keys: {', '.join(missing)}")
    wrong = [f"{k}={cfg[k]!r} (expected {v!r})" for k, v in _DIT_CONFIG_REQUIRED.items() if cfg[k] != v]
    if wrong:
        raise ValueError("checkpoint config does not match this port: " + "; ".join(wrong))

    for key, expected in _DIT_CONFIG_OPTIONAL.items():
        if key not in cfg:
            print(f"[mlx] warning: checkpoint config has no {key!r}; assuming {expected}", flush=True)
        elif cfg[key] != expected:
            raise ValueError(f"checkpoint config {key}={cfg[key]!r} (expected {expected!r})")

    max_pos = cfg.get("audio_positional_embedding_max_pos")
    if max_pos is None:
        print(f"[mlx] warning: checkpoint config has no 'audio_positional_embedding_max_pos'; "
              f"assuming {ROPE_MAX_POS}", flush=True)
    elif _pos_list(max_pos) != ROPE_MAX_POS:
        raise ValueError(
            f"checkpoint config audio_positional_embedding_max_pos={max_pos!r} "
            f"(expected {ROPE_MAX_POS!r})"
        )

    rope_type = cfg.get("rope_type")
    if rope_type != ROPE_TYPE:
        raise ValueError(f"unsupported rope_type {rope_type!r}: this port implements 'split' only")

    heads, head_dim = cfg["audio_num_attention_heads"], cfg["audio_attention_head_dim"]
    if not heads * head_dim == AUDIO_DIM == AUDIO_HEADS * AUDIO_HEAD_DIM:
        raise ValueError(
            f"audio inner dim mismatch: {heads}*{head_dim}={heads * head_dim}, "
            f"hardcoded {AUDIO_HEADS}*{AUDIO_HEAD_DIM}={AUDIO_HEADS * AUDIO_HEAD_DIM}, AUDIO_DIM={AUDIO_DIM}"
        )
    return rope_type


def _assert_quantized_exactly(model: AudioOnlyLTX, num_layers: int) -> None:
    """Check the P0-4 predicate landed: 10*num_layers quantized, rest bf16."""
    quantized, wrong, missed = [], [], []
    for path, module in model.named_modules():
        allowed = _in_quant_allow(path)
        if isinstance(module, nn.QuantizedLinear):
            quantized.append(path)
            if not allowed:
                wrong.append(path)
        elif allowed and isinstance(module, nn.Linear):
            missed.append(path)
        elif isinstance(module, nn.Linear) and module.weight.dtype != mx.bfloat16:
            wrong.append(f"{path} (dtype {module.weight.dtype})")

    expected = len(_QUANT_ALLOW) * num_layers
    if len(quantized) != expected or wrong or missed:
        raise RuntimeError(
            f"quantization predicate mismatch: {len(quantized)} QuantizedLinear (expected {expected}), "
            f"unexpected={wrong[:4]}, missed={missed[:4]}"
        )


def load_dit(path: str, num_layers: int = NUM_LAYERS, quantize_bits: int | None = None) -> AudioOnlyLTX:
    """Load the DramaBox DiT checkpoint into an MLX audio-only model."""
    cfg, metadata = _read_dit_config(path)
    rope_type = _validate_dit_config(cfg) if cfg is not None else ROPE_TYPE

    model = AudioOnlyLTX(num_layers)
    model.rope_type = rope_type

    raw = mx.load(path)
    weights = {}
    for key, value in raw.items():
        new_key = remap_dit_key(key)
        if new_key is not None:
            weights[new_key] = value.astype(mx.bfloat16)
    unmatched = len(raw) - len(weights)
    del raw

    if len(weights) != NUM_DIT_TENSORS:
        msg = (f"{path}: mapped {len(weights)} audio tensors (expected {NUM_DIT_TENSORS}); "
               f"{unmatched} checkpoint tensors are outside the audio branch")
        # Only the known audio-only export is a hard error; a future/full AV pack
        # is reported and left to load_weights(strict) to validate.
        known = (os.path.basename(path) in _KNOWN_DIT_FILES
                 or metadata.get("model_version") in _KNOWN_DIT_VERSIONS)
        if known:
            raise ValueError(msg)
        print(f"[mlx] warning: {msg}", flush=True)
    elif unmatched:
        print(f"[mlx] note: {path}: {unmatched} checkpoint tensors ignored (not in the audio branch)",
              flush=True)

    model.load_weights(list(weights.items()))
    del weights

    if quantize_bits:
        if quantize_bits not in (4, 8):
            raise ValueError(f"quantize_bits must be 0, 4 or 8 (got {quantize_bits!r})")
        nn.quantize(model, group_size=64, bits=quantize_bits, class_predicate=_quant_pred)
        _assert_quantized_exactly(model, num_layers)

    mx.eval(model.parameters())
    mx.clear_cache()
    return model
