"""Streamed audio-text connector vs the inline eager reference (R2b acceptance).

R2b/R2c baseline (real 49x(1,1024,3840) Gemma states, shipped
CONNECTOR_CHUNK_TOKENS=512; streamed is the shipped path, eager is the fallback):

    storyteller fixture: cos(streamed, eager) = 0.99995330, rel-L2 = 9.66e-3
    DEFAULT_NEG:       cos(streamed, eager) = 0.99996726, rel-L2 = 8.09e-3
    peak 3.25 -> 2.74 GB, 0.553 -> 0.454 s at T=1024

    vs PyTorch, this environment's hand-built torch-CPU bf16 helper:
      storyteller: streamed 0.99804022 vs eager 0.99802270 (+1.8e-5)
      DEFAULT_NEG: streamed 0.99822307 vs eager 0.99814666 (+7.6e-5)
    The absolute helper value is kernel/precision dependent; the relative R2b
    clause passes for both prompts.  On the "hello" prompt against the proper
    PyTorch connector reference, the measured pair is streamed 0.99997956 vs
    eager 0.99997973 (within the -1e-6 gate).

R2c is valid only under R2b: the retired absolute `cos >= 0.999999` bar does
not apply to the 512-token streamed path. If a future fixture fails the 0.9999
floor, increase the chunk toward 1024; do not relax the floor.

Acceptance is cos(streamed, eager) >= 0.9999, rel-L2 <= 2e-2, and
cos(streamed, PyTorch) >= cos(eager, PyTorch) - 1e-6.  Do not gate on the
waveform: the pre-fix 4 s golden differs at cos 0.963 after 30 Euler steps
(expected chaotic amplification of a 2e-5 context perturbation) and must be
regenerated; the exact Whisper transcript is the end-to-end gate.
"""

import math
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlx_dramabox.model import AUDIO_DIM
from mlx_dramabox.text import (CONNECTOR_CHUNK_TOKENS, GEMMA_HIDDEN,
                               NUM_HIDDEN_STATES, AudioTextConnector,
                               load_connector)

from _helpers import (COMPONENTS_PATH, GEMMA_DIR, HAVE_GEMMA, HAVE_MODELS,
                      HEAVY, ROOT, SKIP_HEAVY, SKIP_NO_MODELS, cosine, rel_l2)


def _eager_connector(connector: AudioTextConnector, states, attention_mask):
    """The pre-P0-5 eager path: one (T, 3840, 49) stack for the whole sequence."""
    encoded = mx.stack(states, axis=-1)
    variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
    normed = encoded * mx.rsqrt(variance + 1e-6)
    b, seq, dim, layers = normed.shape
    normed = normed.reshape(b, seq, dim * layers)
    normed = normed * attention_mask[:, :, None].astype(normed.dtype)
    feat = connector.audio_aggregate_embed(normed * math.sqrt(AUDIO_DIM / GEMMA_HIDDEN))
    return connector.audio_embeddings_connector(feat, attention_mask=attention_mask)


def _features(connector, states, attention_mask):
    streamed = np.array(connector(states, attention_mask).astype(mx.float32))
    eager = np.array(_eager_connector(connector, states, attention_mask).astype(mx.float32))
    return streamed, eager


def _fake_states(seq_len: int, seed: int = 0):
    # Left-padded like the Gemma tokenizer: zeros first, valid tokens last.
    rng = np.random.default_rng(seed)
    states = [mx.array(rng.standard_normal((1, seq_len, GEMMA_HIDDEN)), dtype=mx.float32)
              .astype(mx.bfloat16) for _ in range(NUM_HIDDEN_STATES)]
    mask = mx.ones((1, seq_len), dtype=mx.float32)
    mask[:, :16] = 0.0
    return states, mask


@pytest.mark.skipif(not HAVE_MODELS, reason=SKIP_NO_MODELS)
def test_connector_streams_in_chunks(monkeypatch):
    assert CONNECTOR_CHUNK_TOKENS == 512
    import mlx_dramabox.text as text_mod
    monkeypatch.setattr(text_mod, "CONNECTOR_CHUNK_TOKENS", 64)
    connector = load_connector(str(COMPONENTS_PATH))
    states, mask = _fake_states(128)  # two chunks at the monkeypatched size 64
    streamed, eager = _features(connector, states, mask)

    assert streamed.shape == eager.shape == (1, 128, AUDIO_DIM)
    assert cosine(streamed, eager) >= 0.9999
    assert rel_l2(streamed, eager) <= 2e-2

    # Left-pad positions are register-filled by the connector, never zeroed.
    pads = slice(0, 16)
    assert np.abs(streamed[:, pads] - eager[:, pads]).max() <= 1e-2
    assert np.linalg.norm(streamed[:, pads]) > 0
    assert np.isfinite(streamed).all()


def _pt_models():
    """Build the PyTorch connector reference once; ImportError means unavailable."""
    if "models" in _PT_CACHE:
        return _PT_CACHE["models"]
    sys.path.insert(0, str(ROOT / "ltx2"))
    import torch
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.text_encoders.gemma.embeddings_connector import (
        Embeddings1DConnector as PTConnector,
    )
    from safetensors import safe_open

    from mlx_dramabox.text import remap_text_key

    state = {}
    with safe_open(str(COMPONENTS_PATH), framework="pt") as handle:
        for key in handle.keys():
            mapped = remap_text_key(key)
            if mapped is not None:
                state[mapped] = handle.get_tensor(key)

    aggregate = torch.nn.Linear(NUM_HIDDEN_STATES * GEMMA_HIDDEN, AUDIO_DIM,
                                bias=True, dtype=torch.bfloat16)
    connector = PTConnector(
        attention_head_dim=64,
        num_attention_heads=32,
        num_layers=8,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        num_learnable_registers=128,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        apply_gated_attention=True,
    ).to(torch.bfloat16)

    aggregate.load_state_dict({
        key.removeprefix("audio_aggregate_embed."): value
        for key, value in state.items() if key.startswith("audio_aggregate_embed.")
    })
    connector.load_state_dict({
        key.removeprefix("audio_embeddings_connector."): value
        for key, value in state.items() if key.startswith("audio_embeddings_connector.")
    })
    aggregate.eval()
    connector.eval()
    _PT_CACHE["models"] = (torch, aggregate, connector)
    return _PT_CACHE["models"]


_PT_CACHE = {}


def _pt_reference(states, attention_mask):
    torch, aggregate, connector = _pt_models()
    # States are bf16 in the reference; keep the per-token RMS in bf16 too.
    feats = torch.stack(
        [torch.from_numpy(np.array(h.astype(mx.float32))).to(torch.bfloat16)
         for h in states], dim=-1)
    variance = feats.pow(2).mean(dim=2, keepdim=True)
    normed = feats * torch.rsqrt(variance + 1e-6)
    b, seq, dim, layers = normed.shape
    normed = normed.reshape(b, seq, dim * layers)
    # Pre-projection masking is binary (1 valid, 0 pad), exactly like MLX.
    binary = torch.from_numpy(np.array(attention_mask)).to(torch.bfloat16)
    normed = normed * binary[:, :, None]
    feat = aggregate((normed * math.sqrt(AUDIO_DIM / GEMMA_HIDDEN)).to(torch.bfloat16))
    # The reference connector expects an additive mask: 0 valid, -max padding.
    additive = ((binary.to(torch.int64) - 1).to(torch.bfloat16).reshape(1, 1, 1, -1)
                * torch.finfo(torch.bfloat16).max)
    with torch.no_grad():
        out, _ = connector(feat, attention_mask=additive)
    return out.float().detach().cpu().numpy()


@pytest.mark.skipif(not (HAVE_GEMMA and HEAVY), reason=SKIP_HEAVY)
def test_gemma_hidden_state_layout():
    """The 49th state must be norm(raw layer-47 output), not the raw output.

    This catches the 49-vs-50-tuple drift class without loading PyTorch's 12B
    model: capture the input to the final norm during the normal extraction and
    compare the last state against norm(input) and input independently.
    """
    from mlx_dramabox.text import GemmaTextEncoder

    encoder = GemmaTextEncoder(str(GEMMA_DIR))
    ids, mask = encoder.tokenize(
        "A low, raspy storyteller says, \"Listen closely.\"")
    orig_norm = encoder.inner.norm
    captured = {}

    def wrapped(x):
        captured["input"] = x
        return orig_norm(x)

    encoder.inner.norm = wrapped
    try:
        states = encoder.hidden_states(ids, mask)
    finally:
        encoder.inner.norm = orig_norm

    assert len(states) == NUM_HIDDEN_STATES == 49
    raw = captured["input"]
    assert states[-1].shape == raw.shape
    assert mx.array_equal(states[-1], orig_norm(raw).astype(states[-1].dtype))
    assert not mx.array_equal(states[-1], raw.astype(states[-1].dtype))
    assert all(s.shape == (1, 1024, GEMMA_HIDDEN) for s in states)


@pytest.mark.skipif(not (HAVE_GEMMA and HEAVY), reason=SKIP_HEAVY)
def test_streamed_connector_with_real_gemma_states_t1024():
    """R2b heavy regression: real 49x(1,1024,3840) Gemma states, one fixed prompt.

    Recorded baseline (see module docstring): storyteller fixture cos(streamed, eager)
    0.99995330 / rel-L2 9.66e-3; DEFAULT_NEG 0.99996726 / 8.09e-3; streamed not
    worse than eager vs PyTorch; peak 3.25 -> 2.74 GB, 0.553 -> 0.454 s.  Both
    prompts are asserted (the failure mode is state-magnitude dependent).
    """
    from mlx_dramabox.pipeline import DEFAULT_NEG
    from mlx_dramabox.text import GemmaTextEncoder

    encoder = GemmaTextEncoder(str(GEMMA_DIR))
    connector = load_connector(str(COMPONENTS_PATH))
    fixed_prompt = ("A low, raspy storyteller says, "
                    "\"Listen closely, and I will tell you how the city forgot the sea.\"")

    try:
        pt_available = _pt_models() is not None
    except ImportError:
        pt_available = False

    for prompt in (fixed_prompt, DEFAULT_NEG):
        ids, mask = encoder.tokenize(prompt)
        states = encoder.hidden_states(ids, mask)
        assert len(states) == NUM_HIDDEN_STATES
        assert states[0].shape == (1, 1024, GEMMA_HIDDEN)

        streamed, eager = _features(connector, states, mask)
        eager_vs_streamed = cosine(streamed, eager)
        print(f"[connector] prompt={prompt[:32]!r} cos(streamed,eager)={eager_vs_streamed:.8f} "
              f"rel-L2={rel_l2(streamed, eager):.2e}")
        assert eager_vs_streamed >= 0.9999
        assert rel_l2(streamed, eager) <= 2e-2

        if pt_available:
            pt = _pt_reference(states, mask)
            cos_streamed_pt = cosine(streamed, pt)
            cos_eager_pt = cosine(eager, pt)
            print(f"[connector] prompt={prompt[:32]!r} cos(streamed,PT)={cos_streamed_pt:.8f} "
                  f"cos(eager,PT)={cos_eager_pt:.8f}")
            assert cos_streamed_pt >= cos_eager_pt - 1e-6
