"""Pure/cheap contract tests for the MLX port (no Gemma, no 18 GB pipeline)."""

import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map
from safetensors import safe_open

from mlx_dramabox import audio, model, pipeline

from _helpers import (COMPONENTS_PATH, DIT_PATH, HEAVY, HAVE_MODELS, SKIP_HEAVY,
                      SKIP_NO_MODELS, make_fake_pipeline)


# --------------------------------------------------------------------------- #
# frame snapping / positions / attention mask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("duration,expected", [
    (1.0, 25),
    (5.0, 129),
    (19.98, 505),
    (20.3, 513),
    (20.62, 521),
])
def test_audio_latent_frames(duration, expected):
    frames = pipeline.audio_latent_frames(duration)
    assert frames == expected
    assert (frames - 1) % 8 == 0  # the reachable 8k+1 invariant behind the >514 guard


def test_audio_positions_first_frame_clamp():
    pos = np.array(pipeline.audio_positions(3))
    assert pos.shape == (1, 1, 3, 2)
    assert np.allclose(pos[0, 0, 0], [0.0, 0.01])   # start clamps -0.03 to 0
    assert np.allclose(pos[0, 0, 1], [0.01, 0.05])
    assert np.allclose(pos[0, 0, 2], [0.05, 0.09])
    assert (pos[..., 1] > pos[..., 0]).all()


def test_attention_mask_values():
    mask = np.array(pipeline._attention_mask(2, 3))
    assert mask.shape == (1, 5, 5)
    expected = np.array([
        [1, 1, 1, 1, 1],   # target tokens see target + reference
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],   # reference tokens only see reference
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ], dtype=np.float32)
    assert np.array_equal(mask[0], expected)


# --------------------------------------------------------------------------- #
# sigma schedule / auto rescale
# --------------------------------------------------------------------------- #
def _sigma_reference(steps: int, latent_tokens: int = 128) -> np.ndarray:
    sigmas = np.linspace(1.0, 0.0, steps + 1)
    x1, x2 = 1024.0, 4096.0
    mm = (2.05 - 0.95) / (x2 - x1)
    b = 0.95 - mm * x1
    e = math.exp(latent_tokens * mm + b)
    out = np.where(sigmas != 0, e / (e + (1.0 / np.maximum(sigmas, 1e-12) - 1.0)), 0.0)
    one_minus = 1.0 - out
    stretched = 1.0 - one_minus / (one_minus[steps - 1] / (1.0 - 0.1))
    return np.where(out != 0, stretched, 0.0)


@pytest.mark.parametrize("steps", [2, 3, 30])
def test_sigma_schedule_matches_numpy(steps):
    got = np.array(pipeline.sigma_schedule(steps))
    assert got.shape == (steps + 1,)
    assert np.allclose(got, _sigma_reference(steps), atol=1e-6)
    assert got[0] == pytest.approx(1.0)
    assert got[-1] == 0.0
    assert np.all(np.diff(got) <= 0)
    assert (got >= 0).all() and (got <= 1).all()


@pytest.mark.parametrize("cfg,expected", [
    (1.0, 0.0), (2.0, 0.0), (2.5, 0.3), (3.0, 0.6), (3.5, 0.7),
    (4.0, 0.8), (5.0, 0.8), (8.0, 0.8), (9.0, 0.9), (10.0, 1.0), (14.0, 1.0),
])
def test_auto_rescale_boundaries(cfg, expected):
    assert pipeline.auto_rescale(cfg) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# crossfade
# --------------------------------------------------------------------------- #
def test_crossfade_length_start_end_and_finite():
    sr = 48000
    prev = mx.ones((1, 2, sr // 10))
    nxt = mx.zeros((1, 2, sr // 10))
    out = np.array(pipeline._crossfade(prev, nxt, sr, 50.0))
    fade = int(round(0.05 * sr))
    assert out.shape == (1, 2, 2 * (sr // 10) - fade)
    assert np.isfinite(out).all()
    assert out[0, 0, 0] == pytest.approx(1.0)
    assert out[0, 0, -1] == pytest.approx(0.0)
    assert np.all(np.diff(out[0, 0, -fade:]) <= 1e-6)


def test_crossfade_fallback_when_fade_is_one_sample():
    prev = mx.ones((1, 2, 500))
    nxt = mx.zeros((1, 2, 500))
    out = np.array(pipeline._crossfade(prev, nxt, 48000, 0.0))
    assert out.shape == (1, 2, 1000)
    assert np.array_equal(out[0, 0, :500], np.ones(500))
    assert np.array_equal(out[0, 0, 500:], np.zeros(500))


def test_crossfade_channel_alignment_and_mismatch():
    mono = mx.full((1, 1, 1000), 0.5)
    stereo = mx.full((1, 2, 1000), 0.25)
    fade = int(round(0.001 * 48000))
    out = np.array(pipeline._crossfade(mono, stereo, 48000, 1.0))
    assert out.shape == (1, 2, 2000 - fade)
    assert np.isfinite(out).all()
    rev = np.array(pipeline._crossfade(stereo, mono, 48000, 1.0))
    assert rev.shape == (1, 2, 2000 - fade)
    assert np.isfinite(rev).all()
    with pytest.raises(ValueError, match="mismatched channel counts"):
        pipeline._crossfade(mx.zeros((1, 2, 100)), mx.zeros((1, 3, 100)), 48000)


# --------------------------------------------------------------------------- #
# >514 silence-guard semantics, exercised through the real generate() tail
# --------------------------------------------------------------------------- #
def _run_generate_tail(monkeypatch, tframes: int) -> np.ndarray:
    import mlx_dramabox.pipeline as P

    captured = {}
    real_unpatchify = P.unpatchify

    def spy(tokens):
        captured["latent"] = np.array(tokens.astype(mx.float32))
        return real_unpatchify(tokens)

    monkeypatch.setattr(P, "unpatchify", spy)
    monkeypatch.setattr(P, "audio_latent_frames", lambda _duration: tframes)
    monkeypatch.setattr(P, "decode_latent", lambda *a, **k: mx.zeros((1, 2, 8)))
    ctx = mx.zeros((1, 1, model.AUDIO_DIM), dtype=mx.bfloat16)
    make_fake_pipeline().generate(
        "hi", cfg_scale=1.0, stg_scale=0.0, steps=2, gen_duration=1.0,
        precomputed=(ctx, None), verbose=False,
    )
    return captured["latent"][0]


def test_silence_guard_repairs_above_514(monkeypatch):
    latent = _run_generate_tail(monkeypatch, 521)
    assert latent.shape == (521, 128)
    for frame, t in ((512, 1.0 / 3.0), (513, 2.0 / 3.0)):
        expected = (1.0 - t) * latent[511] + t * latent[514]
        # bf16 storage of the float32 interpolation.
        assert np.allclose(latent[frame], expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("tframes,end", [(513, 512), (514, 513)])
def test_silence_guard_is_noop_at_or_below_514(monkeypatch, tframes, end):
    latent = _run_generate_tail(monkeypatch, tframes)
    assert latent.shape == (tframes, 128)
    # A random latent does not satisfy the repair identity unless frames were touched.
    interpolated = (2.0 / 3.0) * latent[511] + (1.0 / 3.0) * latent[end]
    assert not np.allclose(latent[512], interpolated, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
# prepare_reference / load_waveform / decode_latent
# --------------------------------------------------------------------------- #
def _write_wav(path, samples: np.ndarray, sr: int):
    import soundfile as sf

    sf.write(str(path), samples, sr, subtype="FLOAT")
    return str(path)


@pytest.mark.parametrize("channels,sr", [(1, 44100), (2, 44100), (2, 48000), (4, 48000)])
def test_prepare_reference_channels_and_rates(tmp_path, channels, sr):
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal((sr // 2, channels)) * 0.1).astype(np.float32)
    path = _write_wav(tmp_path / "ref.wav", samples, sr)
    out = np.array(audio.prepare_reference(path, 0.5))
    assert out.shape == (1, 2, int(0.5 * audio.SAMPLE_RATE))
    assert np.isfinite(out).all()
    if channels == 1:
        assert np.array_equal(out[0, 0], out[0, 1])   # mono duplicated, not downmixed
    else:
        assert not np.allclose(out[0, 0], out[0, 1])


def test_prepare_reference_4ch_uses_first_two_channels(tmp_path):
    rng = np.random.default_rng(1)
    base = (rng.standard_normal((24000, 2)) * 0.1).astype(np.float32)
    extra = (rng.standard_normal((24000, 2)) * 0.1).astype(np.float32)
    out2 = np.array(audio.prepare_reference(_write_wav(tmp_path / "2.wav", base, 48000), 0.5))
    out4 = np.array(audio.prepare_reference(
        _write_wav(tmp_path / "4.wav", np.concatenate([base, extra], axis=1), 48000), 0.5))
    assert np.array_equal(out2, out4)


def test_prepare_reference_tiles_short_clip(tmp_path):
    rng = np.random.default_rng(2)
    # 0.05 s @ 16 kHz source: tiled up, no resampling, so normalisation is exact.
    short = (rng.standard_normal((800, 1)) * 0.1).astype(np.float32)
    out = np.array(audio.prepare_reference(_write_wav(tmp_path / "short.wav", short, audio.SAMPLE_RATE), 1.0))
    assert out.shape == (1, 2, 16000)
    assert np.isfinite(out).all()
    assert np.abs(out).max() == pytest.approx(10 ** (-4.0 / 20), abs=1e-4)


def test_prepare_reference_silent_is_finite_zero(tmp_path):
    silent = np.zeros((24000, 1), dtype=np.float32)
    out = np.array(audio.prepare_reference(_write_wav(tmp_path / "silent.wav", silent, 48000), 0.5))
    assert out.shape == (1, 2, 8000)
    assert np.isfinite(out).all()
    assert np.all(out == 0)


def test_prepare_reference_empty_raises(tmp_path):
    path = _write_wav(tmp_path / "empty.wav", np.zeros((0, 1), dtype=np.float32), 48000)
    with pytest.raises(ValueError, match="no samples"):
        audio.prepare_reference(path, 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_prepare_reference_rejects_non_finite(tmp_path, bad):
    samples = np.zeros((4800, 1), dtype=np.float32)
    samples[10, 0] = bad
    path = _write_wav(tmp_path / "bad.wav", samples, 48000)
    with pytest.raises(ValueError, match="non-finite"):
        audio.prepare_reference(path, 1.0)


@pytest.mark.parametrize("duration,match", [
    (0.0, "finite and > 0"),
    (-1.0, "finite and > 0"),
    (float("nan"), "finite and > 0"),
    (float("inf"), "finite and > 0"),
    (1e9, "exceeds"),
])
def test_prepare_reference_rejects_bad_duration(tmp_path, duration, match):
    path = _write_wav(tmp_path / "ok.wav", np.zeros((48000, 1), dtype=np.float32), 48000)
    with pytest.raises(ValueError, match=match):
        audio.prepare_reference(path, duration)


def test_prepare_reference_rejects_too_short_target(tmp_path):
    # 0.001 s @ 48 kHz -> 16 samples after resampling, below the STFT window.
    path = _write_wav(tmp_path / "tiny.wav", np.zeros((48000, 1), dtype=np.float32), 48000)
    with pytest.raises(ValueError, match="too short"):
        audio.prepare_reference(path, 0.001)


def test_load_waveform_bounded_read(tmp_path, monkeypatch):
    import soundfile as sf

    sr = 8000
    long_samples = np.zeros((sr * 10, 1), dtype=np.float32)
    path = _write_wav(tmp_path / "long.wav", long_samples, sr)

    calls = {}
    real_read = sf.read

    def spy(*args, **kwargs):
        calls.update(kwargs)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(sf, "read", spy)
    wav, got_sr = audio.load_waveform(path, max_duration=1.0)
    assert got_sr == sr
    assert "frames" in calls
    # window (8000) + max(0.1 s, 1%) margin, and definitely not the whole file.
    assert 0 < calls["frames"] <= 8800
    assert calls["frames"] < len(long_samples)
    assert wav.shape == (1, 8000)
    assert np.isfinite(wav).all()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_load_waveform_rejects_bad_max_duration(bad):
    with pytest.raises(ValueError, match="max_duration"):
        audio.load_waveform("unused.wav", max_duration=bad)


def test_decode_latent_rejects_bad_shapes():
    with pytest.raises(ValueError, match="no frames"):
        audio.decode_latent(None, None, mx.zeros((1, 8, 0, 16), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="latent must be"):
        audio.decode_latent(None, None, mx.zeros((1, 7, 2, 16), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="latent must be"):
        audio.decode_latent(None, None, mx.zeros((1, 8, 2, 15), dtype=mx.bfloat16))


@pytest.mark.skipif(not (HAVE_MODELS and HEAVY), reason=SKIP_HEAVY)
def test_codec_heavy_contracts():
    """T=1 decodes to a finite waveform and the output rate matches the BWE factor."""
    _, decoder, vocoder = audio.load_audio_codec(str(COMPONENTS_PATH))
    assert audio.OUTPUT_SAMPLE_RATE == audio.SAMPLE_RATE * vocoder._resampler.upsample_factor
    wav = np.array(audio.decode_latent(
        decoder, vocoder, mx.zeros((1, 8, 1, 16), dtype=mx.bfloat16)).astype(mx.float32))
    assert wav.shape[0] == 1 and wav.shape[1] == 2
    assert np.isfinite(wav).all()


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def _rope_reference(mid_positions: np.ndarray, theta=10000.0, max_pos=20.0,
                    heads=32, dim=2048):
    """Straight numpy port of precompute_freqs_cis for the 1-D split grid."""
    pos = np.asarray(mid_positions, dtype=np.float32)
    grid = (np.power(theta, np.linspace(0.0, 1.0, dim // 2, dtype=np.float64))
            * (np.pi / 2)).astype(np.float32)
    freqs = (pos[..., None] / np.float32(max_pos) * np.float32(2.0)
             - np.float32(1.0)) * grid
    cos = np.cos(freqs).reshape(*freqs.shape[:2], heads, -1).transpose(0, 2, 1, 3)
    sin = np.sin(freqs).reshape(*freqs.shape[:2], heads, -1).transpose(0, 2, 1, 3)
    return cos, sin


def test_make_rope_matches_numpy_split():
    positions = pipeline.audio_positions(9)
    mid = ((positions[..., 0] + positions[..., 1]) / 2.0)[:, 0]
    cos, sin, rope_type = model.make_rope(mid)
    exp_cos, exp_sin = _rope_reference(np.array(mid))
    assert rope_type == "split"
    assert tuple(cos.shape) == (1, 32, 9, 32)
    assert np.allclose(np.array(cos), exp_cos, atol=1e-6)
    assert np.allclose(np.array(sin), exp_sin, atol=1e-6)
    assert np.allclose(np.array(cos) ** 2 + np.array(sin) ** 2, 1.0, atol=1e-5)


def test_make_rope_rejects_interleaved():
    with pytest.raises(ValueError, match="interleaved"):
        model.make_rope(mx.zeros((1, 9), dtype=mx.float32), rope_type="interleaved")


# --------------------------------------------------------------------------- #
# quantization predicate
# --------------------------------------------------------------------------- #
def _bf16(module):
    module.update(tree_map(lambda a: a.astype(mx.bfloat16), module.parameters()))
    return module


def test_quant_pred_quantizes_exactly_ten_per_block():
    net = _bf16(model.AudioOnlyLTX(num_layers=1))
    nn.quantize(net, group_size=64, bits=8, class_predicate=model._quant_pred)

    quantized = {path for path, mod in net.named_modules()
                 if isinstance(mod, nn.QuantizedLinear)}
    expected = {f"transformer_blocks.0.{name}" for name in model._QUANT_ALLOW}
    assert quantized == expected
    assert len(quantized) == 10

    excluded = ["audio_proj_out", "audio_patchify_proj",
                "transformer_blocks.0.audio_attn1.to_gate_logits",
                "transformer_blocks.0.audio_attn2.to_gate_logits",
                "audio_adaln_single.linear", "audio_prompt_adaln_single.linear"]
    modules = dict(net.named_modules())
    for path in excluded:
        assert isinstance(modules[path], nn.Linear)
        assert modules[path].weight.dtype == mx.bfloat16
    assert not model._quant_pred("audio_proj_out", modules["audio_proj_out"])
    assert not model._quant_pred("transformer_blocks.0.audio_attn1.to_q", object())


# --------------------------------------------------------------------------- #
# STG fast path is bit-identical to perturbation_mask=0
# --------------------------------------------------------------------------- #
class _ZeroPerturbation:
    """Delegates to Attention with perturbation_mask=0, exposing the fast-path attrs."""

    def __init__(self, attention):
        self._attention = attention

    def __getattr__(self, name):
        return getattr(self._attention, name)

    def __call__(self, h, rope_freqs=None, attention_mask=None):
        return self._attention(h, rope_freqs=rope_freqs, attention_mask=attention_mask,
                               perturbation_mask=mx.array(0.0, dtype=h.dtype))


def _stg_case(mask_kind: str):
    b, n = 1, 9
    block = _bf16(model.AudioBlock())
    x = mx.random.normal((b, n, model.AUDIO_DIM)).astype(mx.bfloat16)
    context = mx.random.normal((b, 7, model.AUDIO_DIM)).astype(mx.bfloat16)
    adaln = (mx.random.normal((b, n, 9, model.AUDIO_DIM)) * 0.1).astype(mx.bfloat16)
    prompt_ts = (mx.random.normal((b, 1, 2, model.AUDIO_DIM)) * 0.1).astype(mx.bfloat16)
    rope = model.make_rope(mx.arange(n, dtype=mx.float32)[None])

    mask = None
    if mask_kind != "none":
        mask = mx.zeros((b, 1, n, n), dtype=mx.bfloat16)
        for i in range(n):
            mask[:, :, i, :i + 1] = 1.0
        if mask_kind == "asymmetric":
            mask[:, :, 0, 2:] = 0.0  # token 0 loses its causal window

    block.audio_attn1 = _ZeroPerturbation(block.audio_attn1)
    fast = block(x, context, adaln, prompt_ts, rope, mask, skip_self_attn=True)
    reference = block(x, context, adaln, prompt_ts, rope, mask, skip_self_attn=False)
    mx.eval(fast, reference)
    return fast, reference


@pytest.mark.parametrize("mask_kind", ["none", "symmetric", "asymmetric"])
def test_stg_fast_path_exact_equality(mask_kind):
    fast, reference = _stg_case(mask_kind)
    assert mx.array_equal(fast, reference)
    assert float(mx.max(mx.abs(fast.astype(mx.float32)
                               - reference.astype(mx.float32)))) == 0.0


# --------------------------------------------------------------------------- #
# AdaLN slice layout + output-head shift/scale convention
# --------------------------------------------------------------------------- #
def test_ada_and_output_head_layout():
    rng = np.random.default_rng(7)
    d, b, n = 8, 1, 3
    table = mx.array(rng.standard_normal((9, d)).astype(np.float32))
    params = mx.array((rng.standard_normal((b, n, 9, d)) * 0.1).astype(np.float32))

    # The block convention is (shift, scale, gate) for slices (0,3), (3,6), (6,9).
    for sl, offset in ((slice(0, 3), 0), (slice(3, 6), 3), (slice(6, 9), 6)):
        shift, scale, gate = model._ada(table, params, sl, mx.float32)
        values = np.array(table)[None, None, sl] + np.array(params)[:, :, sl]
        assert np.allclose(np.array(shift), values[:, :, 0], atol=1e-6)
        assert np.allclose(np.array(scale), values[:, :, 1], atol=1e-6)
        assert np.allclose(np.array(gate), values[:, :, 2], atol=1e-6)
        assert np.array(shift).shape == (b, n, d)
        # Slices are the per-segment offsets, not the literal table rows.
        assert np.allclose(np.array(shift), np.array(table)[offset] + np.array(params)[:, :, offset],
                           atol=1e-6)

    # The output head is (shift, scale) from the same 2-row table: ss[...,0]=shift,
    # ss[...,1]=scale, applied as x*(1+scale)+shift after affine-free LayerNorm.
    x = mx.array(rng.standard_normal((b, n, d)).astype(np.float32))
    table2 = mx.array(rng.standard_normal((2, d)).astype(np.float32))
    embedded = mx.array((rng.standard_normal((b, n, d)) * 0.1).astype(np.float32))
    got = np.array(model.apply_output_head(x, table2, embedded))
    normed = np.array(mx.fast.layer_norm(x, None, None, model.NORM_EPS))
    ss = np.array(table2)[None, None] + np.array(embedded)[:, :, None]
    expected = normed * (1 + ss[:, :, 1]) + ss[:, :, 0]
    assert np.allclose(got, expected, atol=1e-5)
    # A swap would be caught: shift-first vs scale-first differ.
    swapped = normed * (1 + ss[:, :, 0]) + ss[:, :, 1]
    assert not np.allclose(got, swapped, atol=1e-4)


# --------------------------------------------------------------------------- #
# checkpoint config assertions
# --------------------------------------------------------------------------- #
def _checkpoint_config(path):
    with safe_open(str(path), framework="numpy") as f:
        metadata = f.metadata() or {}
    return json.loads(metadata["config"])["transformer"]


@pytest.mark.skipif(not DIT_PATH.is_file(), reason=SKIP_NO_MODELS)
def test_real_dit_config_validates():
    cfg = _checkpoint_config(DIT_PATH)
    assert model._validate_dit_config(cfg) == "split"
    assert cfg["audio_num_attention_heads"] * cfg["audio_attention_head_dim"] == model.AUDIO_DIM


@pytest.mark.skipif(not DIT_PATH.is_file(), reason=SKIP_NO_MODELS)
@pytest.mark.parametrize("max_pos", [[20], [20.0], 20])
def test_dit_config_accepts_max_pos_forms(max_pos):
    cfg = dict(_checkpoint_config(DIT_PATH))
    cfg["audio_positional_embedding_max_pos"] = max_pos
    assert model._validate_dit_config(cfg) == "split"


@pytest.mark.skipif(not DIT_PATH.is_file(), reason=SKIP_NO_MODELS)
@pytest.mark.parametrize("key,value", [
    ("num_layers", 47),
    ("audio_num_attention_heads", 16),
    ("audio_attention_head_dim", 128),
    ("use_middle_indices_grid", False),
    ("frequencies_precision", "float32"),
    ("positional_embedding_theta", 5000.0),
    ("apply_gated_attention", False),
    ("cross_attention_adaln", False),
    ("text_encoder_norm_type", "layer_norm"),
    ("rope_type", "interleaved"),
    ("audio_positional_embedding_max_pos", [32]),
    ("in_channels", 64),
    ("audio_out_channels", 64),
])
def test_dit_config_mismatch_raises(key, value):
    cfg = dict(_checkpoint_config(DIT_PATH))
    cfg[key] = value
    with pytest.raises(ValueError):
        model._validate_dit_config(cfg)


@pytest.mark.skipif(not DIT_PATH.is_file(), reason=SKIP_NO_MODELS)
def test_dit_config_missing_required_key_raises():
    cfg = dict(_checkpoint_config(DIT_PATH))
    cfg.pop("norm_eps")
    with pytest.raises(ValueError, match="missing required keys"):
        model._validate_dit_config(cfg)


@pytest.mark.skipif(not COMPONENTS_PATH.is_file(), reason=SKIP_NO_MODELS)
def test_real_connector_config_validates():
    from mlx_dramabox import text

    text._validate_connector_config(_checkpoint_config(COMPONENTS_PATH))


@pytest.mark.skipif(not COMPONENTS_PATH.is_file(), reason=SKIP_NO_MODELS)
def test_connector_config_accepts_scalar_max_pos():
    from mlx_dramabox import text

    cfg = dict(_checkpoint_config(COMPONENTS_PATH))
    cfg["connector_positional_embedding_max_pos"] = 4096
    text._validate_connector_config(cfg)


@pytest.mark.skipif(not COMPONENTS_PATH.is_file(), reason=SKIP_NO_MODELS)
@pytest.mark.parametrize("key,value", [
    ("connector_num_layers", 4),
    ("audio_connector_num_attention_heads", 16),
    ("audio_connector_attention_head_dim", 128),
    ("connector_apply_gated_attention", False),
    ("connector_norm_output", False),
    ("connector_positional_embedding_max_pos", [2048]),
])
def test_connector_config_mismatch_raises(key, value):
    from mlx_dramabox import text

    cfg = dict(_checkpoint_config(COMPONENTS_PATH))
    cfg[key] = value
    with pytest.raises(ValueError):
        text._validate_connector_config(cfg)


def test_read_dit_config_missing_metadata(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    path = tmp_path / "nometa.safetensors"
    save_file({"x": np.zeros(1, dtype=np.float32)}, str(path))
    monkeypatch.delenv("DRAMABOX_ALLOW_NO_CONFIG", raising=False)
    with pytest.raises(ValueError, match="missing __metadata__"):
        model._read_dit_config(str(path))
    monkeypatch.setenv("DRAMABOX_ALLOW_NO_CONFIG", "1")
    cfg, _ = model._read_dit_config(str(path))
    assert cfg is None


def test_remap_dit_key():
    assert model.remap_dit_key(
        "model.diffusion_model.audio_ff.net.0.proj.weight") == "audio_ff.proj_in.weight"
    assert model.remap_dit_key(
        "model.diffusion_model.transformer_blocks.0.audio_attn1.to_out.0.weight"
    ) == "transformer_blocks.0.audio_attn1.to_out.weight"
    assert model.remap_dit_key("vae.decoder.conv_in.weight") is None
