"""Validation, CLI-order and long-form guard tests (no model weights loaded)."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_dramabox.pipeline as pipeline
from mlx_dramabox import DramaBoxMLX

from _helpers import ROOT, make_fake_pipeline

CLI = ROOT / "src" / "mlx_tts.py"


# --------------------------------------------------------------------------- #
# validate_generation_params
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs,match", [
    ({"cfg_scale": 0.5}, "cfg_scale"),
    ({"cfg_scale": 0.0}, "cfg_scale"),
    ({"cfg_scale": float("nan")}, "finite"),
    ({"cfg_scale": float("inf")}, "finite"),
    ({"stg_scale": float("nan")}, "stg_scale"),
    ({"stg_scale": float("inf")}, "stg_scale"),
    ({"steps": 1}, "steps"),
    ({"steps": 201}, "steps"),
    ({"steps": 2.5}, "steps"),
    ({"steps": "many"}, "steps"),
    ({"ref_duration": 0.0}, "ref_duration"),
    ({"ref_duration": -1.0}, "ref_duration"),
    ({"ref_duration": float("nan")}, "ref_duration"),
    ({"ref_duration": float("inf")}, "ref_duration"),
    ({"ref_duration": 61.0}, "ref_duration"),
    ({"gen_duration": -1.0}, "duration"),
    ({"gen_duration": float("nan")}, "duration"),
    ({"gen_duration": float("inf")}, "duration"),
    ({"gen_duration": 601.0}, "duration"),
    ({"max_chunk_duration": 0.0}, "max_chunk_duration"),
    ({"max_chunk_duration": -5.0}, "max_chunk_duration"),
    ({"max_chunk_duration": float("nan")}, "max_chunk_duration"),
    ({"max_chunk_duration": 10.0, "target_chunk_duration": 20.0}, "target_chunk_duration"),
    ({"target_chunk_duration": 0.0}, "target_chunk_duration"),
    ({"target_chunk_duration": float("inf")}, "target_chunk_duration"),
    ({"duration_multiplier": 0.0}, "duration_multiplier"),
    ({"duration_multiplier": -1.0}, "duration_multiplier"),
    ({"duration_multiplier": float("nan")}, "duration_multiplier"),
    ({"crossfade_ms": -1.0}, "crossfade_ms"),
    ({"crossfade_ms": float("nan")}, "crossfade_ms"),
    ({"rescale_scale": "bogus"}, "rescale"),
    ({"rescale_scale": 1.5}, "rescale"),
    ({"rescale_scale": -0.1}, "rescale"),
    ({"rescale_scale": float("nan")}, "rescale"),
])
def test_validate_generation_params_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pipeline.validate_generation_params(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {},
    {"cfg_scale": 1.0},
    {"cfg_scale": 2.5, "stg_scale": -1.5},          # any finite STG sign is allowed
    {"steps": 2},
    {"steps": 200},
    {"max_chunk_duration": float("inf")},            # never chunk
    {"gen_duration": 0.0},
    {"gen_duration": 600.0},
    {"rescale_scale": "auto"},
    {"rescale_scale": "0.5"},
    {"rescale_scale": 0},
    {"rescale_scale": 1.0},
    {"crossfade_ms": 0.0},
])
def test_validate_generation_params_accepts(kwargs):
    pipeline.validate_generation_params(**kwargs)


# --------------------------------------------------------------------------- #
# validate_duration_policy
# --------------------------------------------------------------------------- #
def test_duration_policy_at_and_below_max_allowed():
    pipeline.validate_duration_policy("hello world", 10.0, 45.0, 1.1)
    pipeline.validate_duration_policy("hello world", 45.0, 45.0, 1.1)  # == max
    pipeline.validate_duration_policy("hello world", 45.5, float("inf"), 1.1)  # inf = never chunk


def test_duration_policy_above_max_raises_with_both_remediations():
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_duration_policy("hello world", 46.0, 45.0, 1.1)
    message = str(excinfo.value)
    assert "46" in message and "45" in message
    assert "omit" in message
    assert "--max-chunk-duration" in message
    assert "silence prior" in message


# --------------------------------------------------------------------------- #
# CLI: parse first, model load never happens on a validation failure
# --------------------------------------------------------------------------- #
def _run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "DRAMABOX_ALLOW_NO_CONFIG": "0", **(env or {})},
    )


def test_cli_help_exposes_output_alias():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "-o" in result.stdout
    assert "--output" in result.stdout
    assert "--keep-text-encoder" in result.stdout


def test_cli_help_survives_bad_dit_bits_env():
    result = _run_cli("--help", env={"DRAMABOX_DIT_BITS": "not-an-int"})
    assert result.returncode == 0
    assert "--output" in result.stdout


@pytest.mark.parametrize("extra,needle", [
    (["--cfg-scale", "0.5"], "cfg_scale"),
    (["--steps", "1"], "steps"),
    (["--ref-duration", "0"], "ref_duration"),
    (["--rescale", "bogus"], "rescale"),
    (["--duration", "100", "--max-chunk-duration", "45"], "exceeds max_chunk_duration"),
    (["--max-chunk-duration", "10", "--target-chunk-duration", "20"], "target_chunk_duration"),
    (["--dit-bits", "3"], "invalid choice"),
])
def test_cli_invalid_args_exit_before_model_load(extra, needle):
    result = _run_cli("--prompt", "hello", "--model-dir", "/nonexistent", *extra)
    assert result.returncode == 2
    assert needle in result.stderr
    assert "loading DiT" not in result.stderr
    assert "unrecognized arguments" not in result.stderr


def test_cli_output_flag_is_parsed_before_validation(tmp_path):
    out = tmp_path / "out.wav"
    result = _run_cli("--prompt", "hello", "--model-dir", "/nonexistent",
                      "-o", str(out), "--steps", "1")
    assert result.returncode == 2
    assert "steps" in result.stderr
    assert "unrecognized arguments: -o" not in result.stderr
    assert "loading DiT" not in result.stderr


def test_cli_missing_model_dir_fails_before_load():
    result = _run_cli("--prompt", "hello", "--model-dir", "/nonexistent")
    assert result.returncode == 2
    assert "model directory not found" in result.stderr
    assert "loading DiT" not in result.stderr


def test_cli_inf_chunk_rejects_huge_auto_estimate_before_load():
    result = _run_cli("--prompt", "hello", "--model-dir", "/nonexistent",
                      "--max-chunk-duration", "inf",
                      "--duration-multiplier", "1000")
    assert result.returncode == 2
    assert "single-shot limit" in result.stderr
    assert "loading DiT" not in result.stderr


def test_cli_duration_policy_message_has_remediations():
    result = _run_cli("--prompt", "hello", "--model-dir", "/nonexistent",
                      "--duration", "100", "--max-chunk-duration", "45")
    assert result.returncode == 2
    assert "omit" in result.stderr
    assert "--max-chunk-duration" in result.stderr


# --------------------------------------------------------------------------- #
# long-form empty-chunk / None-waveform guards
# --------------------------------------------------------------------------- #
def test_generate_long_rejects_empty_chunks(monkeypatch):
    import mlx_dramabox.text_chunker as text_chunker

    monkeypatch.setattr(text_chunker, "chunk_prompt_for_duration", lambda *a, **k: [])
    server = object.__new__(DramaBoxMLX)
    with pytest.raises(ValueError, match="no chunks"):
        server.generate_long("   ", max_chunk_duration=45.0)


def test_generate_long_rejects_gen_duration():
    server = object.__new__(DramaBoxMLX)
    with pytest.raises(ValueError, match="gen_duration"):
        server.generate_long("hello", gen_duration=5.0)


def test_generate_long_rejects_none_waveform(monkeypatch):
    import mlx_dramabox.text_chunker as text_chunker

    monkeypatch.setattr(
        text_chunker, "chunk_prompt_for_duration",
        lambda *a, **k: [SimpleNamespace(text="hello", est_duration_s=1.0)],
    )
    server = object.__new__(DramaBoxMLX)
    server.keep_text_encoder = True
    server.gemma_dir = "<test>"
    server.connector = object()
    server.text = SimpleNamespace(encode=lambda text, connector: mx.zeros((1, 2, 2048)))
    server._ensure_text = lambda reload=True: None
    server._negative_context = lambda: mx.zeros((1, 2, 2048))
    server.generate = lambda *a, **k: None
    with pytest.raises(RuntimeError, match="no waveform"):
        server.generate_long("hello", max_chunk_duration=45.0)


def test_generate_long_chunk_seed_mode(monkeypatch):
    import mlx_dramabox.text_chunker as text_chunker

    monkeypatch.setattr(
        text_chunker, "chunk_prompt_for_duration",
        lambda *a, **k: [SimpleNamespace(text="a", est_duration_s=1.0),
                         SimpleNamespace(text="b", est_duration_s=1.0)],
    )
    server = object.__new__(DramaBoxMLX)
    server.keep_text_encoder = True
    server.gemma_dir = "<test>"
    server.connector = object()
    server.text = SimpleNamespace(encode=lambda text, connector: mx.zeros((1, 2, 2048)))
    server._ensure_text = lambda reload=True: None
    server._negative_context = lambda: mx.zeros((1, 2, 2048))
    seeds = []

    def fake_generate(prompt, **kwargs):
        seeds.append(kwargs.get("seed"))
        return mx.zeros((1, 2, 8))

    server.generate = fake_generate
    server.generate_long("hello", max_chunk_duration=45.0, cfg_scale=1.0, seed=42)
    assert seeds == [42, 42]                     # reference-identical default
    seeds.clear()
    server.generate_long("hello", max_chunk_duration=45.0, cfg_scale=1.0, seed=42,
                         chunk_seed_mode="indexed")
    assert seeds == [42, 43]                     # opt-in per-chunk noise
    with pytest.raises(ValueError, match="chunk_seed_mode"):
        server.generate_long("hello", chunk_seed_mode="bad")


def test_generate_to_file_rejects_none_waveform(tmp_path):
    server = object.__new__(DramaBoxMLX)
    server.generate = lambda *a, **k: None
    with pytest.raises(RuntimeError, match="no waveform"):
        server.generate_to_file("hello", str(tmp_path / "out.wav"))


# --------------------------------------------------------------------------- #
# precomputed (ctx, neg) semantics
# --------------------------------------------------------------------------- #
def _patch_decode(monkeypatch):
    monkeypatch.setattr(pipeline, "decode_latent", lambda *a, **k: mx.zeros((1, 2, 8)))


def test_precomputed_cfg_gt_one_requires_negative(monkeypatch):
    _patch_decode(monkeypatch)
    server = make_fake_pipeline()
    ctx = mx.zeros((1, 2, 2048), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="negative context"):
        server.generate("hello", cfg_scale=2.0, stg_scale=0.0, steps=2,
                        gen_duration=1.0, precomputed=(ctx, None), verbose=False)


def test_precomputed_cfg_one_ignores_negative(monkeypatch):
    _patch_decode(monkeypatch)
    server = make_fake_pipeline()
    ctx = mx.zeros((1, 2, 2048), dtype=mx.bfloat16)
    neg = mx.ones((1, 2, 2048), dtype=mx.bfloat16)
    server.generate("hello", cfg_scale=1.0, stg_scale=0.0, steps=2,
                    gen_duration=1.0, precomputed=(ctx, neg), verbose=False)
    assert len(server.dit.contexts) == 2
    assert all(context is ctx for context in server.dit.contexts)


def test_precomputed_avoids_text_encoding(monkeypatch):
    _patch_decode(monkeypatch)
    server = make_fake_pipeline()

    def boom(*args, **kwargs):
        raise AssertionError("text encoder must not be used with precomputed contexts")

    server._ensure_text = boom
    server.text = SimpleNamespace(encode=boom)
    server.encode_prompt = boom
    ctx = mx.zeros((1, 2, 2048), dtype=mx.bfloat16)
    server.generate("hello", cfg_scale=1.0, stg_scale=0.0, steps=2,
                    gen_duration=1.0, precomputed=(ctx, None), verbose=False)
