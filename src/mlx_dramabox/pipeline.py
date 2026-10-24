"""DramaBox TTS inference pipeline on MLX."""

from __future__ import annotations

import inspect
import math
import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .audio import (SAMPLE_RATE, decode_latent, encode_reference, load_audio_codec,
                    prepare_reference)
try:  # Transitional: older copies of audio.py may not define OUTPUT_SAMPLE_RATE.
    from .audio import OUTPUT_SAMPLE_RATE
except ImportError:  # pragma: no cover - transitional fallback
    OUTPUT_SAMPLE_RATE = 48000
from .model import load_dit, make_rope
from .text import GemmaTextEncoder, load_connector

DEFAULT_NEG = ("worst quality, inconsistent, robotic, distorted, noise, static, "
               "muffled, unclear, unnatural, monotone")
STG_BLOCKS = (29,)
LATENT_PER_SECOND = SAMPLE_RATE / 160 / 4  # 25

MAX_DURATION = 600.0
MAX_REF_DURATION = 60.0
SAFE_CHUNK_DURATION = 60.0
MIN_STEPS = 2
MAX_STEPS = 200

# Negative-context cache: ~1 MB per entry, keyed on the checkpoint + the fixed prompt.
_NEG_CACHE: "OrderedDict[tuple, tuple[mx.array, object]]" = OrderedDict()
_NEG_CACHE_MAX = 4
# Voice-reference cache: validated file identity + requested duration -> (tokens, positions).
_REF_CACHE: "OrderedDict[tuple, tuple[mx.array, mx.array, object]]" = OrderedDict()
_REF_CACHE_MAX = 4


def audio_latent_frames(duration: float) -> int:
    """Match VideoPixelShape(frame snapping) -> AudioLatentShape.from_video_pixel_shape."""
    n_frames = int(round(duration * 25.0)) + 1
    n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
    return int(round(n_frames / 25.0 * LATENT_PER_SECOND))


def audio_positions(num_frames: int) -> mx.array:
    """(1, 1, T, 2) start/end timestamps in seconds (upstream AudioPatchifier)."""
    idx = mx.arange(num_frames, dtype=mx.float32)
    start = mx.maximum(idx * 4.0 - 3.0, 0.0) * 0.01
    end = mx.maximum((idx + 1) * 4.0 - 3.0, 0.0) * 0.01
    return mx.stack([start, end], axis=-1)[None, None]


def patchify(latent: mx.array) -> mx.array:
    b, c, t, f = latent.shape
    return latent.transpose(0, 2, 1, 3).reshape(b, t, c * f)


def unpatchify(tokens: mx.array) -> mx.array:
    b, t, _ = tokens.shape
    return tokens.reshape(b, t, 8, 16).transpose(0, 2, 1, 3)


def sigma_schedule(steps: int, latent_tokens: int = 128) -> mx.array:
    """LTX2Scheduler.execute with upstream defaults (note: tokens = latent.shape[2:] product)."""
    if steps < 2:
        raise ValueError(f"steps must be >= 2 (got {steps})")
    sigmas = mx.linspace(1.0, 0.0, steps + 1)
    x1, x2 = 1024.0, 4096.0
    mm = (2.05 - 0.95) / (x2 - x1)
    b = 0.95 - mm * x1
    shift = latent_tokens * mm + b
    e = math.exp(shift)
    sigmas = mx.where(sigmas != 0, e / (e + (1.0 / mx.maximum(sigmas, 1e-12) - 1.0)), 0.0)
    one_minus = 1.0 - sigmas
    scale = one_minus[steps - 1] / (1.0 - 0.1)
    stretched = 1.0 - one_minus / scale
    return mx.where(sigmas != 0, stretched, mx.zeros_like(sigmas))


def auto_rescale(cfg: float) -> float:
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


def _attention_mask(num_target: int, num_ref: int) -> mx.array:
    total = num_target + num_ref
    mask = mx.zeros((1, total, total), dtype=mx.float32)
    mask[:, :num_target, :num_target] = 1.0
    mask[:, :num_target, num_target:] = 1.0
    mask[:, num_target:, num_target:] = 1.0
    return mask


def _cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache, key, value, cap: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > cap:
        cache.popitem(last=False)


def _as_float(name: str, value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _parse_rescale(value):
    """Return 'auto' or a finite float in [0, 1]."""
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "auto":
            return "auto"
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"rescale must be 'auto' or a finite number in [0, 1], got {text!r}") from None
    value = _as_float("rescale", value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"rescale must be 'auto' or a finite number in [0, 1], got {value!r}")
    return value


def validate_generation_params(*, cfg_scale=2.5, stg_scale=1.5, steps=30,
                               gen_duration=0.0, duration_multiplier=1.1,
                               ref_duration=10.0, rescale_scale="auto",
                               max_chunk_duration=45.0,
                               target_chunk_duration=37.0,
                               crossfade_ms=50.0) -> None:
    """Validate numeric generation parameters (P0-3/P0-7) before any model work."""
    cfg = _as_float("cfg_scale", cfg_scale)
    stg = _as_float("stg_scale", stg_scale)
    if not math.isfinite(cfg):
        raise ValueError(f"cfg_scale must be finite, got {cfg_scale!r}")
    if not math.isfinite(stg):
        raise ValueError(f"stg_scale must be finite, got {stg_scale!r}")
    if cfg < 1.0:
        raise ValueError(
            f"cfg_scale must be >= 1.0, got {cfg_scale!r}: the reference has no negative "
            "context for sub-1 guidance (this is not inverted CFG)")
    _parse_rescale(rescale_scale)

    steps_f = _as_float("steps", steps)
    if not steps_f.is_integer() or not MIN_STEPS <= steps_f <= MAX_STEPS:
        raise ValueError(
            f"steps must be an integer in [{MIN_STEPS}, {MAX_STEPS}], got {steps!r}")

    duration = _as_float("duration", gen_duration or 0.0)
    if not math.isfinite(duration) or not 0.0 <= duration <= MAX_DURATION:
        raise ValueError(
            f"duration must be finite and in [0, {MAX_DURATION:g}] seconds, got {gen_duration!r}")

    ref = _as_float("ref_duration", ref_duration)
    if not math.isfinite(ref) or not 0.0 < ref <= MAX_REF_DURATION:
        raise ValueError(
            f"ref_duration must be finite and in (0, {MAX_REF_DURATION:g}] seconds, "
            f"got {ref_duration!r}")

    multiplier = _as_float("duration_multiplier", duration_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            f"duration_multiplier must be finite and > 0, got {duration_multiplier!r}")

    max_chunk = _as_float("max_chunk_duration", max_chunk_duration)
    if math.isnan(max_chunk) or max_chunk <= 0.0:
        raise ValueError(
            f"max_chunk_duration must be > 0 or +inf, got {max_chunk_duration!r}")

    target_chunk = _as_float("target_chunk_duration", target_chunk_duration)
    if not math.isfinite(target_chunk) or target_chunk <= 0.0:
        raise ValueError(
            f"target_chunk_duration must be finite and > 0, got {target_chunk_duration!r}")
    if max_chunk < target_chunk:
        raise ValueError(
            f"max_chunk_duration ({max_chunk:g}) must be >= target_chunk_duration "
            f"({target_chunk:g})")

    crossfade = _as_float("crossfade_ms", crossfade_ms)
    if not math.isfinite(crossfade) or crossfade < 0.0:
        raise ValueError(f"crossfade_ms must be finite and >= 0, got {crossfade_ms!r}")


def validate_duration_policy(prompt: str, gen_duration: float = 0.0,
                             max_chunk_duration: float = 45.0,
                             duration_multiplier: float = 1.1) -> None:
    """Reject an explicit duration that cannot fit in one chunk (P0-2)."""
    requested = _as_float("duration", gen_duration or 0.0)
    cap = _as_float("max_chunk_duration", max_chunk_duration)
    if requested <= 0.0 or not math.isfinite(cap) or requested <= cap:
        return
    from .duration_estimator import estimate_speech_duration

    multiplier = _as_float("duration_multiplier", duration_multiplier)
    estimate = max(3.0, round(estimate_speech_duration(prompt) * multiplier, 1))
    raise ValueError(
        f"requested duration {requested:g}s exceeds max_chunk_duration {cap:g}s "
        f"(prompt estimate {estimate:.1f}s); omit gen_duration/--duration to auto-split, "
        f"or raise max_chunk_duration/--max-chunk-duration to >= {requested:g}s "
        f"(warning: durations > 45s may reintroduce the silence prior)")


def _warn_caps(*, max_chunk_duration: float | None = None,
               gen_duration: float | None = None) -> None:
    if max_chunk_duration is not None and max_chunk_duration > SAFE_CHUNK_DURATION:
        warnings.warn(
            f"max_chunk_duration={max_chunk_duration:g}s exceeds {SAFE_CHUNK_DURATION:g}s; "
            "long chunks can trigger the silence prior or run out of memory",
            RuntimeWarning, stacklevel=3)
    if gen_duration is not None and gen_duration > SAFE_CHUNK_DURATION:
        warnings.warn(
            f"single-shot duration={gen_duration:g}s exceeds {SAFE_CHUNK_DURATION:g}s; "
            "this is a large O(T^2) attention graph and may run out of memory",
            RuntimeWarning, stacklevel=3)


def _build_rope(model, positions):
    """Call make_rope with the model's rope type, tolerant of an older model.py."""
    mid = ((positions[..., 0] + positions[..., 1]) / 2.0)[:, 0]
    rope_type = getattr(model, "rope_type", "split")
    try:
        accepts_rope_type = "rope_type" in inspect.signature(make_rope).parameters
    except (TypeError, ValueError):
        accepts_rope_type = False
    return make_rope(mid, rope_type=rope_type) if accepts_rope_type else make_rope(mid)


def _crossfade(prev, nxt, sample_rate: int, fade_ms: float = 50.0):
    if prev.shape[-2] != nxt.shape[-2]:
        if prev.shape[-2] == 1:
            prev = mx.repeat(prev, nxt.shape[-2], axis=-2)
        elif nxt.shape[-2] == 1:
            nxt = mx.repeat(nxt, prev.shape[-2], axis=-2)
        else:
            raise ValueError(
                f"cannot crossfade mismatched channel counts {prev.shape[-2]} and "
                f"{nxt.shape[-2]}")
    fade = max(1, min(int(round(fade_ms * 1e-3 * sample_rate)), prev.shape[-1], nxt.shape[-1]))
    if fade <= 1:
        return mx.concatenate([prev, nxt], axis=-1)
    t = mx.linspace(0.0, 1.0, fade)
    mixed = prev[..., -fade:] * mx.cos(t * mx.pi / 2) + nxt[..., :fade] * mx.sin(t * mx.pi / 2)
    return mx.concatenate([prev[..., :-fade], mixed, nxt[..., fade:]], axis=-1)


def _save_wav(wav: mx.array, output: str) -> None:
    import soundfile as sf

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    data = mx.clip(wav[0], -1, 1).transpose(1, 0).astype(mx.float32)
    sf.write(output, mx_to_np(data), OUTPUT_SAMPLE_RATE)


class DramaBoxMLX:
    def __init__(self, model_dir="models", gemma_dir=None, dit_bits: int | None = None,
                 keep_text_encoder: bool = True):
        if dit_bits not in (None, 0, 4, 8):
            raise ValueError(f"dit_bits must be None/0 (bf16), 4 or 8, got {dit_bits!r}")
        t0 = time.time()
        dit_path = f"{model_dir}/dramabox-dit-v1.safetensors"
        comp_path = f"{model_dir}/dramabox-audio-components.safetensors"
        gemma_dir = gemma_dir or f"{model_dir}/gemma-3-12b-it-4bit"

        print(f"[mlx] loading DiT (bits={dit_bits or 'bf16'})...", flush=True)
        self.dit = load_dit(dit_path, quantize_bits=dit_bits)
        print(f"[mlx] DiT ready ({time.time()-t0:.1f}s)", flush=True)

        print("[mlx] loading audio codec...", flush=True)
        self.encoder, self.decoder, self.vocoder = load_audio_codec(comp_path)
        print(f"[mlx] codec ready ({time.time()-t0:.1f}s)", flush=True)

        print("[mlx] loading audio text connector...", flush=True)
        self.connector = load_connector(comp_path)

        print("[mlx] loading Gemma-3 12B...", flush=True)
        self.gemma_dir = gemma_dir
        self.text = GemmaTextEncoder(gemma_dir)
        self.keep_text_encoder = keep_text_encoder
        mx.clear_cache()
        print(f"[mlx] all models ready ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    def _ensure_text(self, reload: bool = True) -> None:
        if self.text is not None:
            return
        if not reload:
            raise RuntimeError("text encoder is not loaded; pass reload=True")
        print("[mlx] (re)loading Gemma-3 12B...", flush=True)
        self.text = GemmaTextEncoder(self.gemma_dir)

    def _free_text(self) -> None:
        if self.text is not None:
            self.text.free()
            self.text = None
            print("[mlx] text encoder freed", flush=True)

    def _negative_context(self) -> mx.array:
        # nn.Module defines __eq__, so it is unhashable. Use id() for the key and
        # keep the live connector in the cached *value*: the strong reference
        # prevents the id from being recycled while the entry exists.
        connector = self.connector
        key = (self.gemma_dir, DEFAULT_NEG, id(connector))
        cached = _cache_get(_NEG_CACHE, key)
        if cached is not None:
            return cached[0]
        ctx = self.text.encode(DEFAULT_NEG, connector)
        _cache_put(_NEG_CACHE, key, (ctx, connector), _NEG_CACHE_MAX)
        return ctx

    def encode_prompt(self, prompt: str, negative: bool = True,
                      reload: bool = True) -> tuple[mx.array, mx.array | None]:
        self._ensure_text(reload=reload)
        try:
            ctx = self.text.encode(prompt, self.connector)
            neg = self._negative_context() if negative else None
        finally:
            if not self.keep_text_encoder:
                self._free_text()
        return ctx, neg

    def _reference_tokens(self, voice_ref: str, ref_duration: float,
                          verbose: bool = True) -> tuple[mx.array, mx.array]:
        """Load/encode a voice reference, with a validated file-identity cache."""
        real = os.path.realpath(voice_ref)
        try:
            st = os.stat(real)
        except OSError as exc:
            raise FileNotFoundError(f"voice reference not found: {voice_ref}") from exc
        encoder = self.encoder
        key = (real, st.st_size, st.st_mtime_ns, st.st_ino, float(ref_duration), id(encoder))
        cached = _cache_get(_REF_CACHE, key)
        if cached is not None:
            if verbose:
                print("[mlx] voice ref cache hit", flush=True)
            return cached[0], cached[1]
        wave = prepare_reference(voice_ref, ref_duration)
        ref_tokens = patchify(encode_reference(encoder, wave))
        ref_pos = audio_positions(ref_tokens.shape[1]) + 0.5
        mx.eval(ref_tokens, ref_pos)
        _cache_put(_REF_CACHE, key, (ref_tokens, ref_pos, encoder), _REF_CACHE_MAX)
        return ref_tokens, ref_pos

    @staticmethod
    def _denoise_pass(model, latent, timesteps, sigma, context, rope, mask, skip=()):
        velocity = model(latent, timesteps, sigma, context, None, mask, skip_blocks=skip, rope=rope)
        denoised = (latent.astype(mx.float32) - velocity.astype(mx.float32) * timesteps).astype(latent.dtype)
        return denoised

    def generate(self, prompt: str, voice_ref: str | None = None, output: str | None = None,
                 cfg_scale: float = 2.5, stg_scale: float = 1.5,
                 steps: int = 30, seed: int = 42, gen_duration: float = 0.0,
                 duration_multiplier: float = 1.1, ref_duration: float = 10.0,
                 rescale_scale: float | str = "auto", verbose: bool = True,
                 precomputed: tuple[mx.array, mx.array | None] | None = None):
        """Generate audio and return a (1, 2, N) float32 waveform (reference: (C, T))."""
        t_start = time.time()
        gen_duration = float(gen_duration or 0.0)
        validate_generation_params(
            cfg_scale=cfg_scale, stg_scale=stg_scale, steps=steps,
            gen_duration=gen_duration, duration_multiplier=duration_multiplier,
            ref_duration=ref_duration, rescale_scale=rescale_scale)
        cfg_scale = float(cfg_scale)
        stg_scale = float(stg_scale)
        steps = int(steps)
        ref_duration = float(ref_duration)
        duration_multiplier = float(duration_multiplier)
        rescale_scale = _parse_rescale(rescale_scale)
        seed = int(seed) % (2**32)
        if precomputed is not None and cfg_scale > 1.0 and precomputed[1] is None:
            raise ValueError("precomputed negative context is required when cfg_scale > 1.0")
        if gen_duration > SAFE_CHUNK_DURATION:
            _warn_caps(gen_duration=gen_duration)

        if gen_duration > 0:
            duration = gen_duration
        else:
            from .duration_estimator import estimate_speech_duration

            duration = max(3.0, round(estimate_speech_duration(prompt) * duration_multiplier, 1))
        if duration > MAX_DURATION:
            raise ValueError(
                f"resolved duration {duration:g}s exceeds the {MAX_DURATION:g}s hard limit; "
                "shorten the prompt, lower --duration-multiplier, or use smaller chunks")

        num_frames = audio_latent_frames(duration)
        pos_target = audio_positions(num_frames)
        latent = mx.zeros((1, num_frames, 128), dtype=mx.bfloat16)
        clean = mx.zeros_like(latent)
        denoise_mask = mx.ones((1, num_frames, 1), dtype=mx.float32)
        positions = pos_target
        attn_mask = None

        # --- optional voice reference conditioning ---
        if voice_ref:
            t0 = time.time()
            ref_tokens, ref_pos = self._reference_tokens(voice_ref, ref_duration, verbose)
            ref_mask = mx.zeros((1, ref_tokens.shape[1], 1), dtype=mx.float32)
            attn_mask = _attention_mask(num_frames, ref_tokens.shape[1])
            latent = mx.concatenate([latent, ref_tokens.astype(mx.bfloat16)], axis=1)
            clean = mx.concatenate([clean, ref_tokens.astype(mx.bfloat16)], axis=1)
            denoise_mask = mx.concatenate([denoise_mask, ref_mask], axis=1)
            positions = mx.concatenate([positions, ref_pos], axis=2)
            if verbose:
                print(f"[mlx] voice ref ready in {time.time()-t0:.1f}s", flush=True)

        # --- noise ---
        noise = mx.random.normal((1, *latent.shape[1:]), key=mx.random.key(seed)).astype(mx.float32)
        latent = (noise * denoise_mask + latent.astype(mx.float32) * (1 - denoise_mask)).astype(mx.bfloat16)

        # --- prompt ---
        if precomputed is not None:
            ctx, neg = precomputed
            if cfg_scale <= 1.0:
                neg = None
            if verbose:
                print("[mlx] using precomputed prompt context", flush=True)
        else:
            t0 = time.time()
            ctx, neg = self.encode_prompt(prompt, negative=cfg_scale > 1.0)
            if verbose:
                print(f"[mlx] prompt encoded in {time.time()-t0:.1f}s", flush=True)

        resc = auto_rescale(cfg_scale) if rescale_scale == "auto" else float(rescale_scale)
        sigmas = sigma_schedule(steps, latent_tokens=128)
        mx.eval(sigmas)
        rope = _build_rope(self.dit, positions)

        # --- denoise ---
        t0 = time.time()
        for step in range(steps):
            sigma = float(sigmas[step])
            dt = float(sigmas[step + 1] - sigmas[step])
            sigma_arr = mx.array([sigma], dtype=mx.float32)
            timesteps = denoise_mask * sigma_arr

            cond = self._denoise_pass(self.dit, latent, timesteps, sigma_arr, ctx, rope, attn_mask)
            pred = cond
            if neg is not None:
                uncond = self._denoise_pass(self.dit, latent, timesteps, sigma_arr, neg, rope, attn_mask)
                pred = pred.astype(mx.float32) + (cfg_scale - 1.0) * (pred.astype(mx.float32) - uncond.astype(mx.float32))
                pred = pred.astype(mx.bfloat16)
            if stg_scale != 0.0:
                ptb = self._denoise_pass(self.dit, latent, timesteps, sigma_arr, ctx, rope, attn_mask,
                                         skip=STG_BLOCKS)
                pred = pred.astype(mx.float32) + stg_scale * (cond.astype(mx.float32) - ptb.astype(mx.float32))
                pred = pred.astype(mx.bfloat16)

            if resc != 0.0:
                p32 = pred.astype(mx.float32)
                factor = mx.std(cond.astype(mx.float32), ddof=1) / mx.std(p32, ddof=1)
                factor = resc * factor + (1 - resc)
                pred = (p32 * factor).astype(mx.bfloat16)

            pred = pred.astype(mx.float32) * denoise_mask + clean.astype(mx.float32) * (1 - denoise_mask)
            pred = pred.astype(mx.bfloat16)

            velocity = ((latent.astype(mx.float32) - pred.astype(mx.float32)) / sigma).astype(mx.bfloat16)
            latent = (latent.astype(mx.float32) + velocity.astype(mx.float32) * dt).astype(mx.bfloat16)
            mx.eval(latent)
            if verbose and (step + 1) % 5 == 0:
                print(f"[mlx] step {step+1}/{steps}", flush=True)

        if verbose:
            print(f"[mlx] denoised in {time.time()-t0:.1f}s", flush=True)

        # --- strip extra conditioning tokens, unpatchify ---
        latent = latent[:, :num_frames]
        # Reachable token counts are 8k+1; T == 514 cannot occur today. Keeping
        # "> 514" avoids the reference's f=513 repair reading frame 514 out of bounds.
        if latent.shape[1] > 514:
            f0, f1 = 511, 514
            patched = mx.array(latent)
            for f in (512, 513):
                t = (f - f0) / (f1 - f0)
                patched[:, f] = (1.0 - t) * latent[:, f0] + t * latent[:, f1]
            latent = patched
        latent = unpatchify(latent)

        t0 = time.time()
        wav = decode_latent(self.decoder, self.vocoder, latent)
        if verbose:
            print(f"[mlx] decoded in {time.time()-t0:.1f}s", flush=True)

        if output:
            _save_wav(wav, output)
            if verbose:
                dur = wav.shape[-1] / OUTPUT_SAMPLE_RATE
                print(f"[mlx] saved {output} ({dur:.1f}s, total {time.time()-t_start:.1f}s, "
                      f"peak {mx.get_peak_memory()/1e9:.1f} GB)", flush=True)
        return wav

    def generate_long(self, prompt: str, max_chunk_duration: float = 45.0,
                      target_chunk_duration: float = 37.0, crossfade_ms: float = 50.0,
                      chunk_seed_mode: str = "same", **kwargs):
        from .text_chunker import chunk_prompt_for_duration

        if "gen_duration" in kwargs:
            raise ValueError(
                "generate_long() does not accept gen_duration; omit --duration so the prompt "
                "estimator sizes each chunk, or use generate_to_file() for routing")
        if "output" in kwargs:
            print("[mlx] warning: output is ignored by generate_long(); write the returned "
                  "waveform instead", flush=True)
            kwargs.pop("output")
        if "precomputed" in kwargs:
            raise ValueError(
                "generate_long() encodes its own chunk contexts; do not pass precomputed")
        if chunk_seed_mode not in ("same", "indexed"):
            raise ValueError(
                f"chunk_seed_mode must be 'same' or 'indexed', got {chunk_seed_mode!r}")
        max_chunk_duration = float(max_chunk_duration)
        target_chunk_duration = float(target_chunk_duration)
        crossfade_ms = float(crossfade_ms)
        multiplier = float(kwargs.get("duration_multiplier", 1.1))
        validate_generation_params(
            cfg_scale=kwargs.get("cfg_scale", 2.5),
            stg_scale=kwargs.get("stg_scale", 1.5),
            steps=kwargs.get("steps", 30),
            gen_duration=0.0,
            duration_multiplier=multiplier,
            ref_duration=kwargs.get("ref_duration", 10.0),
            rescale_scale=kwargs.get("rescale_scale", "auto"),
            max_chunk_duration=max_chunk_duration,
            target_chunk_duration=target_chunk_duration,
            crossfade_ms=crossfade_ms,
        )
        _warn_caps(max_chunk_duration=max_chunk_duration)

        chunks = chunk_prompt_for_duration(
            prompt, max_duration_s=max_chunk_duration,
            target_duration_s=target_chunk_duration,
            duration_multiplier=multiplier,
        )
        if not chunks:
            raise ValueError("prompt produced no chunks (empty or whitespace-only prompt?)")

        cfg_scale = float(kwargs.get("cfg_scale", 2.5))
        need_neg = cfg_scale > 1.0
        # Encode every chunk once while Gemma is resident, then free once.
        self._ensure_text(reload=True)
        print(f"[mlx] encoding {len(chunks)} chunk prompt(s)...", flush=True)
        try:
            neg = self._negative_context() if need_neg else None
            contexts = [self.text.encode(chunk.text, self.connector) for chunk in chunks]
        finally:
            if not self.keep_text_encoder:
                self._free_text()

        out = None
        for i, (chunk, ctx) in enumerate(zip(chunks, contexts)):
            print(f"[mlx] long-form chunk {i+1}/{len(chunks)} (~{chunk.est_duration_s:.1f}s)", flush=True)
            chunk_kwargs = dict(kwargs)
            if chunk_seed_mode == "indexed":
                # Opt-in: break the reference's identical-noise-for-equal-chunks
                # behavior without changing the default.
                chunk_kwargs["seed"] = int(chunk_kwargs.get("seed", 42)) + i
            wav = self.generate(chunk.text, precomputed=(ctx, neg), **chunk_kwargs)
            if wav is None:
                raise RuntimeError(f"long-form chunk {i+1} produced no waveform")
            out = wav if out is None else _crossfade(out, wav, OUTPUT_SAMPLE_RATE, crossfade_ms)
            mx.eval(out)
            if i + 1 < len(chunks):
                mx.clear_cache()
        if out is None:
            raise RuntimeError("long-form generation produced no audio")
        return out

    def generate_to_file(self, prompt: str, output: str, max_chunk_duration: float = 45.0,
                         target_chunk_duration: float = 37.0, crossfade_ms: float = 50.0,
                         chunk_seed_mode: str = "same", **kwargs):
        from .duration_estimator import estimate_speech_duration

        gen_duration = float(kwargs.get("gen_duration") or 0.0)
        kwargs["gen_duration"] = gen_duration
        if chunk_seed_mode not in ("same", "indexed"):
            raise ValueError(
                f"chunk_seed_mode must be 'same' or 'indexed', got {chunk_seed_mode!r}")
        max_chunk_duration = float(max_chunk_duration)
        target_chunk_duration = float(target_chunk_duration)
        crossfade_ms = float(crossfade_ms)
        multiplier = float(kwargs.get("duration_multiplier", 1.1))
        validate_generation_params(
            cfg_scale=kwargs.get("cfg_scale", 2.5),
            stg_scale=kwargs.get("stg_scale", 1.5),
            steps=kwargs.get("steps", 30),
            gen_duration=gen_duration,
            duration_multiplier=multiplier,
            ref_duration=kwargs.get("ref_duration", 10.0),
            rescale_scale=kwargs.get("rescale_scale", "auto"),
            max_chunk_duration=max_chunk_duration,
            target_chunk_duration=target_chunk_duration,
            crossfade_ms=crossfade_ms,
        )
        validate_duration_policy(prompt, gen_duration, max_chunk_duration, multiplier)

        est = gen_duration if gen_duration > 0 else max(
            3.0, round(estimate_speech_duration(prompt) * multiplier, 1))
        if est > max_chunk_duration:
            kwargs.pop("gen_duration", None)
            wav = self.generate_long(prompt, max_chunk_duration, target_chunk_duration,
                                     crossfade_ms, chunk_seed_mode=chunk_seed_mode, **kwargs)
            if wav is None:
                raise RuntimeError("generation produced no waveform")
            _save_wav(wav, output)
            print(f"[mlx] saved {output} ({wav.shape[-1]/OUTPUT_SAMPLE_RATE:.1f}s, "
                  f"peak {mx.get_peak_memory()/1e9:.1f} GB)", flush=True)
            return wav

        _warn_caps(max_chunk_duration=max_chunk_duration)
        _warn_caps(gen_duration=gen_duration)
        wav = self.generate(prompt, output=output, **kwargs)
        if wav is None:
            raise RuntimeError("generation produced no waveform")
        return wav


def mx_to_np(x: mx.array):
    import numpy as np

    return np.array(x)
