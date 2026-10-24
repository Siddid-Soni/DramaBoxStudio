"""Audio VAE + BigVGAN/BWE codec for DramaBox on MLX.

Notes:
* The inner vocoder runs at ``SAMPLE_RATE`` (16 kHz). The BWE stage upsamples by
  ``VocoderWithBWE._resampler.upsample_factor`` (3) to ``OUTPUT_SAMPLE_RATE`` (48 kHz).
* ``prepare_reference`` uses the first two channels of the input (no downmix) and duplicates
  mono to stereo.
* Unsigned 8-bit PCM (``PCM_U8``) is converted as ``(arr - 128) / 128`` in the PyAV
  fallback, but libsndfile/PyAV builds may refuse to decode ``PCM_U8`` at all, so it
  is treated as possibly unsupported (PyAV 19 decodes it correctly).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import mlx.core as mx

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.audio_vae.encoder import AudioVAEEncoder
from ltx_core_mlx.model.audio_vae.processor import AudioProcessor

SAMPLE_RATE = 16000            # inner vocoder / VAE rate
OUTPUT_SAMPLE_RATE = 48000     # after the BWE stage
MAX_SAMPLE_RATE = 384_000      # reject absurd sample rates
MAX_REF_DURATION = 60.0        # seconds; hard cap on reference decode/decode window
MAX_REF_SAMPLES = int(MAX_REF_DURATION * MAX_SAMPLE_RATE)  # hard cap on target sample count
N_FFT = 1024                   # AudioProcessor STFT window (needs >= this many samples)
LATENT_CHANNELS = 8
LATENT_FREQ_BINS = 16


def _fix_stats(key: str) -> str:
    return (key.replace("._mean_of_means", ".mean_of_means")
               .replace("._std_of_means", ".std_of_means")
               .replace(".mean-of-means", ".mean_of_means")
               .replace(".std-of-means", ".std_of_means"))


def _transpose_vae(key: str, value: mx.array) -> mx.array:
    if key.endswith(".conv.weight") and value.ndim == 4:
        return value.transpose(0, 2, 3, 1)  # (out,in,kH,kW) -> (out,kH,kW,in)
    return value


def _transpose_vocoder(key: str, value: mx.array) -> mx.array:
    if value.ndim == 3 and (key.endswith(".filter") or key.endswith("_basis")):
        return value.transpose(0, 2, 1)     # (1,1,k) -> (1,k,1)
    if value.ndim != 3 or not key.endswith(".weight"):
        return value
    parts = key.split(".")
    if len(parts) >= 2 and parts[-2].isdigit() and "ups" in parts:
        return value.transpose(1, 2, 0)     # ConvTranspose1d (in,out,k) -> (out,k,in)
    return value.transpose(0, 2, 1)         # Conv1d (out,in,k) -> (out,k,in)


def _extract(raw: dict, prefix: str, stats: dict) -> dict:
    out = {}
    for key, value in raw.items():
        if key.startswith(prefix):
            out[_fix_stats(key[len(prefix):])] = _transpose_vae(key, value)
    out.update(stats)
    return out


def load_audio_codec(checkpoint_path: str):
    """Load encoder, decoder and vocoder from dramabox-audio-components.safetensors."""
    raw = mx.load(checkpoint_path)

    stats = {_fix_stats(k[len("audio_vae."):]): v for k, v in raw.items()
             if k.startswith("audio_vae.per_channel_statistics.")}

    encoder = AudioVAEEncoder()
    encoder.load_weights(list(_extract(raw, "audio_vae.encoder.", stats).items()))

    decoder = AudioVAEDecoder()
    decoder.load_weights(list(_extract(raw, "audio_vae.decoder.", stats).items()))

    vocoder = VocoderWithBWE()
    vw = {}
    for key, value in raw.items():
        if not key.startswith("vocoder."):
            continue
        short = key[len("vocoder."):]
        if short.startswith("vocoder."):
            short = short[len("vocoder."):]
        vw[short] = _transpose_vocoder(short, value)
    vocoder.load_weights(list(vw.items()))
    vocoder.upcast_weights_to_fp32()   # fp32 vocoder is required (no autocast in MLX)

    # The exact rate invariant is asserted in tests, not at import/runtime: the
    # resampler is a private third-party attribute, and a pinned-package update
    # must not turn into a hard load failure. Warn instead so a wrong rate label
    # cannot go unnoticed.
    expected_rate = SAMPLE_RATE * getattr(vocoder._resampler, "upsample_factor", 0)
    if OUTPUT_SAMPLE_RATE != expected_rate:
        print(f"[mlx] warning: BWE upsampler gives {expected_rate} Hz, expected {OUTPUT_SAMPLE_RATE}",
              flush=True)

    del raw, vw
    mx.eval(encoder.parameters(), decoder.parameters(), vocoder.parameters())
    mx.clear_cache()
    return encoder, decoder, vocoder


def _check_sample_rate(sr: int) -> int:
    if sr <= 0 or sr > MAX_SAMPLE_RATE:
        raise ValueError(f"unsupported sample rate {sr} Hz (need 0 < sr <= {MAX_SAMPLE_RATE})")
    return sr


def _bounded_frames(max_duration: float, sr: int) -> int:
    """Frames to read: the requested window plus a small (0.1 s or 1%) resampler margin."""
    window = int(math.ceil(max_duration * sr))
    return window + max(int(0.1 * sr), int(0.01 * window))


def load_waveform(path: str, max_duration: float = MAX_REF_DURATION) -> tuple[np.ndarray, int]:
    """Load at most ``max_duration`` seconds of audio as float32 (channels, samples).

    The read is bounded *before* decoding (libsndfile ``frames=`` / PyAV frame loop)
    so an hours-long or maliciously headed file cannot exhaust memory. Unsigned PCM
    is centred as ``(arr - midpoint) / midpoint``; ``PCM_U8`` may still be rejected
    by libsndfile/PyAV and is therefore unsupported.
    """
    if not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError(f"max_duration must be finite and > 0 (got {max_duration})")

    try:
        import soundfile as sf

        info = sf.info(path)
        sr = _check_sample_rate(int(info.samplerate))
        if int(info.channels) < 1:
            raise ValueError(f"no audio channels in {path}")
        frames = _bounded_frames(max_duration, sr)
        if int(info.frames) > 0:
            frames = min(frames, int(info.frames))
        data, sr = sf.read(path, start=0, frames=frames, dtype="float32", always_2d=True)
        data = data.T[..., :int(max_duration * sr)]  # read margin, then trim
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
    except ValueError:
        raise
    except Exception:
        pass  # not libsndfile-readable -> PyAV fallback

    import av

    container = av.open(path)
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"no audio stream in {path}")
        sr = _check_sample_rate(int(stream.rate or 0))
        limit = _bounded_frames(max_duration, sr)
        chunks, total = [], 0
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.dtype.kind in "iu":
                mid = 2 ** (8 * arr.dtype.itemsize - 1)
                unsigned = arr.dtype.kind == "u"
                arr = arr.astype(np.float32)
                arr = (arr - mid) / mid if unsigned else arr / mid
            if not frame.format.is_planar:
                arr = arr.reshape(-1, len(frame.layout.channels)).T
            chunks.append(np.asarray(arr, dtype=np.float32))
            total += arr.shape[-1]
            if total >= limit:
                break
        if not chunks:
            raise ValueError(f"no audio samples decoded from {path}")
        wav = np.concatenate(chunks, axis=-1)[..., :int(max_duration * sr)]
        return wav, sr
    finally:
        container.close()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr

    if x.ndim == 1:
        return soxr.resample(x, sr_in, sr_out)
    return np.stack([soxr.resample(ch, sr_in, sr_out) for ch in x])


def prepare_reference(path: str, duration: float = 10.0) -> mx.array:
    """Load, trim, normalise, tile and resample a voice reference; returns (1, 2, N) @ 16 kHz.

    Only the first two channels are used (no downmix) and mono is duplicated. Short
    source clips are tiled up to ``duration`` rather than rejected. The source length
    does not matter as long as the tiled and resampled 16 kHz target still holds at
    least one STFT window (``N_FFT`` samples, ~64 ms).
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"ref_duration must be finite and > 0 (got {duration})")
    if duration > MAX_REF_DURATION:
        raise ValueError(f"ref_duration {duration} s exceeds the {MAX_REF_DURATION:g} s limit")

    wav, sr = load_waveform(path, duration)
    if wav.size == 0 or wav.shape[-1] == 0:
        raise ValueError(f"voice reference has no samples: {path}")
    if not np.isfinite(wav).all():
        raise ValueError(f"non-finite samples in voice reference: {path}")

    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]  # documented: channels 3..N are dropped, not downmixed

    target = min(int(duration * sr), MAX_REF_SAMPLES)
    if target < 1:
        raise ValueError(f"ref_duration {duration} s is shorter than one sample at {sr} Hz")
    if wav.shape[-1] < target:
        wav = np.tile(wav, (1, math.ceil(target / wav.shape[-1])))
    wav = wav[..., :target].copy()

    peak = float(np.abs(wav).max())
    if peak > 0:
        wav *= 10 ** (-4.0 / 20) / peak

    wav = resample(wav, sr, SAMPLE_RATE)
    if wav.shape[-1] < N_FFT:
        raise ValueError(
            f"voice reference too short: {wav.shape[-1]} samples at {SAMPLE_RATE} Hz "
            f"after tiling/resampling, need >= {N_FFT} (~{N_FFT / SAMPLE_RATE:.3f} s)")
    return mx.array(wav[None], dtype=mx.float32)


@lru_cache(maxsize=1)
def _processor() -> AudioProcessor:
    """Immutable AudioProcessor (mel basis + STFT window); built once."""
    return AudioProcessor(sample_rate=SAMPLE_RATE)


def encode_reference(encoder: AudioVAEEncoder, waveform: mx.array) -> mx.array:
    mel = _processor().waveform_to_mel(waveform)
    latent = encoder.encode(mel.astype(mx.bfloat16))
    mx.eval(latent)
    return latent


def decode_latent(decoder: AudioVAEDecoder, vocoder: VocoderWithBWE, latent: mx.array) -> mx.array:
    """Decode a (B, 8, T, 16) latent to a (B, 2, N) waveform at OUTPUT_SAMPLE_RATE."""
    if (latent.ndim != 4 or latent.shape[1] != LATENT_CHANNELS
            or latent.shape[3] != LATENT_FREQ_BINS):
        raise ValueError(
            f"latent must be (B, {LATENT_CHANNELS}, T, {LATENT_FREQ_BINS}), "
            f"got {tuple(latent.shape)}")
    if latent.shape[2] < 1:
        raise ValueError(f"latent has no frames (T = {latent.shape[2]})")
    mel = decoder.decode(latent.astype(mx.bfloat16))
    wav = vocoder(mel)
    mx.eval(wav)
    return wav
