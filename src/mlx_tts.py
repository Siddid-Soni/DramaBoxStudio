#!/usr/bin/env python3
"""CLI: DramaBox expressive TTS on Apple Silicon (MLX)."""

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from mlx_dramabox import DramaBoxMLX  # noqa: E402
from mlx_dramabox.pipeline import (MAX_DURATION,  # noqa: E402
                                   validate_duration_policy,
                                   validate_generation_params)


def _env_dit_bits() -> int:
    """Parse DRAMABOX_DIT_BITS defensively so a bad env value cannot break --help."""
    raw = os.environ.get("DRAMABOX_DIT_BITS", "0")
    try:
        bits = int(raw)
    except (TypeError, ValueError):
        bits = 0
    if bits not in (0, 4, 8):
        print(f"[mlx] warning: ignoring DRAMABOX_DIT_BITS={raw!r}; using 0 (bf16)",
              file=sys.stderr)
        return 0
    return bits


def main():
    p = argparse.ArgumentParser(description="DramaBox TTS on MLX")
    p.add_argument("--prompt", required=True)
    p.add_argument("--voice-sample", default=None)
    p.add_argument("-o", "--output", default="output.wav")
    p.add_argument("--model-dir", default=os.environ.get("DRAMABOX_MODELS", str(SRC.parent / "models")))
    p.add_argument("--gemma-dir", default=os.environ.get("DRAMABOX_GEMMA"))
    p.add_argument("--dit-bits", type=int, choices=(0, 4, 8), default=_env_dit_bits(),
                   help="DiT quantization bits: 0 = bf16 (default), 4 = experimental, 8 = quantized")
    p.add_argument("--cfg-scale", type=float, default=2.5,
                   help="classifier-free guidance scale (must be >= 1; 1 disables CFG)")
    p.add_argument("--stg-scale", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=30, help="denoising steps (2..200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, default=0.0,
                   help="explicit duration in seconds (0 = prompt estimate). Must be <= "
                        "--max-chunk-duration; omit it for long-form auto-splitting")
    p.add_argument("--duration-multiplier", type=float, default=1.1)
    p.add_argument("--ref-duration", type=float, default=10.0,
                   help="voice-reference length in seconds (0 < value <= 60)")
    p.add_argument("--rescale", default="auto",
                   help="'auto' or a finite value in [0, 1]")
    p.add_argument("--max-chunk-duration", type=float, default=45.0,
                   help="long-form chunk cap in seconds; inf = never chunk (warns above 60s)")
    p.add_argument("--target-chunk-duration", type=float, default=37.0)
    p.add_argument("--crossfade-ms", type=float, default=50.0)
    p.add_argument("--chunk-seed-mode", choices=("same", "indexed"), default="same",
                   help="long-form RNG: 'same' (reference-identical) or 'indexed' "
                        "(different initial noise per chunk)")
    p.add_argument("--keep-text-encoder", action="store_true",
                   help="keep Gemma-3 12B resident for lower latency (higher steady-state memory)")
    p.add_argument("--free-text-encoder", action="store_true",
                   help="deprecated no-op; the text encoder is dropped after prompt encoding "
                        "unless --keep-text-encoder is given")
    args = p.parse_args()

    # --- validate before constructing any model (P1-15) ---
    if not args.prompt.strip():
        print("[mlx] warning: prompt is empty/whitespace", file=sys.stderr)
    try:
        validate_generation_params(
            cfg_scale=args.cfg_scale,
            stg_scale=args.stg_scale,
            steps=args.steps,
            gen_duration=args.duration,
            duration_multiplier=args.duration_multiplier,
            ref_duration=args.ref_duration,
            rescale_scale=args.rescale,
            max_chunk_duration=args.max_chunk_duration,
            target_chunk_duration=args.target_chunk_duration,
            crossfade_ms=args.crossfade_ms,
        )
        validate_duration_policy(args.prompt, args.duration,
                                 args.max_chunk_duration, args.duration_multiplier)
    except ValueError as exc:
        p.error(str(exc))

    if (args.duration == 0 and args.max_chunk_duration == float("inf")):
        # "never chunk" + a huge auto-estimate would load the models and only
        # then fail inside generate(); reject it before construction.
        from mlx_dramabox.duration_estimator import estimate_speech_duration

        estimate = max(3.0, round(estimate_speech_duration(args.prompt)
                                  * args.duration_multiplier, 1))
        if estimate > MAX_DURATION:
            p.error(f"prompt estimate {estimate:.0f}s exceeds the {MAX_DURATION:.0f}s "
                    "single-shot limit; lower --duration-multiplier or use a finite "
                    "--max-chunk-duration so the prompt can be split")

    if args.voice_sample and not os.path.isfile(args.voice_sample):
        p.error(f"voice sample not found: {args.voice_sample}")
    if not os.path.isdir(args.model_dir):
        p.error(f"model directory not found: {args.model_dir}")
    gemma_dir = args.gemma_dir or os.path.join(args.model_dir, "gemma-3-12b-it-4bit")
    if not os.path.isdir(gemma_dir):
        p.error(f"Gemma directory not found: {gemma_dir}")
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        p.error(f"cannot create output directory for {args.output}: {exc}")

    # CLI default is to drop Gemma after prompt encoding; --keep-text-encoder wins
    # even if the deprecated --free-text-encoder no-op is also present.
    server = DramaBoxMLX(
        model_dir=args.model_dir,
        gemma_dir=args.gemma_dir,
        dit_bits=args.dit_bits,
        keep_text_encoder=args.keep_text_encoder,
    )
    server.generate_to_file(
        prompt=args.prompt,
        output=args.output,
        voice_ref=args.voice_sample,
        max_chunk_duration=args.max_chunk_duration,
        target_chunk_duration=args.target_chunk_duration,
        crossfade_ms=args.crossfade_ms,
        chunk_seed_mode=args.chunk_seed_mode,
        cfg_scale=args.cfg_scale,
        stg_scale=args.stg_scale,
        steps=args.steps,
        seed=args.seed,
        gen_duration=args.duration,
        duration_multiplier=args.duration_multiplier,
        ref_duration=args.ref_duration,
        rescale_scale=args.rescale,
    )


if __name__ == "__main__":
    main()
