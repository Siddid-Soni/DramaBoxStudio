# DramaBox on Apple Silicon (MLX)

Native MLX inference port of DramaBox. No CUDA or PyTorch at inference, and no
bitsandbytes. Everything runs on the Apple GPU through `mlx` and `mlx-lm`. PyTorch is only needed
for the optional parity and validation tests.

## What was ported

| Component | Source | MLX implementation |
|---|---|---|
| Audio-only LTX-2.3 DiT (3.3B, 48 blocks) | `dramabox-dit-v1.safetensors` | `src/mlx_dramabox/model.py` (reuses `ltx-core-mlx` attention/AdaLN/FF primitives) |
| Gemma-3 12B text tower (49 hidden states) | `mlx-community/gemma-3-12b-it-4bit` | `src/mlx_dramabox/text.py` |
| Audio text connector + 188160 to 2048 projection | `dramabox-audio-components.safetensors` | `src/mlx_dramabox/text.py` |
| Audio VAE encoder/decoder | same | `src/mlx_dramabox/audio.py` |
| BigVGAN vocoder + 48 kHz BWE | same | `src/mlx_dramabox/audio.py` (uses `VocoderWithBWE` from `ltx-core-mlx`) |
| Scheduler, CFG/STG guidance, Euler loop, reference conditioning, long-form chunking | upstream DramaBox `inference_server.py` (PyTorch reference) | `src/mlx_dramabox/pipeline.py` |

The DiT is numerically validated against the original PyTorch implementation on
the real checkpoint: cosine similarity 0.99977 (bf16 reduction-order noise
only). The vocoder matches PyTorch with cosine 0.9995 on identical mel input.

## Install

This CLI is separate from the Studio app's MPS path. On the Studio layout, pass
`--model-dir models/checkpoints --gemma-dir models/gemma-mlx`. The standalone layout below
uses `models/` and `models/gemma-3-12b-it-4bit`.

```bash
python3 -m venv .venv-mlx          # Python 3.11 to 3.13
.venv-mlx/bin/pip install -r requirements-mlx.txt
```

## Download models

```bash
.venv-mlx/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
DRAMA_REV = "404f967f653fa1170dc15a9d1ddd3fdb9a0a842d"
GEMMA_REV = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
for f in ["config.json", "dramabox-dit-v1.safetensors", "dramabox-audio-components.safetensors"]:
    hf_hub_download("ResembleAI/Dramabox", f, revision=DRAMA_REV, local_dir="models")
snapshot_download("mlx-community/gemma-3-12b-it-4bit", revision=GEMMA_REV,
                  local_dir="models/gemma-3-12b-it-4bit")
PY
```

Total disk: ~16 GB of weights plus ~1.5 GB for the venv. 24 GB of unified memory is recommended.

The process peaks during prompt encoding, when Gemma, the DiT, the codec and the
text connector are all resident. Measured on this M4 Pro / 24 GB: 17.1 GB (bf16, 4 s),
~17.1 to 17.2 GB with a 10 s voice reference, and ~14 GB with `--dit-bits 8`. Streaming the text
connector (chunked projection instead of materialising the whole 49×T×3840 state stack) lowered
the connector's own peak from ~3.25 GB to ~2.74 GB at T=1024 on real hidden states. The
whole-pipeline peak is dominated by the resident weights. The negative prompt is the worst of the
two encoding passes (its hidden states have larger outliers), so it sets the peak. By default the
CLI drops Gemma after prompt encoding, which lowers steady-state RSS but not the peak. Use
`--keep-text-encoder` for a warm server. On a 16 GB Mac, use `--dit-bits 8` and expect some swap.

## Usage

```bash
# prompt only
.venv-mlx/bin/python src/mlx_tts.py \
  --prompt 'A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"' \
  --output output.wav

# with a 10 s voice reference (voice cloning)
.venv-mlx/bin/python src/mlx_tts.py \
  --prompt 'A proud queen speaks with cold fury, "You dare to defy me?"' \
  --voice-sample reference.wav \
  --output output.wav

# explicit duration (seconds) and lower memory
.venv-mlx/bin/python src/mlx_tts.py --prompt '...' --duration 12 --dit-bits 8 -o out.wav

# warm server: keep Gemma resident (higher steady-state memory, lower latency)
.venv-mlx/bin/python src/mlx_tts.py --prompt '...' --dit-bits 8 --keep-text-encoder -o out.wav
```

CLI flags mirror the PyTorch server: `--cfg-scale 2.5`, `--stg-scale 1.5`,
`--steps 30`, `--seed 42`, `--duration-multiplier 1.1`, `--ref-duration 10`.
`--free-text-encoder` is a deprecated no-op (the CLI frees Gemma after prompt encoding by
default); use `--keep-text-encoder` for the warm-server behavior.
Prompts longer than `--max-chunk-duration` (45 s) are automatically split and
stitched with an equal-power crossfade. `--chunk-seed-mode indexed` gives each chunk
different initial noise. The default `same` is reference-identical (equal-length chunks
reuse the same noise stream).

### Python

```python
import sys; sys.path.insert(0, "src")
from mlx_dramabox import DramaBoxMLX

server = DramaBoxMLX(model_dir="models")
server.generate_to_file(
    prompt='A woman speaks warmly, "Hello, how are you today?"',
    output="out.wav",
    voice_ref="reference.wav",   # optional
)
```

## Measured on M4 Pro / 24 GB

| Stage | Time |
|---|---|
| Load all models (warm) | ~7 s |
| Prompt encoding (Gemma + connector) | ~10 to 11 s |
| Denoising, 30 steps × 3 passes, 4 s audio | ~30 to 39 s |
| Denoising, 6 s audio + 10 s voice ref | ~64 to 69 s |
| Decode (VAE + BWE, 6 s) | ~1 to 3 s |

Indicative numbers from one M4 Pro / 24 GB run (no bench script is committed). Generation is
slower than the H100 reference (~2.5 s) but fully on-device. Long-form chunking costs one prompt
encode plus one denoise pass per chunk, so total time scales roughly linearly with the number of
chunks (see the denoising rows above for the per-chunk cost at 4 s and 6 s).

## Validation

The validation snippet needs `mlx-whisper` and `pytest`, which are not runtime
dependencies. They live in `requirements-mlx-validation.txt` and `requirements-mlx-dev.txt`:

```bash
.venv-mlx/bin/pip install -r requirements-mlx-validation.txt
```

`mlx-whisper` pulls in torch through its audio front-end, which is the only reason PyTorch
appears in the MLX environment. The inference path never imports it.

```bash
.venv-mlx/bin/python src/mlx_tts.py --prompt 'A woman speaks warmly, "Hello, how are you today?"' --duration 4 -o /tmp/test.wav
.venv-mlx/bin/python - <<'PY'
import mlx_whisper
print(mlx_whisper.transcribe("/tmp/test.wav", path_or_hf_repo="mlx-community/whisper-small-mlx")["text"])
PY
```

Whisper transcribes the MLX output as "Hello, how are you today?", and the
voice-reference path as "You dare to defy me? I am your queen, and you will kneel."

The same check is automated (opt-in, loads the full pipeline):

```bash
MLX_E2E_WHISPER=1 .venv-mlx/bin/python -m pytest tests/test_mlx_e2e_whisper.py -q
```

The Gemma tower is covered by a hidden-state layout test (49 states; the last is the
post-final-norm output) and by connector parity against real hidden states. A full
per-layer comparison against the HF Gemma-3 12B checkpoint is not part of the default
suite. It needs a separate `transformers==4.51` environment plus ~24 GB for the PyTorch
reference weights. The shipped MLX model also uses a different 4-bit quantizer (MLX affine
g64 vs the reference's bnb-nf4), so exact hidden-state equality is not expected.

## Differences vs. the CUDA server

This port covers the inference pipeline minus the Perth watermark and RE-USE reference
denoising, and it uses no PyTorch at inference.

- No Perth watermark. Perth is PyTorch/CUDA-only, so outputs are unwatermarked.
- No RE-USE reference denoising. `mamba-ssm` and `causal-conv1d` have no
  macOS wheels, so the reference is used as-is. Voice cloning still works without it.
- Text encoder uses `mlx-community/gemma-3-12b-it-4bit` instead of the
  QAT q4_0 checkpoint. It produces the same transcriptions in our tests.
- RNG differs from `torch.Generator`, so the same `--seed` does not
  reproduce the CUDA waveform bit-for-bit.
- Training is still PyTorch/CUDA-only upstream. This port covers inference only.

## License

- The DramaBox checkpoint (`__metadata__["license"]` embeds the LTX-2 Community License
  Agreement), its generated audio and this repository are licensed under the LTX-2 Community
  License (see [`LICENSE`](LICENSE), license date January 5, 2026). In short:
  - Entities with annual revenue of at least $10,000,000 are Commercial Entities and must
    obtain a paid commercial license from the licensor before any commercial use
    (<https://ltx.io/model/licensing>).
  - Use is subject to the Attachment A use restrictions. Among others: no illegal use or
    disinformation, no non-consensual impersonation/deepfakes, no disclosing personal identifiable
    information, no discrimination/harassment, no medical or law-enforcement decisions, no malware,
    and no competing-product use.
  - Redistribution (including derivatives such as fine-tunes) must include a complete copy of
    the agreement, retain the copyright/attribution notices, mark modified files, and propagate all
    use restrictions to downstream users. Machine-generated content should be labelled as such.
- Port code. `src/mlx_dramabox/`, `src/mlx_tts.py` and the other files in this repository are
  provided under the same LTX-2 Community License (`LICENSE`).
- Gemma-3 text encoder. `mlx-community/gemma-3-12b-it-4bit` is a 4-bit MLX conversion of
  Google's Gemma 3 12B. The weights remain subject to the Gemma Terms of Use, and no license
  here relicenses them. Check those terms before redistribution or commercial use.
- Third-party components. `ltx-core-mlx` (pinned fork, see `requirements-mlx.txt`) is MIT.
  That license covers the library only and does not relicense the DramaBox weights, which stay
  under the LTX-2 Community License. `soxr` is LGPL-2.1-or-later: using it as an installed
  Python dependency imposes no obligation on this repository, but redistributing the library or a
  binary that bundles it brings the usual LGPL conditions (license notice, source availability,
  ability to relink). `mlx`, `mlx-lm`, `numpy`, `soundfile` and `av` are MIT/BSD-3-Clause. See
  each package. This is a factual summary, not legal advice; read `LICENSE` and the upstream
  licenses before commercial use.
