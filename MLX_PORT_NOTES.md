> ## Historical, partially superseded
>
> These are the pre-port exploration notes for the PyTorch reference implementation. They are
> kept for traceability and are not a description of the shipped MLX code. The audio codec
> numbers have been corrected below (16 kHz inner vocoder, 3× BWE to 48 kHz, `conv_pre 128→1536`,
> 18 resblocks, `attn_resolutions=[]`, `mid_block_add_attention=False`). The authoritative sources
> are the checkpoint metadata (`models/dramabox-audio-components.safetensors`,
> `__metadata__["config"]`), the installed `ltx_core_mlx.model.audio_vae` package, and
> `src/mlx_dramabox/audio.py`.

> Status: exploration notes. The shipped port is in `src/mlx_dramabox/`
> and `src/mlx_tts.py`, with a working end-to-end CLI (see `README_MLX.md`).
> One correction: the DramaBox checkpoint config sets `rope_type: "split"`
> (not interleaved); `src/mlx_dramabox/model.py:make_rope` implements the split
> variant and is bit-exact against upstream on the real positions.

# DramaBox to Apple MLX: inference path architecture spec

Scope: everything reachable from `TTSServer.generate()` (`src/inference_server.py:307`) and the
CLI mirror `src/inference.py:main()` (`src/inference.py:148`). Line numbers are from the current
tree (`git a70a581`). Paths are relative to `/Users/rsaran/DramaBox`.

Legend:
* **[V]** = verified directly in code (line reference given).
* **[C]** = comes from the checkpoint's safetensors metadata `config` dict, so it must be read at
  runtime; the numbers shown are the ones consistent with the shipped DramaBox weights
  (see the parameter-count cross-check in §2.9).
* **[?]** = ambiguous in this repo (config key lives only in the checkpoint); flagged for
  verification before/while porting.

---

## 0. Summary

DramaBox is LTX-2.3's audio half only: an `LTXModel(model_type=AudioOnly)` "DiT" (3.25 B params,
bf16), a Gemma-3-12B text tower (4-bit NF4 via bitsandbytes) plus a 1-D "embeddings connector",
and a causal 2-D audio VAE decoder plus a BigVGAN-v2 vocoder with a bandwidth-extension (BWE)
stage. There is no video branch at inference time. Video modules are either not instantiated
(`AudioOnly`) or deleted and patched out (`PromptEncoder(audio_only=True)`).

The numeric core is small and ports to MLX without special ops:

| Stage | params | dtype | MLX difficulty |
|---|---|---|---|
| Gemma-3-12B (hidden states only) | 12.2 B (NF4-packed) | bf16 compute | hard (bnb NF4 needs dequant or a new checkpoint) |
| `FeatureExtractorV2.audio_aggregate_embed` | 385 M | bf16 | trivial (big Linear) |
| `audio_connector` (`Embeddings1DConnector`) | ~101 M | bf16 | easy (2 transformer blocks, same code as DiT) |
| audio DiT (`LTXModel`, 48 blocks) | 3.25 B | bf16 | easy (no exotic ops once fp8/compile are dropped) |
| audio VAE decoder | 31.9 M | bf16 | easy (causal Conv2d / GroupNorm / PixelNorm) |
| vocoder + BWE | ~128 M (source comment, `vocoder.py:571`) | fp32 | medium (108 Conv1d + SnakeBeta + sinc resample) |

The only real blockers are (§5) bitsandbytes NF4 Gemma, `torch.compile` and the
`torch.Generator`/CUDA-RNG based noise. Everything else is standard ops plus one custom RoPE
variant.

---

## 1. End-to-end inference pipeline

### 1.1 Call sequence (exact)

```
TTSServer.__init__                                  src/inference_server.py:119
 ├─ PromptEncoder(..., warm=True, use_bnb_4bit=True, audio_only=True)   :155  (blocks.py:312)
 │    ├─ GemmaTextEncoder (Gemma3ForConditionalGeneration, bnb-4bit)   blocks.py:437-488
 │    └─ EmbeddingsProcessor, video parts stripped + create_embeddings monkey-patched (blocks.py:375-435)
 ├─ AudioConditioner(..., warm=True)   :167  → AudioEncoder (VAE encoder, for voice ref)
 ├─ LTXModel(model_type=AudioOnly) from checkpoint metadata `config`   :176-221
 ├─ builder.build(...).to(device).eval()                               :212-218
 ├─ torch.compile(model, mode="default", dynamic=True)                 :227
 └─ AudioDecoder(..., warm=True)       :232  → AudioVAE decoder + Vocoder(VocoderWithBWE)

TTSServer.generate()                                 src/inference_server.py:307  (@torch.inference_mode at :306)
 1  gen_dur  = gen_duration or estimate_duration(prompt, multiplier)   :326-329
 2  fps=25.0; n_frames = round(gen_dur*25)+1; n_frames = ((n_frames-1+4)//8)*8+1   :330-332
 3  pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=25)  :333
 4  target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)  :334
 5  audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), target_shape)  :335, :134
 6  state = audio_tools.create_initial_state(device, dtype)             :338
 7  (optional) voice ref: decode → denoise(RE-USE, optional) → VAE encode →
      AudioConditionByReferenceLatent.apply_to(state)                   :341-364
 8  state = GaussianNoiser(gen(seed))(state, noise_scale=1.0)           :367-370
 9  prompts = [prompt, DEFAULT_NEG] if cfg>1                            :373
10  ctx = PromptEncoder(prompts)  → ctx[i].audio_encoding               :374-377
11  guider = MultiModalGuider(cfg, stg, stg_blocks=[29], rescale, modality=1.0)  :383-389
12  denoiser = GuidedDenoiser(v_context=None, a_context=..., audio_guider=guider) :390-394
13  sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent)     :396
14  x0 = X0Model(self._velocity_model)                                  :399
15  _, audio_state = euler_denoising_loop(sigmas, None, state, EulerDiffusionStep(), x0, denoiser)  :401-404
16  audio_state = audio_tools.clear_conditioning(audio_state)           :408
17  audio_state = audio_tools.unpatchify(audio_state)                   :409
18  latent[:, :, 512:514] ← linear interp(511, 514) if T > 513          :420-430
19  decoded = AudioDecoder(latent)  → VAE decode → vocoder(+BWE)        :432
```

All random and latent state is patchified from step 6 onward (see §1.3).

### 1.2 Duration to latent length math **[V]**

```
frames_latents_per_sec = sample_rate / hop_length / latent_downsample = 16000/160/4 = 25.0
n_frames (video "frames", dummy)   = round(gen_dur*fps)+1 ; fps = 25
                                   → then (n_frames-1) is snapped to the NEAREST multiple of 8:
                                     n_frames = ((n_frames-1+4)//8)*8 + 1
AudioLatentShape.from_video_pixel_shape:  T = round((n_frames/fps) * 25) == n_frames
```
(`src/inference_server.py:330-334`, `ltx2/ltx_core/types.py:128-164`)

Concrete values:

| gen_dur | n_frames == latent T | mel frames = 4T-3 | samples @48 k | out secs @48 k |
|---|---|---|---|---|
| 10 s | 249 | 993 | 476,640 | 9.93 |
| 30 s | 753 | 3,009 | 1,444,320 | 30.09 |
| 33 s | 825 | 3,297 | 1,582,560 | 32.97 |

`height=width=64` are placeholders. They only exist so `VideoPixelShape` has valid fields.

### 1.3 `AudioLatentShape`, patchifier, `AudioLatentTools` **[V]**

`AudioLatentShape(batch, channels, frames, mel_bins) = (B, 8, T, 16)` (`types.py:98-164`).
`channels=8`, `mel_bins=16` are hardcoded defaults of `from_video_pixel_shape`.

`AudioPatchifier(patch_size=1, sample_rate=16000, hop_length=160,
audio_latent_downsample_factor=4, is_causal=True, shift=0)` (`components/patchifiers.py:169-201`):

* `patch_size = (1, 1, 1)`, so patchification is a pure reshape:
  `patchify: (b, c, t, f) → (b, t, c*f)` = `(B, 8, T, 16) → (B, T, 128)` (`patchifiers.py:282-301`).
  `unpatchify: "b t (c f) -> b c t f"` with `c=8, f=16` (`patchifiers.py:303-327`).
* `get_token_count = T` (one token per latent frame).
* `get_patch_grid_bounds` returns timestamps in seconds, shape `(B, 1, T, 2)` (`patchifiers.py:329-348`):

```
frame(i)  = i*4 ; if causal: frame = max(frame + 1 - 4, 0)
t(i)      = frame(i) * 160 / 16000   = max(4i-3, 0)/100     seconds
bounds    = [ t(shift..shift+T), t(shift+1..shift+T+1) ]    # [start, end)
```
So token *i* spans `[max(4i-3,0)/100, (4i+1)/100)` s, which is 40 ms per token, with the first token
special-cased to `t=0`.

`AudioLatentTools(patchifier, target_shape)` is a frozen dataclass (`ltx_core/tools.py:147-190`):

| step | output |
|---|---|
| `create_initial_state(device, dtype)` | `latent=zeros(B,8,T,16)`, `clean_latent=zeros`, `denoise_mask=ones(B,1,T,1)`, `positions=get_patch_grid_bounds(...)` (float32, not cast to dtype), then `patchify()` |
| after patchify | `latent (B,T,128) bf16`, `clean_latent (B,T,128) bf16`, `denoise_mask (B,T,1) f32`, `positions (B,1,T,2) f32`, `attention_mask=None` |
| `clear_conditioning(state)` | truncate tokens to `T` (`patchifier.get_token_count`), `denoise_mask = ones`, `attention_mask = None` (`tools.py:66-85`) |
| `unpatchify(state)` | `(B,T,128) → (B,8,T,16)`; mask unpatchified with `mask_shape() = (B,1,T,1)` |

`LatentState` = `{latent, denoise_mask, positions, clean_latent, attention_mask}` (`types.py:183-209`).

Note for the port: `positions` stay fp32 while latents are bf16. The RoPE table is built in
fp32 then cast to the latent dtype (`transformer_args.py:_prepare_positional_embeddings`), so this
does not change numerics, but the reference's mixed precision is worth mirroring.

### 1.4 Voice-reference conditioning (`AudioConditionByReferenceLatent`) **[V]**

`src/audio_conditioning.py:34-115`. It appends ref tokens at the end of the sequence:

1. `decode_audio_from_file(ref, device, 0.0, ref_duration)` → `Audio(waveform (1,C,S) f32, sr=file_sr)`
   (`ltx_pipelines/utils/media_io.py:301-361`, PyAV; `C` = 1 or 2, mono is broadcast to 2 at
   `inference_server.py:346-352`).
2. Trim/pad to `ref_duration * file_sr`, peak-normalise to -4 dBFS (`inference_server.py:353-359`).
3. Optional RE-USE denoise of the reference (`inference_server.py:239-304`, RE-USE at `:266-269`).
   It is a separate NVIDIA Mamba model, not part of the LTX graph (`src/super_resolution.py`).
   Optional, and can be dropped in the MLX port (`denoise_ref=False`) with a quality delta.
4. `encode_audio(voice, AudioEncoder)` → `(1, 8, T_ref, 16)` = `round(10 s × 25) = 250` frames
   (`audio_vae/audio_vae.py:248-273`; the encoder resamples 48 k → 16 k internally).
5. `tokens = patchify(ref_latent) → (1, T_ref, 128)`
6. `positions_ref = get_patch_grid_bounds(ref_shape) + 0.5`  ← 0.5 s offset (`audio_conditioning.py:57`)
7. `denoise_mask_ref = full(1 - strength) = 0` → frozen
8. Asymmetric self-attention mask `(B, T+T_ref, T+T_ref)` float32 in `[0,1]`
   (`audio_conditioning.py:88-107`):

```
              target(N)   ref(M)
 target(N)  [    1.0        1.0  ]
 ref(M)     [    0.0        1.0  ]      # ref is read-only conditioning
```
9. `latent = cat([target, ref], dim=1)`, `positions = cat([target, ref], dim=2)`,
   `clean_latent = cat([target, ref_tokens], dim=1)`.

Total sequence `T_total = T + T_ref` (e.g. 825 + 250 = 1075).

### 1.5 Noise injection **[V]**

`GaussianNoiser` (`components/noisers.py:15-35`), called with `noise_scale=1.0`:

```
noise   = randn(latent.shape, dtype=latent.dtype=bf16, generator=torch.Generator("cuda").manual_seed(seed))
latent  = noise*denoise_mask*1.0 + latent*(1 - denoise_mask)
```
Target tokens: mask 1 → `latent = noise` (initial latent was zeros).
Ref tokens: mask 0 → ref latent preserved. Noise is drawn in bf16.

### 1.6 Text conditioning **[V]**

`prompts = [prompt, DEFAULT_NEG]` when `cfg_scale > 1.0` (`inference_server.py:373`),
`DEFAULT_NEG = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear,
unnatural, monotone"` (`inference_server.py:51`). Produces `a_ctx, a_ctx_neg` of shape
`(1, 1024, 2048)` bf16 (see §4). No `context_mask` is passed downstream
(`modality_from_latent_state(..., context_mask=None)`, `ltx_pipelines/utils/helpers.py:257-276`).

### 1.7 Scheduler: `LTX2Scheduler` **[V]**

`components/schedulers.py:14-57`, called as
`sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(device)`.

```
tokens = prod(latent.shape[2:])          # ← latent is PATCHIFIED (B,T,128) ⇒ tokens = 128 (!)
sigmas = linspace(1, 0, steps+1)         # 31 values
m  = (2.05 - 0.95)/(4096-1024);  b = 0.95 - m*1024
sigma_shift = tokens*m + b  = 128*0.000358 + 0.5837 = 0.62917      ⇒ exp(σ̂) = 1.87605
s'_i = exp(σ̂) / (exp(σ̂) + 1/s_i - 1)      (s_i == 0 → 0)
# stretch (terminal=0.1):
scale = (1 - s'_min_nonzero) / (1 - 0.1)
s''_i = 1 - (1 - s'_i)/scale             (only for nonzero entries)
```
Because the latent is patchified, `tokens` is 128 regardless of duration, so the schedule is a
fixed, fairly mild shift and does not depend on duration. (If it were computed on the unpatchified
latent it would be `8*T*16` = 16 400+ and the schedule would be much sharper.) Reproduce this quirk
exactly for parity.

Resulting 31 sigmas: `1.0, 0.9827, 0.9649, 0.9464, …, 0.1000, 0.0`, so the last Euler step goes
from σ = 0.1 straight to 0.

### 1.8 Guided denoising: CFG + STG, batched into one forward **[V]**

`GuidedDenoiser` → `_guided_denoise` (`ltx_pipelines/utils/denoisers.py:56-175`). With the
TTSServer defaults (`cfg=2.5, stg=1.5, stg_blocks=[29], modality_scale=1.0`) exactly three
passes are built:

| pass | name | video ctx | audio ctx | perturbations |
|---|---|---|---|---|
| 0 | `cond` | - | `a_ctx` | none |
| 1 | `uncond` | - | `a_ctx_neg` | none |
| 2 | `ptb` | - | `a_ctx` | `SKIP_AUDIO_SELF_ATTN` in block 29 |

(`mod` pass is skipped: `modality_scale == 1.0`; `skip_step` is 0 so no step skipping.)
All three are concatenated into one batch of 3, and that batch is repeated inside
`BatchSplitAdapter` if `max_batch_size > 1` (TTSServer calls `X0Model` directly without the adapter,
so batch 3 goes in one forward pass, `inference_server.py:399-404`).

Batched modality construction (`denoisers.py:135-157` + `helpers.py:257-276`):

```
state repeated 3×:   latent (3,T_total,128), denoise_mask (3,T_total,1),
                     positions (3,1,T_total,2), attention_mask (3,T_total,T_total)
context = cat([a_ctx, a_ctx_neg, a_ctx])            → (3,1024,2048)
sigma   = sigmas[step].expand(3)                     → (3,)
timesteps = denoise_mask * sigma                     → (3,T_total)   (0 for ref tokens)
```

Output combination (`denoisers.py:163-175`, `components/guiders.py:246-287`):

```
pred = cond + (cfg_scale-1)*(cond - uncond_text) + stg_scale*(cond - uncond_perturbed)
if rescale_scale != 0:                      # TTSServer: "auto" → auto_rescale_for_cfg(cfg)
    factor = cond.std()/pred.std(); factor = rescale*factor + (1-rescale)
    pred  *= factor
if cfg_clamp_scale > 0: <per-token norm clamp>            # 0 by default
```
`auto_rescale_for_cfg` (`src/inference_server.py:85-119`) piecewise-linear:
`cfg≤2 → 0`; `2..3 → 0.6(cfg-2)`; `3..4 → 0.6+0.2(cfg-3)`; `4..8 → 0.8`; `>8 → min(1, 0.8+0.1(cfg-8))`.
With cfg=2.5 the rescale is 0.3.

`cond.std()` / `pred.std()` are global (over all elements of the `(1,T_total,128)` slice) in bf16.

STG implementation detail (important for the port). Perturbation is per batch element, not
per pass, so `components/guidance/perturbations.py` builds a `(3,1,1)` mask, `[1,1,0]`, and
`BasicAVTransformerBlock` runs a full attention for all 3 samples, then blends
(`transformer.py:250-272`, `attention.py` `perturbation_mask` branch):

```
a_mask = perturbations.mask_like(SKIP_AUDIO_SELF_ATTN, 29, ax)   # (3,1,1)
ax = ax + audio_attn1(norm_ax, pe, mask, perturbation_mask=a_mask, all_perturbed=False) * agate
# inside Attention.forward:
#   out = attention(q,k,v); out = out*a_mask + v*(1-a_mask)     → v = to_v(context) only
```
So for the perturbed sample in block 29 the self-attention output is replaced by the raw value
projection (`to_v(rms_norm_free x)`). Note that `v` is computed from the un-modulated `x`
(`attention.py:220`: `v = self.to_v(context)` where `context = x` for self-attention), not from
`norm_ax`. `all_perturbed` (skip q/k entirely) is only true when every batch element is perturbed.

### 1.9 Euler step **[V]**

`euler_denoising_loop` (`ltx_pipelines/utils/samplers.py:34-74`), `EulerDiffusionStep`
(`components/diffusion_steps.py:7-22`), 30 iterations:

```
# X0Model wrapper (model.py:461-486)
denoised = to_denoised(latent, v_pred, timesteps)      # = latent - v*sigma_per_token
          = (latent.f32 - v.f32 * timestep).to(dtype)  # timestep is per-token (B,T)!
# post_process_latent (helpers.py:252-254)
denoised = denoised*denoise_mask + clean_latent*(1-denoise_mask)
# EulerDiffusionStep
velocity = (latent - denoised)/sigma                   # fp32 math, cast back
latent   = (latent + velocity*dt)  where dt = sigma_next - sigma
```
Because `timesteps = denoise_mask*sigma`, frozen ref tokens get `denoised = latent` (v×0) and are
then overwritten by `clean_latent` in `post_process_latent`, so the ref latent is exactly preserved.

`to_velocity` raises if `sigma == 0`, and the loop runs over `sigmas[:-1]` (30 steps), so σ=0 is never
used as the current sigma.

### 1.10 Post-processing before decode **[V]**

* `clear_conditioning` drops the ref tokens → `(B,T,128)`, mask = ones, `attention_mask = None`.
* `unpatchify` → `(1, 8, T, 16)` bf16.
* "silence-prior" latent repair (`inference_server.py:411-430`, mirrored at
  `src/inference.py:481-495`): if `T > 513`, frames 512 and 513 are linearly interpolated between
  neighbours 511 and 514. Rationale in the comment: the base model learned a clip-end silence
  prior on a 20 s grid; frame 513 = 8·64+1 = 20.52 s and the latent there has an abnormally high norm
  that the decoder renders as a ~30 ms hard silence dip. Interpolation is a latent-space
  post-process and must be replicated (it is a 2-frame, 128-channel lerp):

```
f0, f1 = 511, 514 ; n = 3
for f in (512, 513):  t = (f-511)/3 ; lat[:, :, f, :] = (1-t)*lat[:,:,511,:] + t*lat[:,:,514,:]
```
(`assets/silence_latent_frame.pt` is a training asset, a real silence latent frame prepended as
data augmentation in `src/train.py:661-690`. It is not used at inference.)

### 1.11 Decode to waveform **[V]**

`AudioDecoder(latent)` → `blocks.py:660-708` (`warm=True` keeps both sub-models resident) →
`audio_vae.decode_audio` (`audio_vae/audio_vae.py:496-508`):

```
mel = self._warm_decoder(latent)              # (1,8,T,16) → (1, 2, 4T-3, 64)  log-mel, bf16
wav = self._warm_vocoder(mel)                 # VocoderWithBWE → (1, 2, S48) fp32, clamped [-1,1]
return Audio(waveform=wav.squeeze(0).float(), sampling_rate=48000)   # → (2, S48)
```
Then `generate_to_file` moves to CPU float32, optionally applies the Perth watermark (numpy/ONNX,
keep off the GPU) and `torchaudio.save`.

### 1.12 Concrete tensor-shape trace (30 s, 10 s ref, cfg 2.5, stg 1.5)

| # | tensor | shape | dtype |
|---|---|---|---|
| 1 | `state.latent` after initial+patchify | `(1, 753, 128)` | bf16 |
| 2 | ref latent `(1,8,250,16)` → tokens | `(1, 250, 128)` | bf16 |
| 3 | latent after conditioning | `(1, 1003, 128)` | bf16 |
| 4 | `denoise_mask` | `(1, 1003, 1)` | f32 |
| 5 | `positions` | `(1, 1, 1003, 2)` | f32 |
| 6 | `attention_mask` | `(1, 1003, 1003)` | f32 `{0,1}` |
| 7 | `a_ctx` | `(1, 1024, 2048)` | bf16 |
| 8 | batched modality latent / timesteps / positions / mask | `(3,1003,128)` / `(3,1003)` / `(3,1,1003,2)` / `(3,1003,1003)` | bf16 / bf16 / f32 / f32 |
| 9 | per forward: `x` inside blocks | `(3, 1003, 2048)` | bf16 |
| 10 | self-attn bias | `(3, 1, 1003, 1003)` | bf16 |
| 11 | output velocity | `(3, 1003, 128)` | bf16 |
| 12 | decoded mel | `(1, 2, 3009, 64)` | bf16→fp32 |
| 13 | waveform | `(2, 1,444,320)` @ 48 kHz | f32 |

Memory for the DiT activations: `3×1003×2048×2 B ≈ 12 MB` per tensor, and the FF
intermediate is `3×1003×8192×2 B ≈ 49 MB`. The 6.5 GB of weights dominate, so everything fits in
unified memory.

---

## 2. Transformer architecture (`ltx2/ltx_core/model/transformer/`)

### 2.1 Configuration actually used **[C]/[V]**

Built by the local `AudioOnlyConfigurator` (`src/inference_server.py:181-210`) rather than
`LTXModelConfigurator`, so `VideoOnly`-only subsystems are never created:

```python
LTXModel(
  model_type = LTXModelType.AudioOnly,
  audio_num_attention_heads = 32,          # t["audio_num_attention_heads"]
  audio_attention_head_dim  = 64,          # → audio_inner_dim = 2048
  audio_in_channels         = 128,         # = 8 × 16 latent features
  audio_out_channels        = 128,
  num_layers                = 48,
  audio_cross_attention_dim = 2048,        # text context width
  norm_eps                  = 1e-6,
  attention_type            = "default" → xformers if installed else SDPA,
  positional_embedding_theta= 10000.0,     # hardcoded
  audio_positional_embedding_max_pos = [20.0],   # hardcoded
  timestep_scale_multiplier = 1000,
  use_middle_indices_grid   = True,
  rope_type                 = "split",
  double_precision_rope     = True,        # t["frequencies_precision"] == "float64"
  apply_gated_attention     = True,        # config value
  audio_caption_projection  = None if t["caption_proj_before_connector"] else PixArtAlphaTextProjection,
  cross_attention_adaln     = True,        # config value
)
```
`num_attention_heads/attention_head_dim/cross_attention_dim` (video) are not passed, so their
defaults are used but never instantiated, because `is_video_enabled()` is False (`model.py:24-29`).

### 2.2 Module inventory & parameter shapes **[V]** (audio only)

Model level (`LTXModel._init_audio`, `model.py:150-176`):

| module | shape | notes |
|---|---|---|
| `audio_patchify_proj` | `Linear(128 → 2048, bias)` | `(3,T,128) → (3,T,2048)` |
| `audio_adaln_single` | below | timestep → 6×2048 modulation |
| `audio_scale_shift_table` | `Parameter(2, 2048)` | output head modulation |
| `audio_norm_out` | `LayerNorm(2048, elementwise_affine=False, eps=1e-6)` | |
| `audio_proj_out` | `Linear(2048 → 128)` | |
| `audio_prompt_adaln_single` | `None` unless `cross_attention_adaln` | |
| `transformer_blocks` | 48 × `BasicAVTransformerBlock` | |

`AdaLayerNormSingle(2048, embedding_coefficient=6)` (`transformer/adaln.py:19-45`):

```
PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=2048, size_emb_dim=2048//3=682)
  time_proj   = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
  timestep_embedder = TimestepEmbedding(256 → 2048):  Linear(256,2048) → SiLU → Linear(2048,2048)
linear = Linear(2048 → 6*2048 = 12288)          applied to SiLU(emb)
returns (linear(silu(emb))            -> (N, 12288)   # modulation  (reshape to (B,T,6,2048))
         emb                          -> (N, 2048))   # embedded_timestep (output head)
```
(`size_emb_dim` is stored but unused; `timestep_embedding.py:118-143`.)

Timestep embedding math (`timestep_embedding.py:6-54`) **[V]**, `half_dim=128`:

```
exponent = -ln(10000) * arange(128) / (128 - 0)
emb      = exp(exponent);  emb = (t * emb)
emb      = cat([sin(emb), cos(emb)], -1)         # 256
flip_sin_to_cos=True → emb = cat([emb[:,128:], emb[:,:128]])   # cos first, then sin
```
`t` fed in is `timesteps * 1000` (`transformer_args.py:_prepare_timestep`), i.e. `sigma*1000` per
token (0 for ref tokens).

Per block (`BasicAVTransformerBlock.__init__`, `transformer.py:24-124`), audio branch only:

| module | shape |
|---|---|
| `audio_attn1` (self, `context_dim=None`) | `to_q Linear(2048,2048,b)`, `to_k Linear(2048,2048,b)`, `to_v Linear(2048,2048,b)`, `to_out.0 Linear(2048,2048,b)`, `q_norm RMSNorm(2048)`, `k_norm RMSNorm(2048)` |
| `audio_attn2` (text cross, `context_dim=2048`) | same shapes (k/v take 2048-wide context) |
| `audio_ff` | `net.0.proj Linear(2048, 8192)`, GELU(tanh), `net.2 Linear(8192, 2048)` |
| `audio_scale_shift_table` | `Parameter(6, 2048)` (9 if `cross_attention_adaln`) |
| `audio_prompt_scale_shift_table` | `Parameter(2, 2048)` only if `cross_attention_adaln` |

No `audio_to_video_attn` / `video_to_audio_attn` (requires both modalities, `transformer.py:88-121`).
No `to_gate_logits` if `apply_gated_attention=False`.

### 2.3 Preprocessing per forward (audio-only path) **[V]**

`TransformerArgsPreprocessor.prepare` (`transformer_args.py:44-190`). For `AudioOnly` the simple
preprocessor is used, not `MultiModalTransformerArgsPreprocessor`, so
`cross_positional_embeddings = cross_scale_shift_timestep = cross_gate_timestep = None`
(`model.py:260-278`).

```
x                    = audio_patchify_proj(latent)                    # (B,T,128)→(B,T,2048)
timestep, emb_t      = adaln((timesteps*1000).flatten())              # (B,T,12288), (B,T,2048)
context              = caption_projection(context) if not None else context
                       then .view(B, -1, 2048)                        # (B,1024,2048) unchanged
context_mask         = None  (input None)                             # no text masking
pe                   = precompute_freqs_cis(positions, ...)           # (§2.4)
self_attention_mask  = None or additive bias from [0,1] mask          # (§2.5)
```

### 2.4 RoPE: split, applied to the full inner dim, shared across heads [V]

`transformer/rope.py`. This is the most error-prone part to port.

```
rope_type = SPLIT (config value; interleaved is the legacy default)
dim       = audio_inner_dim = 2048
n_pos_dims= positions.shape[1] = 1               (audio: time only)
n_freq    = inner_dim // (2 * n_pos_dims) = 1024
```

Frequency grid (`generate_freq_grid_pytorch`, `rope.py:90-111`). This is not the standard
`theta^(-2i/d)`; the exponents are increasing:

```
grid[j] = 10000 ** (j / (n_freq - 1)) * (pi/2)     j = 0 … 1023       # 1.5708 … 15708.0
```
(because `torch.linspace(log_θ(1), log_θ(θ), n_freq)` = `linspace(0, 1, 1024)`; the `numpy` variant
`generate_freq_grid_np`, used when `frequencies_precision == "float64"`, is identical up to fp
rounding).

Angles (`generate_freqs`, `rope.py:126-141`), with `use_middle_indices_grid=True`:

```
mid  = (positions[...,0] + positions[...,1]) / 2          # seconds, (B,1,T)
frac = mid / max_pos                                      # max_pos = [20.0]
angle[b,t,j] = grid[j] * (2*frac[b,t] - 1)
```
Since `mid ∈ [0, T/25]`, the factor `(2·mid/20 - 1) ∈ [-1, 1]`, so it is negative for the first
10 s, zero at 10 s, and positive afterwards. Ports must not clamp or normalise this away.

Application (`split_freqs_cis` + `apply_split_rotary_emb`, `rope.py:29-39, 167-175`):

```
cos, sin = cos(angle), sin(angle)               # (B,T,1024); pad_size = 0
x1, x2   = x[..., :1024], x[..., 1024:]
out      = cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```
Applied to q and k (after `q_norm`/`k_norm`, before attention) and to the whole 2048-vector.
`cos/sin` have no head dimension, so all 32 heads share the same rotation.

Not applied to text cross-attention (`audio_attn2` is called without `pe`/`k_pe`,
`transformer.py:158-185`). Cross-attention has no positional encoding at all.

`SPLIT` variant (unused here, for completeness, `rope.py:42-66, 144-164`): per-head reshape
`(B,T,2048) → (B,H,T,64)`, pairs are `(i, i+32)` inside each head, cos/sin shaped `(B,H,T,32)`.

Porting note: `precompute_freqs_cis` is recomputed on every forward pass (~48× redundancy per
step, and the same table for all 30 steps since positions never change). Cache
`(cos2, sin2)` per request; the whole table is only `3×1003×2048×2 B ≈ 12 MB`.

### 2.5 Attention module (`Attention`, `attention.py:142-252`) **[V]**

```
inner_dim = heads*dim_head = 2048 ; context_dim default = query_dim
q = to_q(x);  q = q_norm(q)             # RMSNorm over the FULL 2048 (learnable weight, eps=1e-6)
k = to_k(ctx); k = k_norm(k)
v = to_v(ctx)                           # NOTE: ctx is the *un-normalised* input
q = rope(q); k = rope(k)                # (§2.4)
out = attention_func(q, k, v, heads, mask)          # (B,T,2048)
if perturbation_mask is not None: out = out*mask + v*(1-mask)
if to_gate_logits is not None:                       # apply_gated_attention
    gates = 2*sigmoid(to_gate_logits(x))             # (B,T,32)
    out   = out.view(B,T,32,64) * gates[...,None] → (B,T,2048)
return to_out(out)                      # Linear(2048,2048) + Identity
```
* QKV layout: three independent `Linear(·, 2048)`; head `h` owns the contiguous slice
  `[h*64 : (h+1)*64]` (`t.view(b,-1,heads,dim_head).transpose(1,2)`, `attention.py:33-35`).
* Scaling: `F.scaled_dot_product_attention` default `1/sqrt(64) = 0.125` (`attention.py:44`).
* `q_norm`/`k_norm` are `torch.nn.RMSNorm` with a learnable per-element weight of width 2048
  (not per-head, not weightless). Checkpoint keys: `…audio_attn1.q_norm.weight`, `.k_norm.weight`.
* Per-head gating (if enabled) `2σ(Wx)` so zero-init logits ⇒ identity. **[?]** confirm from
  config; the DramaBox DiT is presumably `apply_gated_attention=false` (13% param overhead otherwise).
* Attention masks used: only the self-attention additive bias from `attention_mask`
  (`transformer_args.py:_prepare_self_attention_mask`, `:108-138`):

```
bias = finfo(dtype).min  where mask <= 0 ;  log(mask.clamp(min=tiny))  where mask > 0
shape (B, 1, T, T), dtype = latent dtype (bf16)   → finfo(bf16).min = -3.39e38
```
This bias is constant across the 30 steps, so precompute it once per request.
`context_mask`/text mask is None ⇒ cross-attention is unmasked.

### 2.6 Block forward order (audio-only) **[V]**

`BasicAVTransformerBlock.forward`, `transformer.py:187-377`; `run_vx = run_a2v = run_v2a = False`:

```
ashift_msa, ascale_msa, agate_msa = ada(audio_sst, slice(0,3))       # audio.timesteps
norm_ax = rms_norm(ax) * (1 + ascale_msa) + ashift_msa               # weightless RMSNorm, eps=1e-6
ax = ax + audio_attn1(norm_ax, pe=audio_pe, mask=self_attn_bias,
                      perturbation_mask=a_mask) * agate_msa          # timestep-modulated *gate*
ax = ax + audio_attn2(rms_norm(ax))                                  # text cross-attn, NO modulation
ashift_mlp, ascale_mlp, agate_mlp = ada(audio_sst, slice(3,6))
ax = ax + audio_ff(rms_norm(ax)*(1+ascale_mlp)+ashift_mlp) * agate_mlp
```
`get_ada_values` (`transformer.py:126-135`): `table[idx][None,None] + timestep.view(B,T,6,2048)[:,:,idx,:]`,
a per-token modulation tensor, with all 6 values produced by the single `audio_adaln_single`
linear. All values are added (`1 + scale`, `+ shift`).

If `cross_attention_adaln=True` the block instead does (`transformer.py:379-400`):
`q_shift, q_scale, gate = ada(slice(6,9))`; `x_q = rms_norm(x)*(1+q_scale)+q_shift`;
`k/v = context*(1+scale_kv)+shift_kv` where `scale_kv/shift_kv` come from
`audio_prompt_scale_shift_table[None,None] + prompt_timestep.view(B,T,2,-1)`; and
`out = attn(x_q, k/v)*gate`. Plus an extra `audio_prompt_adaln_single` `Linear(2048→2*2048)` fed by
`modality.sigma`. **[?]** verify `cross_attention_adaln` from the checkpoint config: it changes the
block `scale_shift_table` shape from `(6,2048)` to `(9,2048)` and adds two modules, so a wrong guess
fails loudly at weight load.

### 2.7 Output head **[V]**

`LTXModel._process_output` (`model.py:369-387`):

```
ss   = audio_scale_shift_table[None,None] + embedded_timestep[:, :, None]   # (B,T,2,2048)
shift, scale = ss[...,0], ss[...,1]
ax   = audio_proj_out( audio_norm_out(ax) * (1+scale) + shift )              # (B,T,128)
```
`audio_norm_out` has no affine parameters. Because `embedded_timestep` is per-token, the output
scale/shift vary per token.

`X0Model` then converts velocity to denoised (`model.py:472-486`).

### 2.8 Non-standard ops / backend dependencies in the transformer

| op | where | port action |
|---|---|---|
| `torch.nn.functional.scaled_dot_product_attention` | `attention.py:44` | `mx.fast.scaled_dot_product_attention` + additive mask |
| xformers `memory_efficient_attention` | `attention.py:7-11, 50-95`, selected by `AttentionFunction.DEFAULT` when installed | drop; single MLX path. Note xformers pads the mask to a multiple of 8 and casts q/k to v dtype |
| FlashAttention-3 `flash_attn_interface.flash_attn_func` | `attention.py:15-19, 97-120` (unmasked only) | drop |
| fp8 `cublas_scaled_mm` / `static_quantize_e4m3_per_tensor` (TensorRT-LLM) | `quantization/fp8_scaled_mm.py:44, 51-58` | not used for this bf16 checkpoint; ignore |
| Triton `fused_add_round_kernel` | `loader/kernels.py:1-40`, `quantization/fp8_cast.py:10-40` | only for fp8 cast with stochastic rounding / LoRA fusion, not on the TTS path |
| `torch.compile` on the whole model (inductor) | `src/inference_server.py:227`; also `model/transformer/compiling.py:9, 11-22` (per-block variant used by `DiffusionStage(torch_compile=True)`) | drop; optionally `mx.compile` the block |
| `torch.utils.checkpoint` | `model.py:339-366` (training only) | ignore |
| `einops.rearrange` in RoPE | `rope.py:32-35` | pure reshape/stack, trivial |
| meta-device construction + `load_state_dict(assign=True)` | `loader/single_gpu_model_builder.py:76-119` | replace with `mx.load` + weight-name map |
| `LayerScale`/`Dropout`/`Identity` | `feed_forward.py:12`, `attention.py:181` | `Identity` is a no-op; no dropout at inference |

### 2.9 Parameter count / memory cross-check (validates §2.1 dims)

```
per block: attn1 16,789,504 + attn2 16,789,504 + FF 33,570,816 + sst 12,288 = 67,162,112
48 blocks                                            = 3,223,781,376
model level (patchify 264,192 + adaln 29,900,800 + sst 4,096 + proj_out 262,272) = 30,431,360
                                            total    ≈ 3.254 B  → ×2 B = 6.51 GB bf16
```
which matches the shipped `dramabox-dit-v1.safetensors` = 6.6 GB. If the ported module count drifts
from ≈3.25 B, a config key is wrong.

---

## 3. Audio VAE + vocoder (`ltx2/ltx_core/model/audio_vae/`)

### 3.1 I/O contract

```
waveform (B, C=2, S) @ 16 kHz  →  log-mel (B, 2, S/160, 64) → AudioEncoder → latent (B, 8, T, 16)
latent (B, 8, T, 16)           →  AudioDecoder → log-mel (B, 2, 4T-3, 64) → Vocoder(BWE) → (B, 2, 160*(4T-3)) @ 16 kHz → ×3 → (B, 2, 480*(4T-3)) @ 48 kHz
```
`LATENT_DOWNSAMPLE_FACTOR = 4` (`audio_vae/audio_vae.py:18`).

### 3.2 Encoder (voice-ref path only) **[V]**

`AudioEncoder` (`audio_vae.py:59-245`), config from
`config["audio_vae"]["model"]["params"]["ddconfig"]` (`model_configurator.py:144-182`):

| field | code default | consistency constraint |
|---|---|---|
| `ch` | 128 | must be 128: `PerChannelStatistics(latent_channels=ch)` must match `z_channels × latent_mel_bins = 8×16 = 128` |
| `ch_mult` | `(1,2,4)` | 3 resolution levels ⇒ 2× spatial downsample on both time and freq |
| `num_res_blocks` | 2 | |
| `attn_resolutions` | `[]` | checkpoint value: no inner `AttnBlock`s in encoder or decoder |
| `mid_block_add_attention` | `False` | checkpoint value: no mid-block attention either |
| `resolution` | 256 | only used for the (empty) `attn_resolutions` decision |
| `z_channels` | 8 | latent channels |
| `double_z` | True | `conv_out` → 16 ch; only the first half (mean) is used (`audio_vae.py:236`) |
| `in_channels` | 2 | stereo |
| `norm_type` | `"pixel"` | → `PixelNorm(dim=1, eps=1e-6)`, weightless RMS over channels (`common/normalization.py:14-40`) |
| `causality_axis` | `"height"` | ⇒ causal along time (dim 2 of `(B,C,T,F)`); see below |
| `sample_rate` | 16000 | from `model.params.sampling_rate` |
| `mel_hop_length` | 160 | `preprocessing.stft.hop_length` |
| `is_causal` | True | `preprocessing.stft.causal` |
| `mel_bins` | `ddconfig.mel_bins or mel.n_mel_channels or variables.mel_bins` | 64 (the vocoder's mel count) |

Causality axis convention (`causality_axis.py:4-10`, `causal_conv_2d.py:39-47`): the tensor is
`(B, C, rows=time, cols=freq)`. `HEIGHT` means rows, which is time. Padding is
`(pad_w//2, pad_w-pad_w//2, pad_h, 0)`, symmetric on freq and left-only (causal) on time.
`WIDTH` would pad time on the right, and `WIDTH_COMPATIBILITY` pads `(pad_w, 0)`.

Encoder body:

```
conv_in      CausalConv2d(2 → 128, k=3, s=1)
SiLU
down[level]  for level 0..2:  num_res_blocks × ResnetBlock(k=3, temb=None)   (no AttnBlock: attn_resolutions=[])
             Downsample(Conv2d k3 s2, pad (0,1,2,0) for HEIGHT) between levels      (downsample.py:34-51)
mid          ResnetBlock → ResnetBlock                    (mid_block_add_attention=False, no AttnBlock)
norm_out     PixelNorm(128*4=512)          # block_in after 3 levels = ch*ch_mult[-1] = 512
SiLU → conv_out CausalConv2d(512 → 16, k=3)
chunk(2, dim=1)[0] → (8, T, 16)            # mean half
patchify → (B,T,128) → PerChannelStatistics.normalize → unpatchify
```
`ResnetBlock` (`resnet.py:115-176`): `norm1 → SiLU → conv1 → norm2 → SiLU → dropout(0) → conv2` +
`nin_shortcut` (1×1 causal conv) when `in≠out`. `temb_channels=0` ⇒ no time embedding.
`AttnBlock` (`attention.py:16-55`): `norm → 1×1 conv q,k,v`, `bmm` softmax attention over the
flattened (time·freq) grid, `proj_out`, residual. It is not instantiated in the shipped checkpoint
(`attn_resolutions=[]`, `mid_block_add_attention=False`), and `ltx-core-mlx`'s encoder/decoder
contain no attention weights at all.

`AudioProcessor.waveform_to_mel` (`audio_vae/ops.py:8-55`) is the hardest part to port:

```
torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
    hop_length=160, f_min=0.0, f_max=8000.0, n_mels=64, window_fn=hann_window,
    center=True, pad_mode="reflect", power=1.0, mel_scale="slaney", norm="slaney")
mel = log(clamp(mel, min=1e-5));  permute (B,C,n_mels,frames) → (B,C,frames,n_mels)
```
`power=1.0` means magnitude, not power, STFT; `mel_scale/norm="slaney"`; `center=True` +
`pad_mode="reflect"`; resampling 48 k→16 k via `torchaudio.functional.resample` (sinc, kaiser) only
if the file's rate differs. MLX must reproduce the slaney mel filterbank and the reflect-padded
centered STFT exactly (build the filterbank once on CPU with librosa/torchaudio and store as a
constant; the resampler can stay on CPU/Python or use `mx` polyphase).

### 3.3 Decoder **[V]**

`AudioDecoder` (`audio_vae.py:276-493`), same ddconfig, `out_ch=2`,
`mel_bins = 64` (decoder output mel count, see the constraint note below).

```
_denormalize_latents:
   patchify (B,8,T,16) → (B,T,128) → PerChannelStatistics.un_normalize → unpatchify
   target_frames = T*4 - 3          (because causality_axis != NONE)
conv_in      CausalConv2d(8 → 512, k=3)          # base_block_channels = ch*ch_mult[-1] = 128*4 = 512
mid          ResnetBlock(512) → ResnetBlock(512)  # no AttnBlock (mid_block_add_attention=False)
up[level]    reversed(0..2), num_res_blocks+1 = 3 blocks each:
   level2: 3×ResnetBlock(512→512); curr_res=16 (no attention: attn_resolutions=[]); Upsample(512)
   level1: 3×ResnetBlock(512→256); curr_res=32; Upsample(256)
   level0: 3×ResnetBlock(256→128); curr_res=64; no upsample
norm_out     PixelNorm(128) → SiLU → CausalConv2d(128 → 2, k=3)
_adjust_output_shape: crop/pad (time, freq) to (B, 2, 4T-3, 64)
```
`Upsample` (`upsample.py:12-55`): `F.interpolate(scale_factor=2, mode="nearest")` → causal
`CausalConv2d(k=3)` → drop the first element along the causal axis
(`x[:, :, 1:, :]` for HEIGHT) to undo the encoder's left padding. This "drop first, not last" detail
matters for bit-exactness.

`per_channel_statistics` buffers: `std-of-means` and `mean-of-means`, each shape `(128,)`,
`un_normalize: x*std + mean` applied in the patchified `(B,T,128)` layout (so the 128 statistics
are indexed by `channel*16 + mel_bin`).

Constraint that pins the config: `Vocoder.conv_pre = Conv1d(128 → 1536, k=7)` is hardcoded in
the 2.3+ `VocoderWithBWE` (`bwe.py:266`) and the vocoder front-end does
`rearrange("b s c t -> b (s c) t")` with `s=2` stereo channels, so the decoder output must be
`(B, 2, T', 64)` and `mel_bins` at the decoder output is 64. The latent frequency axis is
16 (`64 / 2 / 2`, the two freq downsamples).

### 3.4 Vocoder / BigVGAN-v2 + BWE **[V]**

`VocoderConfigurator` (`model_configurator.py:42-89`) auto-detects the 2.3+ layout
(`config["vocoder"]["bwe"]` present) and returns `VocoderWithBWE` containing two `Vocoder`s:

```
vocoder       : 16 kHz BigVGAN-v2, resblock="AMP1", activation="snakebeta",
bwe_generator : same architecture (512 base chans), apply_final_activation=False,
                input 16 kHz mel (MelSTFT n_fft=512, hop=80) → output 48 kHz residual
mel_stft      : MelSTFT(filter_length=n_fft=512, hop_length=80, win_length=512, n_mel_channels=64)
                with checkpoint-loaded `forward_basis`, `inverse_basis`, `mel_basis` buffers
resampler     : HannSincResampler(upsample_factor=3, persistent=False)  [computed on CPU]
```

`Vocoder` (`vocoder.py:271-416`), config-driven:

```
conv_pre   Conv1d(128 → 1536, k=7, p=3)
ups[i]     ConvTranspose1d(1536→768, k11, s5, p3)
           ConvTranspose1d( 768→384, k4,  s2, p1)
           ConvTranspose1d( 384→192, k4,  s2, p1)
           ConvTranspose1d( 192→ 96, k4,  s2, p1)
           ConvTranspose1d(  96→ 48, k4,  s2, p1)
           ConvTranspose1d(  48→ 24, k4,  s2, p1)      # ∏ rates = 5·2·2·2·2·2 = 160
resblocks  3 per up stage (18 total), AMPBlock1(ch, k∈{3,7,11}, dilations (1,3,5)):
             for each of 3 conv pairs:  x = x + conv2(act2(conv1(act1(x))))
             act = Activation1d(SnakeBeta(ch)) = upsample×2 → SnakeBeta → downsample÷2
           outputs of the 3 resblocks are stacked and averaged (vocoder.py:404-408)
act_post   Activation1d(SnakeBeta(24))
conv_post  Conv1d(24 → 2, k=7, p=3, bias=False)
final      tanh  (inner vocoder only)
```
* `AMP1` means the `F.leaky_relu` between upsamples is skipped (`vocoder.py:395-396`).
* `SnakeBeta`: `x + (1/(β+1e-9)) * sin(x·α)²`, with `α = exp(alpha_param)`, `β = exp(beta_param)`
  (both logscale parameters, width = channels) (`vocoder.py:186-208`).
* `Activation1d` uses anti-aliased up/down sampling with kaiser-sinc filters
  (`vocoder.py:82-162`):
  * `UpSample1d(ratio=2, kernel_size=12)`: `kernel=12`, `pad = kernel//ratio - 1 = 5`,
    `pad_left = pad*stride + (kernel-stride)//2 = 15`, `pad_right = pad*stride + (kernel-stride+1)//2 = 15`;
    forward = `replicate` pad by 5 → `ratio * conv_transpose1d(groups=C)` → slice `[15:-15]`.
  * `DownSample1d(ratio=2)` → `LowPassFilter1d(cutoff=0.5/2, half_width=0.6/2, stride=2, kernel=12)`,
    `pad_left = 12//2 - 1 = 5`, `pad_right = 6`, `replicate` pad, `conv1d(groups=C)`.
  * The filter itself (`kaiser_sinc_filter1d`, `vocoder.py:30-48`) uses `torch.kaiser_window` and is
    registered with `persistent=True`, so these filters are present in the checkpoint
    (`…upsample.filter`, `…lowpass.filter`); load them rather than recomputing. The only filter
    buffer that is not in the checkpoint is `VocoderWithBWE._resampler.filter` (`persistent=False`,
    `bwe.py:46-80`), so recompute it deterministically (hann-windowed sinc, 2·width·ratio+1 = 43 taps).
* `resblocks` implementations: `ResBlock1` (`resnet.py:12-80`, used when `resblock="1"`) or
  `AMPBlock1` (2.3, above).

`VocoderWithBWE.forward` (`vocoder.py:551-594`):

```
x   = vocoder(mel)                              # (B,2,4T-3,64) → (B,2,S16)
pad x to a multiple of hop_length (right pad with zeros)
mel2 = mel_stft(stft over x)                    # (B,C,64,T_frames), 16 kHz STFT with hop 80
residual = bwe_generator(mel2.transpose(2,3))   # (B,2,S48), 240× over 80-hop mel frames
skip     = resampler(x)                         # hann-windowed sinc ×3  (ratio=3)
return clamp(residual + skip, -1, 1)[..., :S48]
```
The vocoder runs in fp32 (`upcast_weights_to_fp32()` after load; MLX has no autocast).
`_STFTFn` (`vocoder.py:419-454`) is an STFT implemented as a single `conv1d` with precomputed
DFT×Hann bases from the checkpoint (`forward_basis`, shape `2*(n_fft/2+1) × 1 × n_fft`), with
left-only padding of `win_length - hop_length` (causal); magnitude = `sqrt(re²+im²)`,
phase = `atan2` (phase is unused downstream).

### 3.5 Rate map (latent ↔ waveform) **[V]**

```
1 latent frame  = 4 mel frames  = 4*160 samples @16k = 640 samples = 40 ms
T latent frames ⇒ 4T-3 mel frames (causal loss of 3)  ⇒ 160*(4T-3) samples @16 kHz
                ⇒ 480*(4T-3) samples @48 kHz  (BWE ×3)
25 latent frames/s ↔ 100 mel frames/s ↔ 16000 samples/s ↔ 0.04 s per latent frame
```
For the 20.5 s artifact: frame 513 ⇒ `4*513-3 = 2049` mel frames ⇒ `2049*160 = 327,840` samples
@16 kHz = 20.49 s ⇒ 983,520 samples @48 kHz.

---

## 4. Text encoder path (`ltx2/ltx_core/text_encoders/gemma/`)

### 4.1 Gemma configuration **[V]** (`gemma/config.py:10-37`, hardcoded, not read from disk)

```
model_type                 = gemma3_text
num_hidden_layers = 48 , hidden_size = 3840 , intermediate_size = 15360
num_attention_heads = 16 , num_key_value_heads = 8 , head_dim = 256   (GQA 2:1)
head_dim × heads = 4096  (≠ hidden_size 3840 → q/k/v/o proj are not square)
rms_norm_eps = 1e-6 , hidden_activation = gelu_pytorch_tanh
rope_theta (global) = 1_000_000 , rope_local_base_freq = 10_000 , sliding_window = 1024
sliding_window_pattern = 6      # layer i is GLOBAL iff (i+1) % 6 == 0
rope_scaling = {rope_type: "linear", factor: 8.0}      # applied to the global layers
query_pre_attn_scalar = 256     # = head_dim ⇒ vanilla 1/sqrt(256) scaling
attn_logit_softcapping = None , final_logit_softcapping = None   # Gemma-3 dropped soft-capping
vocab_size = 262208 , eos = [1, 106]
```
Weights: `unsloth/gemma-3-12b-it-bnb-4bit` (pre-quantized NF4), loaded through
`transformers.Gemma3ForConditionalGeneration` (`blocks.py:437-488`). The LM head is unused at
inference: `GemmaTextEncoder.encode` calls `self.model.model(...)` (the `Gemma3Model`, skipping
`lm_head`) (`base_encoder.py:44`). The checkpoint loader duplicates `embed_tokens` into `lm_head`
purely for the optional prompt-enhancement path (`encoder_configurator.py:114-121`).

### 4.2 Which hidden states are needed: all 49 **[V]**

`GemmaTextEncoder.encode` (`base_encoder.py:31-47`):

```
token_pairs  = tokenizer.tokenize_with_weights(text)["gemma"]
input_ids    = tensor([[t[0] …]], device=model.device)      # (1, 1024)  max_length=1024
attn_mask    = tensor([[w[1] …]])                           # (1, 1024)  0/1, LEFT padding
outputs      = model.model(input_ids, attention_mask, output_hidden_states=True)
hidden_states = outputs.hidden_states                       # tuple of 49 × (1, 1024, 3840)
```
`FeatureExtractorV2` (`feature_extractor.py:112-141`) does `torch.stack(hidden_states, dim=-1)` →
`(1, 1024, 3840, 49)`, and the aggregate embed is built with
`flat_dim = hidden_size * (num_hidden_layers + 1) = 3840 * 49 = 188,160`
(`encoder_configurator.py:69-98`). Therefore every hidden-state index is required:

| index | content (HF `Gemma3Model` semantics) |
|---|---|
| 0 | `embed_tokens(x) * sqrt(3840)` (embedding scale is applied; buffer set at `encoder_configurator.py:169-170`) |
| 1 … 47 | output of decoder layer `i-1`, pre final norm |
| 48 | `model.norm(layer 47 output)`, the only post-final-norm entry |

So a standard MLX Gemma-3 must expose `hidden_states` for `n_layers + 1 = 49` indices, including the
embedding output and the final post-norm state. Layer-level notes for a correct MLX Gemma-3:
local layers use `rope_local_base_freq=10000` (no scaling) with `sliding_window=1024`;
global layers (i%6==5) use `rope_scaling=linear(8)` on `rope_theta=1e6`; QK-norm is applied
(`q_norm`/`k_norm` over `head_dim=256`). Gemma-3 has no attention soft-capping, uses GQA with
`n_kv_heads=8` and `embed_scale = sqrt(hidden_size)`.

### 4.3 `feature_extractor` math (`FeatureExtractorV2`) **[V]**

```
stack hidden states             (B,T,3840,49)
norm_and_concat_per_token_rms:  v = mean(x², dim=2, keepdim=True)            # per-token, PER LAYER
                                normed = x * rsqrt(v + 1e-6)                 # (B,T,3840,49)
                                normed = reshape → (B,T,188160)
                                masked: zero where attention_mask == 0
rescale:  x * sqrt(target_dim / 3840)      target_dim = 2048 (audio)        # _rescale_norm
audio_aggregate_embed: Linear(188160 → 2048, bias=True)                     # 385 M params
```
Note the norm is RMS over the hidden dim only, independently for each of the 49 layers, and there
is no cross-layer mean/range normalisation (that is `FeatureExtractorV1`, the 19 B variant,
`feature_extractor.py:91-109`, unused here).

### 4.4 `embeddings_processor` / `audio_connector` **[V]**

`EmbeddingsProcessor.process_hidden_states` (`embeddings_processor.py:69-89`):
`feature_extractor → convert_to_additive_mask → create_embeddings → video_connector + audio_connector`.

`convert_to_additive_mask` (`embeddings_processor.py:15-19`):
`(mask-1).to(dtype) * finfo(dtype).max`, reshaped `(B,1,1024,1024)`: 0 for valid tokens,
`-finfo.max` for padding (bf16).

TTSServer patches this (`blocks.py:411-431`). It deletes the video connector and
`video_aggregate_embed`, and replaces `create_embeddings` with a version that only runs the audio
connector:

```
m = additive_attention_mask ; while m.dim() > 2: m = m[:, 0]     # → (B, 1024)
binary_mask = (m >= -1.0).int()                                  # 1 = valid   (returned, unused)
audio_encoded, _ = audio_connector(audio_features, additive_attention_mask)   # (1,1024,2048)
```
So the only text output consumed by the DiT is `EmbeddingsProcessorOutput.audio_encoding`
(`inference_server.py:375`); `video_encoding`/`attention_mask` are ignored.

`Embeddings1DConnector` (`embeddings_connector.py:72-198`), audio variant
(`AudioEmbeddings1DConnectorConfigurator`, `:228-261`):

```
num_attention_heads = audio_connector_num_attention_heads  (fallback connector_num_attention_heads=30)
attention_head_dim  = audio_connector_attention_head_dim   (fallback 128)
num_layers          = audio_connector_num_layers           (fallback 2)
inner_dim = heads*head_dim  MUST equal audio_cross_attention_dim = 2048   ← constraint [C]
positional_embedding_max_pos = connector_positional_embedding_max_pos (default [1])
num_learnable_registers = 128 (hardcoded default)
```
Per layer `_BasicTransformerBlock1D` (`:15-69`): weightless `rms_norm` → `Attention(query_dim=2048,
heads, dim_head, rope)` (same `Attention` class as the DiT, same q/k RMSNorm-by-2048 and RoPE) →
residual → `rms_norm` → `FeedForward(2048, 2048)` (4×, GELU tanh) → residual. Final weightless
`rms_norm` on the output.

Positional handling inside the connector (`:176-198`):

```
indices_grid = arange(S)[None,None,:]  → (1,1,S), S = 1024
precompute_freqs_cis(..., use_middle_indices_grid=False, max_pos=[1])   # frac = pos/1 = pos
   ⇒ angle[j] = grid[j] * (2*pos - 1), grid as in §2.4 (n_freq = 2048//2 = 1024)
```
i.e. the same increasing-frequency grid, with "position" = raw token index and `max_pos = 1`
(`2*pos-1` grows to 2047 at the end of the sequence).

Learnable-register trick (`:135-161`), to be ported exactly:

```
assert S % 128 == 0 ; dups = S/128 ; regs = tile(learnable_registers (128,2048), (dups,1))
valid = (mask_binary >= -9000).int()                       # (B,S,1)
x     = x[:, valid_bool, :]                                # compact valid tokens to the LEFT
x     = pad(x, (0,0,0, S-n_valid))                         # zero tail
flip  = flip(valid, dims=[1])                              # 1s at the FRONT (left padding input)
x     = flip*x + (1-flip)*regs                             # registers fill the tail
mask  = zeros_like(mask)                                   # → full attention afterwards
```
Consequence: the text context is always 1024 tokens (registers included) and there is no text
masking, so `_to_binary_mask` (`embeddings_processor.py:22-27`) yields an all-ones binary mask. Also
`torch.rand` initialises `learnable_registers` in bf16 (`:131-133`), though the values come from the
checkpoint anyway.

### 4.5 Can a standard MLX Gemma-3 supply this?

Yes, provided it (a) returns all 49 hidden states including the embedding output and the final
normed state, (b) uses `embed_scale = sqrt(3840)`, (c) alternates local (`θ=1e4`, window 1024) and
global (`θ=1e6`, linear scaling 8) RoPE with pattern 6, (d) applies QK-norm per head, (e) uses GQA
8 KV heads. `mlx_lm`'s `gemma3_text` matches (b)-(e) structurally but does not collect hidden
states. You either patch its `Model.__call__` to append `h` per layer (using the pre-norm value for
indices 1..47 and the post-`norm` value for index 48), or re-implement 48 layers (~250 lines).

Things that must be regenerated rather than copied from the bnb checkpoint:

* NF4 packing: bitsandbytes stores `weight` (uint8 packed 2 values/byte), `absmax`, `quant_map`
  (nf4 codebook) and, for double-quant, `nested_absmax`/`nested_quant_map`. MLX has no NF4 kernel,
  so either dequantize to bf16/fp16 (≈24 GB) or use `mlx-community/gemma-3-12b-it-4bit` (MLX
  group-wise quantized). Note that MLX 4-bit is not NF4, so outputs will not be bit-identical to the
  CUDA reference.
* tokenizer: `tokenizer.model` (SentencePiece) + HF tokenizer files; `max_length = 1024`,
  `padding_side="left"`, `pad_token = eos` (`gemma/tokenizer.py:18-26`), truncation on.
* Gemma's attention mask must follow HF's left padding; the additive mask derived from the
  *binary* mask is what drives the learnable-register placement (§4.4), so tokenisation and
  padding must match exactly, or the text conditioning shifts.

---

## 5. CUDA assumptions in the inference path, ranked

Rank = how blocking it is for the MLX port of `TTSServer.generate`.

| # | Assumption | file:line | MLX plan |
|---|---|---|---|
| 1 | bitsandbytes NF4 4-bit Gemma (`BitsAndBytesConfig(load_in_4bit, nf4, compute_dtype)`, `device_map`, `.to(device)`, `torch.cuda.memory_allocated`) | `ltx_pipelines/utils/blocks.py:437-488` (esp. `:461-483`, `:467`, `:486`); invoked at `blocks.py:360-361` from `src/inference_server.py:160`,`:172`; `src/inference.py:290`; default `bnb_4bit=True` at `src/inference_server.py:121` | hard blocker: no bitsandbytes on macOS. Use an MLX/bf16 Gemma-3 checkpoint (dequantize offline) and drop the `_load_bnb_4bit_encoder` branch entirely |
| 2 | `torch.compile(..., mode="default", dynamic=True)` on the whole DiT (inductor, dynamic shapes) | `src/inference_server.py:227`; per-block variant `ltx_core/model/transformer/compiling.py:9` and `:11-22` (inductor/dynamo config patches), `:32-37` (SD key rewrite `_orig_mod.`); `blocks.py:179-206` | drop. Optionally `mx.compile` the block (static shapes per request; `T` changes per request → recompile or pad) |
| 3 | CUDA RNG: `torch.Generator(device=cuda).manual_seed(seed)`, bf16 `randn` | `src/inference_server.py:367-368`; `components/noisers.py:24-29`; also `src/inference.py:281-283`; res2s/samplers `samplers.py:293-294` | `mx.random.key(seed)` + `mx.random.normal(..., dtype=bf16)`; document that seeds are not cross-platform reproducible |
| 4 | xformers / FlashAttention-3 selection (import-probe at module load, `AttentionFunction.DEFAULT` → xformers) | `ltx_core/model/transformer/attention.py:7-19`, `49-95` (xformers, mask padded to a multiple of 8), `97-120` (FA3, mask unsupported), `122-139` (`to_callable`) | single `mx.fast.scaled_dot_product_attention` path with `scale=1/sqrt(64)`; MLX adds a broadcastable additive `mask`, so a `(B,1,T,T)` bias works, but verify (a) bf16 bias promotion, (b) mask dtype requirement, (c) fused-kernel support for `B=3 × 32 heads × T=1003`. Fallback: manual `q@kᵀ*scale + bias` → `mx.softmax` → `@v` |
| 5 | `device="cuda"` defaults and `torch.device("cuda")` fallbacks | `src/inference_server.py:121`, `:130`; `src/inference.py:180`; `app.py:38`; `ltx_core/loader/single_gpu_model_builder.py:107`; `ltx_pipelines/utils/helpers.py:30-33`; `ltx_core/loader/fuse_loras.py:11-12` | replace with a device abstraction (`mx.Device(mx.gpu)`) |
| 6 | CUDA memory lifecycle helpers run on every model build/free (`torch.cuda.synchronize`, `empty_cache`, `.to("meta")`) | `ltx_pipelines/utils/gpu_model.py:26-29`; `ltx_pipelines/utils/helpers.py:36-39` (`cleanup_memory`); `blocks.py:433` (`torch.cuda.empty_cache()` in the audio-only strip), `blocks.py:486` (`memory_allocated`) | delete; MLX frees on refcount. Keep an explicit `mx.clear_cache()` after the Gemma pass if memory pressure matters |
| 7 | CUDA streams / pinned host memory / events (layer streaming: `pin_memory()`, `torch.cuda.Stream`, `Event`, `wait_event`, `current_stream`) | `ltx_core/layer_streaming.py:68-72`, `87-94`, `132`, `139-149`, `203`, `231`, `287` | not on the TTSServer path (`streaming_prefetch_count=None` at `inference_server.py:400`), but reachable from the pipeline CLIs. Port as "load layers on demand" without streams, or drop |
| 8 | `torch.autocast(device_type="cuda"/…, dtype=float32)` in the vocoder (fp32 required: bf16 degrades spectral metrics 40-90 %) | `ltx_core/model/audio_vae/vocoder.py:575` (and the comment block `:503-506`, `:561-574`) | cast the vocoder weights and inputs to fp32 in MLX (`voc.astype(mx.float32)`); MLX has no autocast. Cost: ~2× memory for ~250 M params, negligible |
| 9 | fp8 `cublas_scaled_mm` / `static_quantize_e4m3_per_tensor` (TensorRT-LLM) | `ltx_core/quantization/fp8_scaled_mm.py:44`, `:51-58`; gate `quantization/policy.py:29-38`; CLI `ltx_pipelines/utils/args.py:114-143` | not used for the bf16 DramaBox DiT (`DiffusionStage` is called without a `QuantizationPolicy`, and `TTSServer` builds the model directly). Drop; if an fp8 checkpoint is ever used, dequantize to bf16 |
| 10 | Triton kernel `fused_add_round_kernel` (fp8 stochastic rounding) | `ltx_core/loader/kernels.py:1-40`; called from `quantization/fp8_cast.py:10-40` (`import triton` at `:12`), `loader/fuse_loras.py:6-8`, `:40-44`, `:111-120` | only on fp8-cast / LoRA-fusion paths. No Triton on macOS → those code paths must be excluded |
| 11 | `torch.nn.functional.scaled_dot_product_attention` | `attention.py:44` | portable, but see #4 (mask support) |
| 12 | meta-device + `load_state_dict(assign=True)` + `safe_open(..., non_blocking=True)` weight loading | `loader/single_gpu_model_builder.py:75-119`; `loader/sft_loader.py:30-44`; `model/configurator` `with torch.device("meta")` at `blocks.py:186-206`, `model_configurator.py:142-145`, `encoder_configurator.py:27`; `assets`-free `torch.device("cpu")` at `vocoder.py:526-529` | replace with `mx.load` + a name map (`model.diffusion_model.` prefix strip, `audio_vae.decoder.` → ``, `audio_vae.per_channel_statistics.*_of_means` → `mean_of_means`/`std_of_means`, `vocoder.vocoder.` strip, `vocoder.bwe.` → `vocoder.bwe_generator.`, `text_embedding_projection.audio_aggregate_embed.` → `audio_aggregate_embed.`, `model.diffusion_model.audio_embeddings_connector.` → `audio_connector.`) |
| 13 | `torch.inference_mode()` decorators, `torch.Generator`, `torch.tensor(..., device=…)` scattered through the loop | `inference_server.py:306`, `:441`; `noisers.py`; `schedulers.py:57` (`sigmas.to(torch.float32)`) | trivial (`mx.async_eval`/`mx.eval`) |
| 14 | `torchaudio` mel / resample / save (CPU-safe but torch-only) | `audio_vae/ops.py:20-42` (`MelSpectrogram`, `functional.resample`); `inference_server.py:30`, `:526` (`torchaudio.save`) | reimplement mel filterbank as a constant; save via `soundfile`/`mlx` I/O |
| 15 | `av` (PyAV) audio decode | `ltx_pipelines/utils/media_io.py:301-361` | `soundfile`/`librosa`/`ffmpeg`, CPU only, no blocker |
| 16 | Perth watermark (numpy/ONNX, CPU) | `src/inference_server.py:520-531`; `src/inference.py:520-530` | keep on CPU / make optional |

Practical order of work: strip (1), (2), (5)-(7) and (12) first to get a pure "numpy-in / numpy-out"
reference path. Then re-implement attention (#4) and the vocoder fp32 policy (#8).

---

## 6. Porting plan & parity checkpoints

1. **Weight export.** Convert `dramabox-dit-v1.safetensors` (bf16, keys prefixed
   `model.diffusion_model.`) and `dramabox-audio-components.safetensors` (`audio_vae.decoder.*`,
   `audio_vae.per_channel_statistics.*`, `vocoder.*` incl. `vocoder.bwe_generator.*` and `vocoder.mel_stft.*`,
   `text_embedding_projection.audio_aggregate_embed.*`,
   `model.diffusion_model.audio_embeddings_connector.*`) into an MLX tree.
   The exact maps used by the shipped code are in `src/mlx_dramabox/audio.py`
   (`audio_vae.per_channel_statistics.<kind>_of_means` → `mean_of_means` / `std_of_means`,
   `vocoder.vocoder.*` → `vocoder.*` and `vocoder.bwe_generator.*` → `bwe_generator.*` after the outer prefix strip) and
   `src/mlx_dramabox/text.py` (`text_embedding_projection.audio_aggregate_embed.*` →
   `audio_aggregate_embed.*`, `model.diffusion_model.audio_embeddings_connector.*` →
   `audio_connector.*`).
   Verify the DiT parameter count is ≈ 3.254 B (§2.9) and that `audio_aggregate_embed` is
   `(2048, 188160)`.
2. **Read the real config from the safetensors metadata** and log
   `num_layers, audio_num_attention_heads, audio_attention_head_dim, audio_cross_attention_dim,
   apply_gated_attention, cross_attention_adaln, rope_type, frequencies_precision,
   caption_proj_before_connector, audio_connector_{num_attention_heads,attention_head_dim,num_layers},
   timestep_scale_multiplier, use_middle_indices_grid`. Every [?] above resolves here.
3. **Numeric parity harness.** Port stage by stage, comparing against the torch run with
   identical inputs dumped to disk:
   | checkpoint | reference tensor |
   |---|---|
   | C1 | `AudioLatentTools.create_initial_state` → latent/mask/positions |
   | C2 | ref latent from the VAE encoder (`(1,8,250,16)`) |
   | C3 | `a_ctx` (`(1,1024,2048)`), validates tokenizer + 49 hidden states + feature extractor |
   | C4 | the `pe` cos/sin table (`(3,1003,2048)`), catches any RoPE mistake early |
   | C5 | one DiT forward at σ=1 for the 3-pass batch (velocity) |
   | C6 | `sigmas` (31 values), validates the patchified-latent `tokens=128` quirk |
   | C7 | final latent after 30 steps |
   | C8 | decoded mel `(1,2,3009,64)` and waveform `(2,1444320)` @48 kHz |
   Expect exact bf16 agreement (±1-2 ulp) up to C6, then divergence growing slowly.
4. **Performance expectations.** Per step the DiT does 3 passes over 3.25 B params: ~72 GFLOP of
   FF + ~19 GFLOP of attention per block-equivalent pass ⇒ ≈ 3.5 TFLOP per pass, ≈ 10.5 TFLOP per
   step, ≈ 315 TFLOP for 30 steps. On Apple silicon this is *bandwidth*-bound at ~6.5 GB per pass
   × 90 passes ≈ 585 GB of weight traffic ⇒ order 1-3 s/step on an M-series Max/Ultra. The Gemma
   pass (~12 B params × 1024 tokens, prefill) is a comparable cost. Concrete optimizations that do
   not change numerics:
   * cache the RoPE cos/sin table, the self-attention additive bias, and the connector PE per
     request (§2.4, §2.5);
   * cache cross-attention `to_k`/`to_v` projections of `a_ctx` / `a_ctx_neg` per pass
     (the context is identical for all 30 steps and all 48 blocks see the same context
     projection input per block, at minimum cache per (pass, block));
   * `mx.async_eval` the denoise loop.
5. **Numerics that must not be "cleaned up"**: the `tokens=128` scheduler quirk (§1.7), the
   `(2·pos/20 - 1)` signed RoPE argument (§2.4), the increasing frequency grid (§2.4), the
   `reshape`-only patchifier (§1.3), the 0.5 s ref-position offset (§1.4), the asymmetric ref
   attention mask (§1.4), the "drop first element" causal upsample (§3.3), the latent repair at
   frames 512/513 (§1.10), the global-`std` CFG rescale (§1.8), and the fp32 vocoder (§5 #8).

---

## Appendix A: file map

| concern | file |
|---|---|
| entry point / orchestration | `src/inference_server.py` (`TTSServer`), `src/inference.py` (CLI), `app.py` (Gradio) |
| ref conditioning | `src/audio_conditioning.py` |
| blocks owning model lifecycles | `ltx2/ltx_pipelines/utils/blocks.py` (`PromptEncoder`, `AudioConditioner`, `AudioDecoder`) |
| denoise loop / guiders / samplers | `ltx_pipelines/utils/samplers.py`, `utils/denoisers.py`, `ltx_core/components/guiders.py`, `components/diffusion_steps.py` |
| perturbations (STG) | `ltx_core/guidance/perturbations.py` |
| latents / tools / patchifiers | `ltx_core/types.py`, `ltx_core/tools.py`, `ltx_core/components/patchifiers.py` |
| transformer | `ltx_core/model/transformer/{model,transformer,attention,rope,adaln,timestep_embedding,feed_forward,gelu_approx,text_projection,transformer_args,modality}.py` |
| audio VAE / vocoder | `ltx_core/model/audio_vae/*` , `ltx_core/model/common/normalization.py` |
| text encoder | `ltx_core/text_encoders/gemma/{config,feature_extractor,embeddings_processor,embeddings_connector}.py`, `encoders/{base_encoder,encoder_configurator}.py`, `tokenizer.py` |
| weight loading / sd ops | `ltx_core/loader/*`, `ltx_core/quantization/*` |
| not on the TTS path | `model/upsampler/*` (latent upsampler, video only: `LatentUpsampler`: `Conv3d(128→512,k3)` + `GroupNorm(32)` + SiLU + 4 `ResBlock`s + `Conv3d(→2048,k3)` + `PixelShuffleND` + 4 post blocks + `Conv3d(→128,k3)`, used by `VideoUpsampler`/`upsample_video`), `model/video_vae/*`, `conditioning/types/{keyframe,reference_video,noise_mask,latent}_cond.py`, `modality_tiling.py`, `batch_split.py` (TTSServer bypasses it), `layer_streaming.py`, `ltx_trainer/*`, `src/train.py`, `src/preprocess.py`, `src/validate.py` |
