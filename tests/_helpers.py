"""Shared paths, gates and cheap fakes for the DramaBox MLX tests."""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
def _first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.is_file()), paths[0])


DIT_PATH = _first_existing(MODELS / "dramabox-dit-v1.safetensors",
                           MODELS / "checkpoints" / "dramabox-dit-v1.safetensors")
COMPONENTS_PATH = _first_existing(MODELS / "dramabox-audio-components.safetensors",
                                 MODELS / "checkpoints" / "dramabox-audio-components.safetensors")
GEMMA_DIR = next((p for p in (MODELS / "gemma-3-12b-it-4bit", MODELS / "gemma-mlx")
                  if p.is_dir()), MODELS / "gemma-3-12b-it-4bit")

HAVE_MODELS = DIT_PATH.is_file() and COMPONENTS_PATH.is_file()
HAVE_GEMMA = HAVE_MODELS and GEMMA_DIR.is_dir()
HEAVY = os.environ.get("MLX_HEAVY_TESTS") == "1"

SKIP_NO_MODELS = "models/ not present"
SKIP_HEAVY = "needs models + MLX_HEAVY_TESTS=1"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


class FakeDiT:
    """Zero-velocity stand-in that records the context it was called with."""

    rope_type = "split"

    def __init__(self):
        self.contexts = []

    def __call__(self, latent, timesteps, sigma, context, positions,
                 attention_mask=None, skip_blocks=(), eval_every=8, rope=None):
        self.contexts.append(context)
        return mx.zeros_like(latent)


def make_fake_pipeline():
    """DramaBoxMLX without __init__: no models loaded, only generate() is usable."""
    from mlx_dramabox.pipeline import DramaBoxMLX

    obj = object.__new__(DramaBoxMLX)
    obj.dit = FakeDiT()
    obj.decoder = None
    obj.vocoder = None
    obj.text = None
    obj.keep_text_encoder = True
    obj.gemma_dir = "<test>"
    return obj
