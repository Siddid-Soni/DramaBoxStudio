"""End-to-end Whisper gate (heavy; opt in with MLX_E2E_WHISPER=1).

Loads the full ~17 GB pipeline and transcribes the generated hello clip. This is
the automated version of the manual validation in README_MLX.md.
"""

import os

import pytest

from _helpers import (COMPONENTS_PATH, DIT_PATH, GEMMA_DIR, HAVE_GEMMA,
                      HAVE_MODELS)

E2E = os.environ.get("MLX_E2E_WHISPER") == "1"


@pytest.mark.skipif(
    not (HAVE_MODELS and HAVE_GEMMA and E2E),
    reason="needs models + MLX_E2E_WHISPER=1",
)
def test_e2e_whisper_hello(tmp_path):
    from mlx_dramabox import DramaBoxMLX
    import mlx_whisper

    out = str(tmp_path / "hello.wav")
    server = DramaBoxMLX(
        model_dir=str(DIT_PATH.parent),
        gemma_dir=str(GEMMA_DIR),
        keep_text_encoder=False,
    )
    server.generate_to_file(
        prompt='A woman speaks warmly, "Hello, how are you today?"',
        output=out,
        gen_duration=4.0,
        steps=30,
    )
    text = mlx_whisper.transcribe(
        out, path_or_hf_repo="mlx-community/whisper-small-mlx", verbose=False
    )["text"].lower()
    assert "hello" in text
    assert "how are you" in text
