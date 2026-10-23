"""Dialogue workshop — script parsing, batch generation, joining.

Modified by Siddid Soni, 2026-10: all lines are encoded in one batch before voicing.
"""
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read as wav_read, write as wav_write

from config import OUTPUT_DIR

log = logging.getLogger(__name__)

_LINE_RE = re.compile(r"^(.+?)[：:]\s*(.+)$")

# ── Emotion presets: UI label → English direction fed to the model ──
# The direction goes outside the quotes (performed, not spoken). Tune an emotion by editing its sentence.
EMOTION_PRESETS: dict[str, str] = {
    "Normal": "A person speaks calmly and naturally",
    "Happy": "A person speaks warmly and brightly, a smile they can't hide in their voice",
    "Excited": "A person speaks loudly with full energy, the pace quickening, every word ringing",
    "Angry": "A person seethes through clenched teeth, then suddenly raises their voice in a furious shout",
    "Sad": "A person speaks in a choked voice, every word heavy, holding back tears",
    "Afraid": "A person speaks with a trembling voice, breath short and unsteady, frightened",
    "Gentle": "A person speaks softly, their voice light, warm and tender",
    "Affectionate": "A person speaks with deep feeling, sincere and moving, lingering on each word",
    "Surprised": "A person freezes for a beat, then speaks with sudden weight, unable to believe it",
    "Laughing": "A person can't hold it in any longer and speaks through hearty, open laughter",
    "Cold": "A person speaks flatly and distantly, without a flicker of emotion",
    "Sarcastic": "A person speaks with a cold smirk, drawing out the words in a mocking, sardonic tone",
    "Serious": "A person's expression darkens; they speak gravely, word by deliberate word",
    "Tired": "A person sighs the words out slowly, voice low and drained of energy",
    "Commanding": "A person speaks firmly from above, leaving no room for argument, a sharp command",
    # Naturally feminine deliveries, marked (female) in the UI so they aren't used on male voices
    "Pleading (female)": "A young woman whines sweetly, her pitch lifting and her words drawn out in a soft, pouty plea",
    "Coquettish (female)": "A woman speaks in a sweet, soft and flirtatious voice, playful and charming",
    "Seductive (female)": "A woman speaks low and slightly husky, slow and lingering, each word curling with an alluring smile",
    "Sultry whisper (female)": "A woman drops her voice to an intimate, breathy whisper, as if close to the ear, languid and sultry",
}
# Labels for the UI dropdowns
EMOTION_LABELS: list[str] = list(EMOTION_PRESETS.keys())


def parse_script(text: str) -> list[dict]:
    """Parse a script into [{character, line}, ...].

    Formats (ASCII or Chinese colon):
        Alex: Hello there!
        Sam：Long time no see.
        Alex: "Quoted dialogue"
        (a line without a name prefix is appended to the previous line)

    Lines with no detectable character are given to the Narrator.
    """
    results: list[dict] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if m:
            char = (m.group(1) or "").strip()
            content = m.group(2).strip()
        else:
            char = ""
            content = line

        # strip surrounding quotes
        for _q in ('"', "'", '“', '”', '‘', '’', '＂', '「', '」'):
            if content.startswith(_q):
                content = content[len(_q):]
            if content.endswith(_q):
                content = content[:-len(_q)]

        if not content:
            continue

        character = char if char else None
        # no character: append to the previous line
        if character is None and results:
            results[-1]["line"] += content
        else:
            results.append({"character": character or "Narrator", "line": content})

    return results


def generate_dialogue(
    lines: list[dict],
    voice_map: dict[str, str],
    tts,
    progress_cb=None,
) -> str | None:
    """Voice each line → join → return the path of the full audio.

    lines:       parse_script() output
    voice_map:   {character name: reference clip path}
    tts:         a TTSServer
    progress_cb: optional callback (current, total, character, line)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segments: list[np.ndarray] = []
    sr = 24000
    total = len(lines)

    # Official format: direction outside the quotes (not spoken), the line inside double quotes
    prompts = [f'{EMOTION_PRESETS.get(item.get("emotion") or "Normal", EMOTION_PRESETS["Normal"])}, "{item["line"]}"'
               for item in lines]
    # Encode every line first: the low-VRAM tiers swap GPU stages once
    from inference_server import DEFAULT_NEG
    tts.encode([*prompts, DEFAULT_NEG])

    for i, item in enumerate(lines):
        char = item["character"]
        line = item["line"]
        ref = voice_map.get(char) if char else None
        prompt = prompts[i]

        log.info(f"[{i+1}/{total}] {char}: {line[:40]}...")
        if progress_cb:
            progress_cb(i + 1, total, char or "", line)

        waveform, sample_rate = tts.generate(prompt=prompt, voice_ref=ref)
        wav = waveform.cpu().float()
        if wav.dim() == 3:
            wav = wav.squeeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_np = wav.squeeze(0).numpy()
        segments.append(wav_np)
        sr = sample_rate

        # 0.3 s pause between lines
        silence = np.zeros(int(sr * 0.3), dtype=np.float32)
        segments.append(silence)

    if not segments:
        return None

    merged = np.concatenate(segments)
    merged_int16 = (merged * 32767).clip(-32768, 32767).astype(np.int16)
    out_path = OUTPUT_DIR / f"dialogue_{int(time.time())}.wav"
    wav_write(str(out_path), sr, merged_int16)
    log.info(f"Dialogue done: {out_path} ({len(segments)//2} lines, {len(merged)/sr:.1f}s)")
    return str(out_path)
