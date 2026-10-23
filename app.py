"""
DramaBox Studio — AI voice studio
Based on Resemble AI / Lightricks LTX-2.3

Modified by Siddid Soni, 2026-10: server picked per VRAM tier (create_server), Apple Silicon allowed.
Modified by Siddid Soni, 2026-10: Gradio UI replaced by a FastAPI JSON API + the Night City web frontend in web/.
"""
import json
import logging
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

# ── 环境检查 ────────────────────────────────────

def _check_environment():
    """启动前检查，中文路径和 GPU 驱动问题直接报错退出。"""
    cwd = str(APP_DIR)
    non_ascii = [c for c in cwd if ord(c) > 127]
    if non_ascii:
        print("\n" + "=" * 50)
        print("[ERROR] Path contains non-English characters!")
        print(f"Current: {cwd}")
        print("CUDA / PyTorch do NOT support paths with Chinese characters.")
        print("Please move this folder to a path with English letters only.")
        print(f"Example: D:\\Tools\\DramaBoxStudio\\")
        print("=" * 50 + "\n")
        sys.exit(1)

    if sys.platform == "darwin":
        import platform
        if platform.machine() != "arm64":
            print("\n[ERROR] macOS needs an Apple Silicon Mac (M1 or later).\n")
            sys.exit(1)
        return

    import subprocess
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n" + "=" * 50)
        print("[ERROR] NVIDIA graphics driver not detected!")
        print("DramaBox requires an NVIDIA GPU with CUDA drivers.")
        print("Please install drivers: https://www.nvidia.com/download/")
        print("=" * 50 + "\n")
        sys.exit(1)

_check_environment()
# 检查通过才能走到这里
print("Environment checks passed: path OK, GPU OK")

from inference_server import create_server
from config import (GRADIO_PORT, DIT_CHECKPOINT, AUDIO_COMPONENTS, OUTPUT_DIR, VOICES_DIR, TMP_DIR,
                    CFG_SCALE, STG_SCALE, SEED, TIER, PARK, USE_INT8, BACKEND, GEMMA_OUTPUT_MODES)
from voice_library import list_voices, save_voice, delete_voice, _resolve_path
from dialogue import parse_script, EMOTION_LABELS

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("DramaBox")

WEB_DIR = APP_DIR / "web"
BUILTIN_VOICES_DIR = APP_DIR / "assets" / "voices"
UPLOAD_DIR = TMP_DIR / "uploads"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_QUEUE = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── 加载模型（全局单例）──────────────────────────────
log.info("DramaBox Studio starting, loading models...")
tts = create_server(
    checkpoint=str(DIT_CHECKPOINT),
    full_checkpoint=str(AUDIO_COMPONENTS),
    dtype="bf16",
)
log.info("Models ready.")

# ── Presets ─────────────────────────────────────────
# (title, subtitle, built-in voice stem, prompt)
EXAMPLES: list[tuple[str, str, str, str]] = [
    ("Mob boss", "cold threat", "male_gangster",
     'An aging mob boss speaks in a gravelly, quiet voice, calm but pulsing with menace, '
     '"You thought you could walk through that door and leave alive?" '
     'He chuckles darkly, "Hahaha." He lets the silence stretch. His voice drops to a lethal murmur, '
     '"Young man... who gave you the nerve?"'),
    ("Furious", "slams the table", "male_furious",
     'A man speaks through clenched teeth, barely containing his fury, each word deliberate and crushing, '
     '"I will ask you... one last time. Where. Is. It." '
     'His restraint snaps. His voice detonates into a full-throated roar, '
     '"Don\'t make me do this myself!"'),
    ("Stand-up", "laughing breathless", "male_comedian",
     'A stand-up comedian gasps in theatrical disbelief, his voice shooting up, '
     '"No way! You actually said that?!" '
     'He erupts into helpless laughter, "Hahaha, hahaha." '
     'He wheezes, struggling for air, "Oh my god, I\'m dying, I can\'t breathe!" '
     'He breaks down again, "Hahaha."'),
    ("Late-night radio", "soothing", "male_deep_warm",
     'A deep, warm male voice speaks into the mic, slow and unhurried, as if murmuring directly against the ear, '
     '"It\'s late. To everyone still awake out there, this next song is for you, carrying something heavy tonight." '
     'A long pause. His voice softens even further, dropping to a tender whisper, '
     '"You don\'t have to hold it all together... tonight, just get some sleep."'),
    ("Young hero", "awakening", "male_young",
     'A young man\'s voice trembles, on the verge of tears, fragile and breaking, '
     '"Can I... can I really do this?" '
     'A long pause. His breathing steadies. His voice hardens with sudden, fierce resolve, '
     'each word landing like a blow, "Enough! This time, I am not running away!" '
     'He shouts the last line, his voice cracking with raw conviction.'),
    ("Drunk dad", "late-night confession", "male_weathered",
     'A middle-aged man slurs his words, his voice thick with alcohol and exhaustion, heavy with guilt, '
     '"Son... your old man never amounted to much. I\'m sorry you had to suffer for it." '
     'He sighs deeply. A long pause. He speaks quieter now, his voice cracking, barely holding back tears, '
     '"You... you have to do better than me, okay?"'),
    ("Palace consort", "smiling dagger", "female_refined",
     'A woman speaks in a silken voice, graceful and perfectly poised, dripping with honeyed insincerity, '
     '"What a lovely dress you\'re wearing, little sister." '
     'She lets out a soft, cold laugh, "Hehehe." '
     'Her voice drops to an icy whisper, venom wrapped in velvet, '
     '"Such a pity... I doubt you\'ll ever get to wear it again."'),
    ("Sweet & clingy", "pleading", "female_sweet",
     'A young woman speaks in a bright, singsong voice, her tone lifting into a playful whine, '
     'every sentence curling upward, "Nooo, I don\'t care! You promised you\'d spend today with me!" '
     'She giggles, "Hehehe." She drags her words out in an exaggerated, pouty drawl, '
     '"Pleeease, let\'s go, let\'s go, pretty please?"'),
]

GEMMA_OUTPUT_CHOICES = [
    ("real", "exact, lowest GPU peak (default)"),
    ("full", "exact, keeps Gemma's padded output (original behaviour)"),
    ("project", "least memory, ~0.5–1.4% rounding differences"),
    ("project-exact", "exact, little memory, ~7 GB peak on 8 GB cards"),
]


# ── Voice references ────────────────────────────────
# The frontend names a reference as "lib:<name>" (voice library), "builtin:<stem>" (assets/voices),
# "upload:<id>" (a clip uploaded this session) or "" (the model's default voice).

_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{2,4}$")


def _builtin_voices() -> list[Path]:
    return sorted(p for p in BUILTIN_VOICES_DIR.glob("*") if p.suffix.lower() in AUDIO_EXTS)


def resolve_ref(ref: str | None) -> str | None:
    """Reference spec → absolute file path (None = default voice). Raises 400 on anything unknown."""
    if not ref:
        return None
    kind, _, key = ref.partition(":")
    path = None
    if kind == "lib":
        entry = next((v for v in list_voices() if v["name"] == key), None)
        path = _resolve_path(entry["path"]) if entry else None
    elif kind == "builtin":
        path = next((p for p in _builtin_voices() if p.stem == key), None)
    elif kind == "upload" and _UPLOAD_ID.match(key):
        path = UPLOAD_DIR / key
    if path is None or not path.exists():
        raise HTTPException(400, f"Voice reference not found: {ref}")
    return str(path)


def _duration(path: Path) -> float:
    from voice_library import _get_duration
    return _get_duration(path)


def _check_voice_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "Please enter a voice name.")
    if len(name) > 64 or any(c in name for c in '/\\:*?"<>|') or name.startswith("."):
        raise HTTPException(400, "Voice names can't contain / \\ : * ? \" < > | or start with a dot.")
    return name


# ── Job queue: one worker, so the GPU runs one generation at a time ──

class Jobs:
    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._mu = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="dramabox-worker").start()

    def submit(self, kind: str, fn) -> dict:
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "kind": kind, "status": "queued", "progress": None,
               "result": None, "error": None, "created": time.time()}
        with self._mu:
            self._jobs[job_id] = job
            self._order.append(job_id)
            for old in self._order[:-50]:   # keep the last 50 jobs
                self._jobs.pop(old, None)
            self._order = self._order[-50:]
        try:
            self._q.put_nowait((job_id, fn))
        except queue.Full:
            with self._mu:
                self._jobs.pop(job_id, None)
            raise HTTPException(429, "The queue is full. Try again when a render finishes.")
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._mu:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            out = dict(job)
            if job["status"] == "queued":
                queued = [j for j in self._order if self._jobs.get(j, {}).get("status") == "queued"]
                out["position"] = queued.index(job_id) + 1
            return out

    def busy(self) -> dict:
        with self._mu:
            states = [j["status"] for j in self._jobs.values()]
        return {"running": states.count("running"), "queued": states.count("queued")}

    def _set(self, job_id: str, **kw):
        with self._mu:
            if job_id in self._jobs:
                self._jobs[job_id].update(kw)

    def _worker(self):
        while True:
            job_id, fn = self._q.get()
            self._set(job_id, status="running", started=time.time())
            try:
                result = fn(lambda **p: self._set(job_id, progress=p))
                self._set(job_id, status="done", result=result, finished=time.time())
            except HTTPException as e:
                self._set(job_id, status="error", error=e.detail)
            except Exception as e:
                log.exception("Job failed")
                self._set(job_id, status="error", error=str(e) or type(e).__name__)


jobs = Jobs()


# ── Outputs ─────────────────────────────────────────

def _apply_speed(path: str, speed: float) -> str:
    """Tempo change without pitch shift (PyAV atempo, offline); the unsped intermediate is removed."""
    if not speed or abs(float(speed) - 1.0) < 0.02:
        return path
    try:
        from audio_speed import change_speed
        out = change_speed(path, float(speed))
        Path(path).unlink(missing_ok=True)
        return out
    except Exception as e:
        log.warning(f"Speed change failed, keeping original speed: {e}")
        return path


def _write_meta(wav: str, meta: dict) -> dict:
    p = Path(wav)
    meta = {**meta, "file": p.name, "url": f"/output/{p.name}",
            "duration": _duration(p), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _output_path(name: str) -> Path:
    p = (OUTPUT_DIR / name).resolve()
    if p.parent != OUTPUT_DIR.resolve() or p.suffix.lower() != ".wav" or not p.exists():
        raise HTTPException(404, "Output not found.")
    return p


# ══════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════

app = FastAPI(title="DramaBox Studio", docs_url="/api/docs", redoc_url=None)


class GenerateReq(BaseModel):
    prompt: str
    voice: str = ""
    cfg: float = CFG_SCALE
    stg: float = STG_SCALE
    dur_mult: float = 1.1
    gen_dur: float = 0.0
    ref_dur: float = 10.0
    seed: int = SEED
    speed: float = 1.0


class OptionsReq(BaseModel):
    trim: bool | None = None
    gemma_output: str | None = None


class VoiceReq(BaseModel):
    name: str
    ref: str


class ScriptReq(BaseModel):
    script: str


class DialogueLine(BaseModel):
    character: str
    line: str
    emotion: str = "Normal"


class DialogueReq(BaseModel):
    lines: list[DialogueLine]
    voices: dict[str, str] = {}
    speed: float = 1.0


@app.get("/api/status")
def status():
    vram = None
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram = {"used_gb": round((total - free) / 2**30, 1), "total_gb": round(total / 2**30, 1)}
    except Exception:
        pass
    return {
        "tier": TIER, "park": PARK, "int8": USE_INT8, "backend": BACKEND, "vram": vram,
        "trim": tts.trim, "always_trim": tts.ALWAYS_TRIM, "gemma_output": tts.gemma_output,
        "gemma_choices": [{"value": v, "label": d} for v, d in GEMMA_OUTPUT_CHOICES],
        "defaults": {"cfg": CFG_SCALE, "stg": STG_SCALE, "seed": SEED},
        "emotions": EMOTION_LABELS,
        "jobs": jobs.busy(),
    }


@app.post("/api/options")
def options(req: OptionsReq):
    """Speed / memory options; they apply to the next prompt."""
    if req.gemma_output is not None and req.gemma_output not in GEMMA_OUTPUT_MODES:
        raise HTTPException(400, f"gemma_output must be one of {GEMMA_OUTPUT_MODES}")
    try:
        msg = tts.set_options(trim=req.trim, gemma_output=req.gemma_output)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"message": msg, "trim": tts.trim, "gemma_output": tts.gemma_output}


@app.get("/api/examples")
def examples():
    return [{"title": t, "subtitle": s, "voice": f"builtin:{v}", "prompt": p} for t, s, v, p in EXAMPLES]


@app.get("/api/voices")
def voices():
    lib = [{"ref": f"lib:{v['name']}", "name": v["name"], "duration": v.get("duration_seconds", 0),
            "created": v.get("created_at", ""), "builtin": False} for v in list_voices()]
    builtin = [{"ref": f"builtin:{p.stem}", "name": p.stem.replace("_", " "), "duration": _duration(p),
                "created": "", "builtin": True} for p in _builtin_voices()]
    return lib + builtin


@app.get("/api/voices/audio")
def voice_audio(ref: str):
    return FileResponse(resolve_ref(ref))


@app.post("/api/uploads")
def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, f"Unsupported audio type. Use one of: {', '.join(sorted(AUDIO_EXTS))}")
    name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ref": f"upload:{name}", "name": Path(file.filename).stem, "duration": _duration(dest)}


@app.post("/api/voices")
def add_voice(req: VoiceReq):
    name = _check_voice_name(req.name)
    if save_voice(name, resolve_ref(req.ref)) is None:
        raise HTTPException(400, "Save failed. Please check the audio file.")
    return {"ref": f"lib:{name}"}


@app.delete("/api/voices/{name}")
def remove_voice(name: str):
    if not delete_voice(name):
        raise HTTPException(404, f'Voice "{name}" not found. It may already have been deleted.')
    return {"ok": True}


@app.post("/api/generate")
def generate(req: GenerateReq):
    """Single take. Queued; poll /api/jobs/{id}."""
    if not req.prompt.strip():
        raise HTTPException(400, "Please enter a prompt.")
    ref_path = resolve_ref(req.voice)

    def run(progress):
        progress(label="Rendering take")
        t0 = time.time()
        out = str(OUTPUT_DIR / f"output_{int(time.time() * 1000)}.wav")
        tts.generate_to_file(
            prompt=req.prompt, output=out, voice_ref=ref_path,
            cfg_scale=req.cfg, stg_scale=req.stg, duration_multiplier=req.dur_mult,
            seed=int(req.seed), gen_duration=float(req.gen_dur), ref_duration=float(req.ref_dur),
        )
        out = _apply_speed(out, req.speed)
        log.info(f"Generated: {time.time() - t0:.1f}s → {out}")
        return _write_meta(out, {"kind": "take", **req.model_dump(), "render_s": round(time.time() - t0, 1)})

    return jobs.submit("take", run)


@app.post("/api/dialogue/parse")
def dialogue_parse(req: ScriptReq):
    if not req.script.strip():
        raise HTTPException(400, "Please paste a script.")
    lines = parse_script(req.script)
    if not lines:
        raise HTTPException(400, "No dialogue lines found.")
    chars = list(dict.fromkeys(item["character"] for item in lines))
    return {"lines": lines, "characters": chars}


@app.post("/api/dialogue/generate")
def dialogue_generate(req: DialogueReq):
    """Whole dialogue. Queued; progress reports the line being voiced."""
    if not req.lines:
        raise HTTPException(400, "Please parse the script first.")
    voice_map = {char: path for char, ref in req.voices.items() if (path := resolve_ref(ref))}
    lines = [{"character": l.character, "line": l.line,
              "emotion": l.emotion if l.emotion in EMOTION_LABELS else "Normal"} for l in req.lines]

    def run(progress):
        from dialogue import generate_dialogue
        t0 = time.time()
        progress(current=0, total=len(lines), label="Encoding prompts")

        def cb(cur, total, character, line):
            progress(current=cur, total=total, label=f"{character}: {line[:48]}")

        out = generate_dialogue(lines, voice_map, tts, cb)
        if out is None:
            raise HTTPException(500, "Generation failed.")
        out = _apply_speed(out, req.speed)
        return _write_meta(out, {"kind": "dialogue", "lines": lines, "voices": req.voices,
                                 "speed": req.speed, "render_s": round(time.time() - t0, 1)})

    return jobs.submit("dialogue", run)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "Job not found.")
    return j


@app.get("/api/outputs")
def outputs(limit: int = 40):
    """Recent renders, newest first. Renders made before the web UI have no metadata."""
    items = []
    for wav in sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta_path = wav.with_suffix(".json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        except Exception:
            meta = None
        items.append(meta or {"kind": "legacy", "file": wav.name, "url": f"/output/{wav.name}",
                              "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wav.stat().st_mtime))})
    return items


@app.get("/output/{name}")
def output_file(name: str):
    return FileResponse(_output_path(name), media_type="audio/wav")


@app.delete("/api/outputs/{name}")
def delete_output(name: str):
    p = _output_path(name)
    p.unlink()
    p.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    port = int(os.environ.get("DRAMABOX_PORT", os.environ.get("GRADIO_SERVER_PORT", str(GRADIO_PORT))))
    if os.environ.get("DRAMABOX_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
