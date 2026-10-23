"""
DramaBox Studio — AI voice studio
Based on Resemble AI / Lightricks LTX-2.3

Modified by Siddid Soni, 2026-10: server picked per VRAM tier (create_server), Apple Silicon allowed.
"""
import logging
import os
import sys
import time
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
from config import (GRADIO_PORT, DIT_CHECKPOINT, AUDIO_COMPONENTS, GEMMA_DIR,
                    OUTPUT_DIR, VOICES_DIR, CFG_SCALE, STG_SCALE, STEPS, SEED,
                    TIER, PARK, USE_INT8)
from voice_library import list_voices, save_voice, delete_voice
from dialogue import parse_script, EMOTION_LABELS

# Per-line emotion: the UI offers an emotion picker for the first N lines (the rest use "Normal")
MAX_EMOTION_LINES = 30

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("DramaBox")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ── 加载模型（全局单例）──────────────────────────────
log.info("DramaBox Studio starting, loading models...")
tts = create_server(
    checkpoint=str(DIT_CHECKPOINT),
    full_checkpoint=str(AUDIO_COMPONENTS),
    dtype="bf16",
)
log.info("Models ready.")

# ══════════════════════════════════════════════════
# UI — 暗色影棚风
# ══════════════════════════════════════════════════

CSS = """
/* ═══════════════════════════════════════════════════
   DramaBox Studio · 暗夜鎏金 (Midnight Gold)
   纯外观样式 — 不含任何布局/定位/遮罩/flex容器规则
   ═══════════════════════════════════════════════════ */
:root {
    --bg:        #0E0E10;
    --surface:   #16161A;
    --surface-2: #1C1C21;
    --line:      rgba(255,255,255,.065);
    --line-2:    rgba(255,255,255,.11);
    --txt:       #EDEBE4;
    --txt-dim:   #9C9A90;
    --txt-mute:  #66645C;
    --gold:      #C9A24B;
    --gold-2:    #E2C277;
    --gold-soft: rgba(201,162,75,.10);
    --gold-line: rgba(201,162,75,.30);
    --r:    16px;
    --r-sm: 10px;
    --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 34px rgba(0,0,0,.30);
}

/* ── 全局基调 ── */
body, .gradio-container {
    background: var(--bg) !important;
    color: var(--txt) !important;
    font-family: "Inter", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif !important;
    letter-spacing: .005em !important;
}
/* 默认就放大到 125%（等价于浏览器缩放 125%）；加在 body 上避免横向溢出 */
body { zoom: 1.25; }
/* 全宽利用 16:9 屏，超宽屏封顶；紧凑垂直间距以求接近一屏 */
.gradio-container {
    max-width: 1680px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 12px 30px 8px !important;
}

/* ── 卡片：组件块统一面板 ── */
.block, .form {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--r) !important;
    box-shadow: var(--shadow) !important;
}

/* ── Tab 栏：杂志感金色下划线 ── */
.tabs > .tab-nav {
    border-bottom: 1px solid var(--line) !important;
    margin-bottom: 16px !important; gap: 2px !important;
}
.tabs > .tab-nav > button {
    background: transparent !important; color: var(--txt-dim) !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; margin: 0 !important;
    font-size: 15px !important; font-weight: 600 !important;
    padding: 13px 26px !important;
    transition: color .2s ease, border-color .2s ease !important;
}
.tabs > .tab-nav > button:hover { color: var(--txt) !important; }
.tabs > .tab-nav > button.selected {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── 输入框 / 文本域 / 数字 ── */
textarea, input[type="text"], input[type="number"],
.gradio-container input[type="text"], .gradio-container textarea {
    background: var(--surface-2) !important; color: var(--txt) !important;
    border: 1px solid var(--line) !important; border-radius: var(--r-sm) !important;
    font-size: 14px !important; line-height: 1.6 !important;
    transition: border-color .2s ease, background .2s ease !important;
}
textarea::placeholder, input::placeholder { color: var(--txt-mute) !important; }
textarea:focus, input:focus {
    border-color: var(--gold-line) !important; background: #202024 !important;
    outline: none !important;
}

/* ── 主按钮：鎏金 ── */
button.primary, .gr-button-primary {
    background: linear-gradient(180deg, var(--gold-2), var(--gold)) !important;
    color: #1A1509 !important; font-weight: 700 !important;
    border: none !important; border-radius: var(--r-sm) !important;
    font-size: 15px !important; letter-spacing: .02em !important;
    box-shadow: 0 2px 10px rgba(201,162,75,.18) !important;
    transition: filter .18s ease, transform .12s ease !important;
}
button.primary:hover, .gr-button-primary:hover { filter: brightness(1.08); transform: translateY(-1px); }
button.primary:active { transform: translateY(0); }

/* ── 次按钮：描边幽灵 ── */
button.secondary, .gr-button-secondary {
    background: transparent !important; color: var(--txt-dim) !important;
    border: 1px solid var(--line-2) !important; border-radius: var(--r-sm) !important;
    font-size: 14px !important; font-weight: 600 !important;
    transition: color .18s ease, border-color .18s ease !important;
}
button.secondary:hover, .gr-button-secondary:hover {
    color: var(--gold-2) !important; border-color: var(--gold-line) !important;
}

/* ── 标签：暗底 + 浅字（主要靠上面的 theme.set()，这里做兜底）── */
.gradio-container {
    --block-label-background-fill: #1C1C21 !important;
    --block-label-text-color: #D2CFC6 !important;
    --block-label-border-color: #2A2A30 !important;
    /* 组件标题(角色/音色/台词等标签) = 按钮同款哑光金 + 深字（这才是橙色的真正来源） */
    --block-title-background-fill: #C9A24B !important;
    --block-title-text-color: #1A1509 !important;
    --block-title-border-color: #C9A24B !important;
    --block-info-text-color: var(--txt-mute) !important;
}
.block-label, span.block-label, .block .block-label, [class*="block_label"], [class*="block-label"] {
    background: #1C1C21 !important;
    color: #D2CFC6 !important;
    border: 1px solid #2A2A30 !important; box-shadow: none !important;
    font-weight: 600 !important; letter-spacing: .02em !important;
}
.block-label svg, .block-label .icon { color: var(--gold) !important; opacity: .85; }
.label-text { color: var(--txt-dim) !important; }

/* ── 徽标 ── */
.badge {
    display: inline-block; background: var(--gold-soft); color: var(--gold);
    border: 1px solid var(--gold-line); border-radius: 999px;
    padding: 3px 13px; font-size: 11.5px; font-weight: 700;
}

/* ── Banner ── */
.studio-banner {
    background: linear-gradient(120deg, #1A1712 0%, var(--surface) 55%) !important;
    border: 1px solid var(--line);
    border-left: 3px solid var(--gold);
    border-radius: var(--r); padding: 18px 28px; margin-bottom: 32px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: var(--shadow);
}
.studio-banner .brand-mark {
    font-size: 30px; line-height: 1;
    width: 58px; height: 58px; flex: 0 0 58px;
    display: flex; align-items: center; justify-content: center;
    background: var(--gold-soft); border: 1px solid var(--gold-line);
    border-radius: 14px;
}
.studio-banner .logo-text {
    font-size: 26px; font-weight: 800; color: var(--txt);
    letter-spacing: .01em; margin-bottom: 4px;
}
.studio-banner .logo-text .accent { color: var(--gold); }
.studio-banner .sub { color: var(--txt-mute); font-size: 13px; letter-spacing: .04em; }

/* ── Markdown 说明文字 ── */
.prose, .prose p, .md { color: var(--txt-dim) !important; font-size: 13.5px !important; line-height: 1.75 !important; }
.prose strong, .md strong { color: var(--txt) !important; }
.prose code, .md code {
    background: var(--surface-2) !important; color: var(--gold-2) !important;
    border: 1px solid var(--line) !important; border-radius: 5px !important;
    padding: 1px 7px !important; font-size: 12.5px !important;
}

/* ── DataFrame（仅配色，不动结构） ── */
.gr-dataframe table th, table thead th {
    background: var(--surface-2) !important; color: var(--gold) !important;
    font-weight: 700 !important; letter-spacing: .03em !important;
    border-color: var(--line) !important;
}
.gr-dataframe table td, table tbody td {
    color: var(--txt-dim) !important; border-color: var(--line) !important;
}

/* ── 滑块 ── */
input[type="range"] { accent-color: var(--gold) !important; }

/* ── Accordion ── */
.label-wrap > span { color: var(--txt) !important; font-weight: 600 !important; }

/* ── 进度条 ── */
.progress-bar, .gr-progress .progress-bar { background: var(--gold) !important; }

/* ── Examples（胶囊标签） ── */
.gr-samples-table td, .examples td {
    background: var(--surface-2) !important; color: var(--txt-dim) !important;
    border: 1px solid var(--line) !important; border-radius: 999px !important;
    transition: color .18s ease, border-color .18s ease !important;
}
.gr-samples-table td:hover, .examples td:hover {
    color: var(--gold) !important; border-color: var(--gold-line) !important;
}

/* ── 滚动条 ── */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: var(--bg); }
*::-webkit-scrollbar-thumb { background: #2B2B30; border-radius: 999px; border: 2px solid var(--bg); }
*::-webkit-scrollbar-thumb:hover { background: #3A3A40; }

/* ── 底部 ── */
footer { visibility: hidden; }

/* ── 自定义页脚（署名 + 链接） ── */
.studio-footer {
    margin-top: 30px; padding: 18px 28px;
    background: linear-gradient(120deg, var(--surface) 60%, #1A1712 100%);
    border: 1px solid var(--line); border-top: 2px solid var(--gold);
    border-radius: var(--r); box-shadow: var(--shadow);
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 16px;
}
.studio-footer .ft-left { color: var(--txt-dim); font-size: 13px; letter-spacing: .05em; }
.studio-footer .ft-name {
    color: var(--gold); font-weight: 800; font-size: 15px;
    letter-spacing: .08em; margin-left: 4px;
}
.studio-footer .ft-links { display: flex; gap: 12px; flex-wrap: wrap; }
.studio-footer .ft-links a {
    color: var(--txt-dim); text-decoration: none;
    font-size: 13px; font-weight: 600; letter-spacing: .03em;
    padding: 9px 20px; border-radius: 999px;
    border: 1px solid var(--line-2);
    transition: color .18s ease, border-color .18s ease, background .18s ease;
}
.studio-footer .ft-links a:hover {
    color: #1A1509; background: var(--gold); border-color: var(--gold);
}
"""


# ── 本地内嵌 Inter 字体（自包含 / 离线 / 不依赖外网或系统字体） ──
def _build_font_css() -> str:
    """读取 assets/fonts 下的 Inter woff2，base64 内嵌为 @font-face。
    随包分发，离线可用；用 data URI 而非文件路径，避免任何机器上的路径/服务问题。"""
    import base64
    font_dir = APP_DIR / "assets" / "fonts"
    faces = []
    for weight in (400, 600, 700, 800):
        fp = font_dir / f"inter-latin-{weight}-normal.woff2"
        if not fp.exists():
            continue
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        faces.append(
            "@font-face{font-family:'Inter';font-style:normal;font-weight:%d;"
            "font-display:swap;src:url(data:font/woff2;base64,%s) format('woff2')}" % (weight, b64)
        )
    return "\n".join(faces)


CSS = _build_font_css() + "\n" + CSS

# ── 预设示例 ─────────────────────────────────────────
_VOICES_DIR = APP_DIR / "assets" / "voices"

EXAMPLES: list[tuple[str, str, str]] = [
    ("Mob boss · cold threat", str(_VOICES_DIR / "male_gangster.wav"),
     'An aging mob boss speaks in a gravelly, quiet voice, calm but pulsing with menace, '
     '"You thought you could walk through that door and leave alive?" '
     'He chuckles darkly, "Hahaha." He lets the silence stretch. His voice drops to a lethal murmur, '
     '"Young man... who gave you the nerve?"'),
    ("Furious · slams the table", str(_VOICES_DIR / "male_furious.wav"),
     'A man speaks through clenched teeth, barely containing his fury, each word deliberate and crushing, '
     '"I will ask you... one last time. Where. Is. It." '
     'His restraint snaps. His voice detonates into a full-throated roar, '
     '"Don\'t make me do this myself!"'),
    ("Stand-up · laughing breathless", str(_VOICES_DIR / "male_comedian.wav"),
     'A stand-up comedian gasps in theatrical disbelief, his voice shooting up, '
     '"No way! You actually said that?!" '
     'He erupts into helpless laughter, "Hahaha, hahaha." '
     'He wheezes, struggling for air, "Oh my god, I\'m dying, I can\'t breathe!" '
     'He breaks down again, "Hahaha."'),
    ("Late-night radio · soothing", str(_VOICES_DIR / "male_deep_warm.wav"),
     'A deep, warm male voice speaks into the mic, slow and unhurried, as if murmuring directly against the ear, '
     '"It\'s late. To everyone still awake out there, this next song is for you, carrying something heavy tonight." '
     'A long pause. His voice softens even further, dropping to a tender whisper, '
     '"You don\'t have to hold it all together... tonight, just get some sleep."'),
    ("Young hero · awakening", str(_VOICES_DIR / "male_young.wav"),
     'A young man\'s voice trembles, on the verge of tears, fragile and breaking, '
     '"Can I... can I really do this?" '
     'A long pause. His breathing steadies. His voice hardens with sudden, fierce resolve, '
     'each word landing like a blow, "Enough! This time, I am not running away!" '
     'He shouts the last line, his voice cracking with raw conviction.'),
    ("Drunk dad · late-night confession", str(_VOICES_DIR / "male_weathered.wav"),
     'A middle-aged man slurs his words, his voice thick with alcohol and exhaustion, heavy with guilt, '
     '"Son... your old man never amounted to much. I\'m sorry you had to suffer for it." '
     'He sighs deeply. A long pause. He speaks quieter now, his voice cracking, barely holding back tears, '
     '"You... you have to do better than me, okay?"'),
    ("Palace consort · smiling dagger", str(_VOICES_DIR / "female_refined.wav"),
     'A woman speaks in a silken voice, graceful and perfectly poised, dripping with honeyed insincerity, '
     '"What a lovely dress you\'re wearing, little sister." '
     'She lets out a soft, cold laugh, "Hehehe." '
     'Her voice drops to an icy whisper, venom wrapped in velvet, '
     '"Such a pity... I doubt you\'ll ever get to wear it again."'),
    ("Sweet & clingy · pleading", str(_VOICES_DIR / "female_sweet.wav"),
     'A young woman speaks in a bright, singsong voice, her tone lifting into a playful whine, '
     'every sentence curling upward, "Nooo, I don\'t care! You promised you\'d spend today with me!" '
     'She giggles, "Hehehe." She drags her words out in an exaggerated, pouty drawl, '
     '"Pleeease, let\'s go, let\'s go, pretty please?"'),
]


# ── 回调函数 ─────────────────────────────────────────

def on_generate(prompt: str, audio_ref, cfg: float, stg: float,
                dur_mult: float, gen_dur: float, ref_dur: float, seed: int,
                speed: float = 1.0):
    """Single line. The reference comes from the player below (a picked voice loads there, or upload one). speed = tempo."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    t0 = time.time()
    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    output = str(OUTPUT_DIR / f"output_{int(time.time())}.wav")
    tts.generate_to_file(
        prompt=prompt, output=output, voice_ref=ref_path,
        cfg_scale=cfg, stg_scale=stg,
        duration_multiplier=dur_mult, seed=int(seed),
        gen_duration=float(gen_dur), ref_duration=float(ref_dur),
    )
    # Tempo change without pitch shift (PyAV atempo, offline)
    if speed and abs(float(speed) - 1.0) >= 0.02:
        try:
            from audio_speed import change_speed
            output = change_speed(output, float(speed))
        except Exception as e:
            log.warning(f"Speed change failed, keeping original speed: {e}")
    dur = time.time() - t0
    log.info(f"Generated: {dur:.1f}s → {output}")
    return output


GEMMA_OUTPUT_CHOICES = [
    ("real — exact, lowest GPU peak (default)", "real"),
    ("full — exact, keeps Gemma's padded output (original behaviour)", "full"),
    ("project — least memory, ~0.5–1.4% rounding differences", "project"),
    ("project-exact — exact, little memory, ~7 GB peak on 8 GB cards", "project-exact"),
]


def on_options(trim: bool, gemma_output: str):
    """Speed / memory options from the inference settings; they apply to the next prompt in both tabs."""
    try:
        return f"✓ {tts.set_options(trim=trim, gemma_output=gemma_output)} — applies to the next prompt", gemma_output
    except ValueError as e:
        gr.Warning(str(e))
        return f"Unchanged: {tts.set_options()}", tts.gemma_output


def on_save_voice(name: str, audio) -> list:
    if not name or not name.strip():
        raise gr.Error("Please enter a voice name.")
    if audio is None:
        raise gr.Error("Please upload a reference clip.")
    save_voice(name.strip(), str(audio))
    return refresh_voice_table()


def on_delete_voice(name: str) -> list:
    if not name or not name.strip():
        raise gr.Error("Please enter the name of the voice to delete.")
    if not delete_voice(name.strip()):
        raise gr.Error(f"Voice \"{name}\" not found.")
    return refresh_voice_table()


def refresh_voice_table() -> list:
    voices = list_voices()
    if not voices:
        return []
    return [[v["name"], f'{v["duration_seconds"]:.1f}s', v["created_at"]] for v in voices]


def get_voice_choices() -> list:
    """[(name, path)] for the dropdown."""
    from voice_library import _resolve_path
    return [(v["name"], str(_resolve_path(v["path"]))) for v in list_voices()]


def on_quick_save_voice(name: str, audio_ref):
    """Save the uploaded reference as a voice on the Quick tab and refresh the dropdown."""
    if not name or not name.strip():
        raise gr.Error("Please enter a voice name first.")
    if not audio_ref:
        raise gr.Error("Please upload a reference clip before saving.")
    if save_voice(name.strip(), str(audio_ref)) is None:
        raise gr.Error("Save failed. Please check the audio file.")
    return gr.update(choices=get_voice_choices())


def on_refresh_voices():
    """Reload the voice dropdown (after a save on another tab)."""
    return gr.update(choices=get_voice_choices())


def on_delete_selected_voice(selected_path):
    """Delete the voice picked in the dropdown (its value is the file path; mapped back to the name)."""
    if not selected_path:
        raise gr.Error("Pick the voice to delete in the dropdown above first.")
    from voice_library import _resolve_path
    target = str(selected_path)
    name = next((v["name"] for v in list_voices()
                 if str(_resolve_path(v["path"])) == target), None)
    if name is None or not delete_voice(name):
        raise gr.Error("Voice not found. It may already have been deleted.")
    return gr.update(choices=get_voice_choices(), value=None)


def on_parse_script(text: str):
    """Parse the script → update the character assignment UI."""
    if not text or not text.strip():
        raise gr.Error("Please paste a script.")
    lines = parse_script(text)
    if not lines:
        raise gr.Error("No dialogue lines found.")
    chars = list(dict.fromkeys(item["character"] for item in lines if item["character"]))
    choices = get_voice_choices()
    updates = []
    for i in range(8):
        if i < len(chars):
            updates.append(gr.update(visible=True, value=chars[i], label=f"Character {i+1}"))
        else:
            updates.append(gr.update(visible=False, value=""))
    # Give each character a voice from the library in turn (empty = the built-in default voice)
    voice_paths = [c[1] for c in choices]
    dropdown_updates = []
    for i in range(8):
        if i < len(chars):
            default_voice = voice_paths[i % len(voice_paths)] if voice_paths else None
            dropdown_updates.append(gr.update(visible=True, choices=choices, value=default_voice))
        else:
            dropdown_updates.append(gr.update(visible=False, choices=choices))
    # Per-line emotion dropdowns for the first MAX_EMOTION_LINES lines, labelled with a preview
    emotion_updates = []
    for i in range(MAX_EMOTION_LINES):
        if i < len(lines):
            spk = lines[i]["character"] or "Narrator"
            text = lines[i]["line"]
            preview = text[:16] + ("…" if len(text) > 16 else "")
            emotion_updates.append(gr.update(visible=True, value="Normal",
                                             label=f"{i+1}. {spk}：{preview}"))
        else:
            emotion_updates.append(gr.update(visible=False))
    title_update = gr.update(visible=True)
    # store parsed data in state
    parsed_json = [{"character": item["character"], "line": item["line"]} for item in lines]
    status = f"Found {len(lines)} lines · {len(chars)} characters"
    if len(lines) > MAX_EMOTION_LINES:
        status += f" (emotion can be set for the first {MAX_EMOTION_LINES} lines; the rest use \"Normal\")"
    return updates + dropdown_updates + emotion_updates + [title_update, parsed_json, status]


def on_generate_dialogue(parsed_lines, speed, *dropdown_values):
    """Generate the whole dialogue. speed = tempo; dropdown_values = 8 voice dropdowns, then N per-line emotion dropdowns."""
    if not parsed_lines:
        raise gr.Error("Please parse the script first.")
    voices = dropdown_values[:8]
    emotions = dropdown_values[8:]
    # character → voice
    voice_map = {}
    chars = list(dict.fromkeys(item["character"] for item in parsed_lines if item["character"]))
    for i, char in enumerate(chars):
        if i < len(voices) and voices[i]:
            voice_map[char] = voices[i]
    # Attach each line's emotion (lines past the pickers use "Normal")
    lines_with_emotion = []
    for i, item in enumerate(parsed_lines):
        emo = emotions[i] if i < len(emotions) and emotions[i] else "Normal"
        lines_with_emotion.append({**item, "emotion": emo})
    from dialogue import generate_dialogue

    def progress_cb(cur, total, character, line):
        log.info(f"[{cur}/{total}] {character}: {line[:30]}...")

    out = generate_dialogue(lines_with_emotion, voice_map, tts, progress_cb)
    if out is None:
        raise gr.Error("Generation failed.")
    # Tempo change without pitch shift
    if speed and abs(float(speed) - 1.0) >= 0.02:
        try:
            from audio_speed import change_speed
            out = change_speed(out, float(speed))
        except Exception as e:
            log.warning(f"Speed change failed, keeping original speed: {e}")
    return out


# ══════════════════════════════════════════════════
with gr.Blocks(
    title="DramaBox Studio — AI Voice Studio",
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="gray",
        neutral_hue="gray",
        font=gr.themes.Font("Inter"),
        font_mono=gr.themes.Font("ui-monospace"),
    ).set(
        block_label_background_fill="#1C1C21",
        block_label_text_color="#D2CFC6",
        block_label_border_color="#2A2A30",
        # 组件标题(角色/音色/台词等标签) → 用按钮同款哑光金 + 深字（替换刺眼橙）
        block_title_background_fill="#C9A24B",
        block_title_text_color="#1A1509",
        block_title_border_color="#C9A24B",
        block_info_text_color="#66645C",
    ),
    css=CSS,
    analytics_enabled=False,
) as app:

    # ── Header ──
    gr.HTML(
        '<div class="studio-banner">'
        '<div class="brand-mark">🎭</div>'
        '<div class="brand-text">'
        '<div class="logo-text">DramaBox <span class="accent">Studio</span></div>'
        '<div class="sub">LTX-2.3 · Resemble AI DramaBox · fully local AI voice studio</div>'
        '</div>'
        '</div>'
    )

    with gr.Tabs(elem_classes=["tabs"]) as tabs:

        # ══════════════════════════════════════════
        # Tab 1: Quick
        # ══════════════════════════════════════════
        with gr.TabItem("🎤 Quick Generate", id="tab_quick"):
            gr.Markdown(
                "Write a scene prompt, optionally add a reference clip (10 s or more), and generate expressive speech. "
                "**Format:** dialogue goes inside `\"double quotes\"`, actions and directions outside them."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    prompt_box = gr.Textbox(
                        label="Scene prompt",
                        placeholder='A woman speaks warmly, "Hello, how are you today?" '
                                    'She laughs, "Hahaha, it is so good to see you!"',
                        lines=6,
                        elem_id="prompt_box",
                    )
                    gr.HTML("""
<img src="x" style="display:none" onerror="
window._cu=function(t){
  var ta=document.querySelector('#prompt_box textarea')||document.querySelector('#prompt_box input');
  if(!ta)return;
  var p=ta.selectionStart!=null?ta.selectionStart:ta.value.length;
  var b=ta.value.slice(0,p),a=ta.value.slice(p);
  var s=(p>0&&ta.value[p-1]!==' '&&ta.value[p-1]!=='\\n')?' ':'';
  var nv=b+s+t+a;
  var st=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;
  st.call(ta,nv);
  ta.dispatchEvent(new Event('input',{bubbles:true}));
  var np=p+s.length+t.length;
  ta.selectionStart=ta.selectionEnd=np;
  ta.focus();
};
window._qw=function(){
  var ta=document.querySelector('#prompt_box textarea')||document.querySelector('#prompt_box input');
  if(!ta)return;
  var p0=ta.selectionStart!=null?ta.selectionStart:ta.value.length;
  var p1=ta.selectionEnd!=null?ta.selectionEnd:p0;
  if(p0!==p1){
    var sel=ta.value.slice(p0,p1);
    var nv=ta.value.slice(0,p0)+'\\u201c'+sel+'\\u201d'+ta.value.slice(p1);
  }else{
    var nv=ta.value.slice(0,p0)+'\\u201c\\u201d'+ta.value.slice(p0);
  }
  var st=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;
  st.call(ta,nv);
  ta.dispatchEvent(new Event('input',{bubbles:true}));
  if(p0!==p1){
    ta.selectionStart=ta.selectionEnd=p0+2+sel.length;
  }else{
    ta.selectionStart=ta.selectionEnd=p0+1;
  }
  ta.focus();
};
">
<!-- speaker tags -->
<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
  <span style="color:var(--txt-dim);font-size:12.5px;font-weight:600;letter-spacing:.03em;margin-right:4px;">🎤 Speaker:</span>
  <span class="cue-tag" onclick="window._cu('A young woman speaks warmly, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Gentle woman</span>
  <span class="cue-tag" onclick="window._cu('A deep, authoritative man speaks coldly, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Commanding man</span>
  <span class="cue-tag" onclick="window._cu('A young man speaks with fierce energy, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Fiery youth</span>
  <span class="cue-tag" onclick="window._cu('A refined woman speaks in a silken voice, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Cool, refined woman</span>
  <span class="cue-tag" onclick="window._cu('A middle-aged man speaks in a gravelly voice, raspy and weary, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Weathered man</span>
  <span class="cue-tag" onclick="window._cu('A playful young woman speaks in a bright, singsong voice, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Sweet, playful girl</span>
  <span class="cue-tag" onclick="window._cu('A deep, warm male voice speaks gently, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Deep, warm man</span>
  <span class="cue-tag" onclick="window._cu('A menacing figure speaks in a low, cold voice, ')" style="
    background:var(--gold-soft);color:var(--gold);border:1px solid var(--gold-line);
    border-radius:999px;padding:5px 14px;font-size:12.5px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;user-select:none;
    white-space:nowrap;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='var(--gold-soft)';this.style.color='var(--gold)'">Sinister villain</span>
</div>
<!-- cue insert row -->
<div id="cue_insert_row" style="display:flex;align-items:center;gap:10px;margin-bottom:12px;
  background:var(--surface);border:1px solid var(--line);border-radius:var(--r-sm);
  padding:10px 14px;">
  <select id="cue_selector" style="flex:2;background:var(--surface-2);color:var(--txt);
    border:1px solid var(--line);border-radius:var(--r-sm);padding:9px 12px;
    font-size:13px;font-family:inherit;cursor:pointer;outline:none;
    transition:border-color .18s ease;"
    onfocus="this.style.borderColor='var(--gold-line)'"
    onblur="this.style.borderColor='var(--line)'"
    onkeydown="if(event.key==='Enter'){event.preventDefault();window._cu(this.value)}">
    <optgroup label="── Emotional actions ──">
      <option value="She sighs deeply. ">Sigh — She sighs deeply.</option>
      <option value="He gasps. ">Gasp — He gasps.</option>
      <option value="He wheezes, struggling for air. ">Out of breath — He wheezes, struggling for air.</option>
      <option value="She groans. ">Groan — She groans.</option>
      <option value="She breathes deeply. ">Deep breath — She breathes deeply.</option>
      <option value="He sniffles. ">Sniffle — He sniffles.</option>
      <option value='"Hahaha." '>Laugh — "Hahaha."</option>
      <option value='"Hehehe." '>Giggle — "Hehehe."</option>
    </optgroup>
    <optgroup label="── Pauses & rhythm ──">
      <option value="A long pause. ">Long pause — A long pause.</option>
      <option value="He clears his throat. ">Clear throat — He clears his throat.</option>
      <option value="He gulps nervously. ">Nervous gulp — He gulps nervously.</option>
      <option value="He coughs. ">Cough — He coughs.</option>
      <option value="He yawns. ">Yawn — He yawns.</option>
    </optgroup>
    <optgroup label="── Voice changes ──">
      <option value="She drops to a whisper. ">Whisper — She drops to a whisper.</option>
      <option value="He mutters. ">Mutter — He mutters.</option>
      <option value="He mumbles. ">Mumble — He mumbles.</option>
      <option value="His voice trembles. ">Trembling voice — His voice trembles.</option>
      <option value="His voice cracks. ">Voice cracks — His voice cracks.</option>
    </optgroup>
  </select>
  <button style="flex:0 0 auto;background:linear-gradient(180deg,var(--gold-2),var(--gold));
    color:#1A1509;font-weight:700;border:none;border-radius:var(--r-sm);padding:9px 18px;
    font-size:13px;letter-spacing:.02em;cursor:pointer;
    box-shadow:0 2px 10px rgba(201,162,75,.18);
    transition:filter .18s ease,transform .12s ease;"
    onmouseover="this.style.filter='brightness(1.08)'"
    onmouseout="this.style.filter='none'"
    onmousedown="this.style.transform='translateY(0)'"
    onmouseup="this.style.transform='translateY(-1px)'"
    onclick="window._cu(document.getElementById('cue_selector').value)">
    + Insert at cursor</button>
  <button style="flex:0 0 auto;background:transparent;color:var(--gold);
    border:1px solid var(--gold-line);border-radius:var(--r-sm);padding:9px 16px;
    font-size:13px;font-weight:700;cursor:pointer;
    transition:background .18s,color .18s;"
    onmouseover="this.style.background='var(--gold)';this.style.color='#1A1509'"
    onmouseout="this.style.background='transparent';this.style.color='var(--gold)'"
    onclick="window._qw()">
    「」→ \" \"
  </button>
</div>
""")
                    with gr.Row():
                        quick_voice = gr.Dropdown(
                            label="Saved voices (picking one loads it below so you can listen)",
                            choices=get_voice_choices(), value=None,
                            interactive=True, scale=5,
                        )
                        refresh_voice_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                        delete_voice_btn = gr.Button("🗑️ Delete selected", variant="secondary", scale=1)
                    audio_ref = gr.Audio(
                        label="Reference clip · playable (a voice picked above loads here, or upload 10 s+)",
                        type="filepath",
                    )
                    with gr.Row():
                        quick_voice_name = gr.Textbox(
                            label="Save the uploaded clip as a voice", placeholder="Name the voice, e.g. Heroine",
                            scale=4,
                        )
                        quick_save_btn = gr.Button("💾 Save voice", variant="secondary", scale=1)
                    gen_btn = gr.Button("✦ Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Accordion("⚙️ Inference settings", open=False):
                        cfg_slider = gr.Slider(1.0, 10.0, value=CFG_SCALE, step=0.5, label="CFG guidance scale")
                        stg_slider = gr.Slider(0.0, 5.0, value=STG_SCALE, step=0.5, label="STG (skip-token guidance)")
                        dur_slider = gr.Slider(0.8, 2.0, value=1.1, step=0.05,
                                               label="Duration multiplier (used when target duration = 0)")
                        gen_dur_slider = gr.Slider(0.0, 60.0, value=0.0, step=1.0,
                                                   label="Target duration (s) — 0 = auto")
                        ref_dur_slider = gr.Slider(3.0, 30.0, value=10.0, step=1.0, label="Reference length used (s)")
                        speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                                 label="Speed (1 = original, >1 faster, <1 slower; pitch unchanged)")
                        seed_input = gr.Number(value=SEED, label="Seed", precision=0)
                        gr.Markdown("**Speed / memory**")
                        trim_cb = gr.Checkbox(
                            value=tts.trim, interactive=not tts.ALWAYS_TRIM,
                            label="Padding trim — Gemma runs on the prompt's real tokens only "
                                  "(faster; ~0.5% rounding differences; always on with Apple Silicon)")
                        gemma_dd = gr.Dropdown(choices=GEMMA_OUTPUT_CHOICES, value=tts.gemma_output,
                                               label="Gemma output (how the text encoder hands off to the embeddings processor)")
                        opts_status = gr.Markdown(f"Current: {tts.set_options()}")
                        gr.Markdown(
                            f"Tier **{TIER}** · parking **{PARK}** · DiT INT8 **{'on' if USE_INT8 else 'off'}** — "
                            "set with `DRAMABOX_TIER` / `DRAMABOX_PARK` / `DRAMABOX_INT8` before starting "
                            "(restart needed).")
                    audio_out = gr.Audio(label="Result", type="filepath")

            trim_cb.input(on_options, inputs=[trim_cb, gemma_dd], outputs=[opts_status, gemma_dd])
            gemma_dd.input(on_options, inputs=[trim_cb, gemma_dd], outputs=[opts_status, gemma_dd])
            gen_btn.click(
                on_generate,
                inputs=[prompt_box, audio_ref, cfg_slider, stg_slider,
                        dur_slider, gen_dur_slider, ref_dur_slider, seed_input, speed_slider],
                outputs=[audio_out],
            )

            # Picking a saved voice loads it into the reference player
            quick_voice.change(lambda p: p, inputs=[quick_voice], outputs=[audio_ref])
            # Save the current reference as a voice / refresh the dropdown
            quick_save_btn.click(on_quick_save_voice,
                                 inputs=[quick_voice_name, audio_ref], outputs=[quick_voice])
            refresh_voice_btn.click(on_refresh_voices, outputs=[quick_voice])
            # Delete the selected voice and refresh the dropdown
            delete_voice_btn.click(on_delete_selected_voice,
                                   inputs=[quick_voice], outputs=[quick_voice])

            gr.Examples(
                label="📋 Click an example to try it",
                examples=[
                    [name, prompt, voice_path, CFG_SCALE, STG_SCALE, 1.1, 0.0, 10.0, SEED]
                    for name, voice_path, prompt in EXAMPLES
                ],
                example_labels=[name for name, _, _ in EXAMPLES],
                inputs=[gr.Textbox(visible=False, label="Scene"),
                        prompt_box, audio_ref,
                        cfg_slider, stg_slider, dur_slider, gen_dur_slider,
                        ref_dur_slider, seed_input],
                outputs=[audio_out],
                fn=lambda _n, prompt, ref, cfg, stg, dur, gd, rd, sd: on_generate(
                    prompt, ref, cfg, stg, dur, gd, rd, sd),
                cache_examples=False, run_on_click=False, examples_per_page=20,
            )

        # ══════════════════════════════════════════
        # Tab 2: Dialogue
        # ══════════════════════════════════════════
        with gr.TabItem("🎬 Dialogue Workshop", id="tab_dialogue"):
            gr.Markdown(
                "Paste a script → characters are detected and **voices assigned from your library** (change them in the dropdowns) → **pick an emotion per line** → generate.\n\n"
                "**Script format:** `Name: line`, one line each (the Chinese colon `：` works too)."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    script_input = gr.Textbox(
                        label="Script",
                        placeholder='Alex: Long time no see, brother!\nSam: Yeah, almost ten years now.\nAlex: Do you remember that time we...',
                        lines=12,
                    )
                    with gr.Row():
                        parse_btn = gr.Button("🔍 Parse characters", variant="primary")
                    parse_status = gr.Markdown("")

                with gr.Column(scale=1):
                    gr.Markdown("**Character → voice**")
                    char_names = []
                    voice_drops = []
                    for i in range(8):
                        with gr.Row():
                            cn = gr.Textbox(label=f"Character {i+1}", visible=False, interactive=False, scale=1)
                            vd = gr.Dropdown(label="Voice", visible=False, interactive=True, scale=1)
                            char_names.append(cn)
                            voice_drops.append(vd)

                    emotion_title = gr.Markdown(
                        "**🎭 Emotion per line** · pick an emotion for each line (default Normal)"
                        "  |  options marked (female) are for female voices only, or the voice may change", visible=False)
                    emotion_drops = []
                    for i in range(MAX_EMOTION_LINES):
                        ed = gr.Dropdown(
                            choices=EMOTION_LABELS, value="Normal",
                            label=f"Line {i+1}", visible=False, interactive=True,
                        )
                        emotion_drops.append(ed)

                    dlg_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05,
                                          label="Speed (1 = original, >1 faster, <1 slower; pitch unchanged)")
                    dialogue_gen_btn = gr.Button("🎬 Generate full dialogue", variant="primary", visible=False)
                    dialogue_out = gr.Audio(label="Full dialogue", type="filepath")

            # State to hold parsed lines between parse and generate
            parsed_state = gr.State([])

            all_char_outputs = char_names + voice_drops
            parse_btn.click(
                on_parse_script,
                inputs=[script_input],
                outputs=all_char_outputs + emotion_drops + [emotion_title, parsed_state, parse_status],
            ).then(
                lambda: gr.update(visible=True),
                outputs=[dialogue_gen_btn],
            )

            dialogue_gen_btn.click(
                on_generate_dialogue,
                inputs=[parsed_state, dlg_speed] + voice_drops + emotion_drops,
                outputs=[dialogue_out],
            )

    # ── Footer ──
    gr.HTML(
        '<div class="studio-footer">'
        '<div class="ft-left">Integration package by <span class="ft-name">Wang Zhifeng</span></div>'
        '<div class="ft-links">'
        '<a href="https://wangzhifeng.vip/" target="_blank" rel="noopener">More AI tools</a>'
        '<a href="https://wangzhifeng.vip/" target="_blank" rel="noopener">Tutorial</a>'
        '</div>'
        '</div>'
    )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", str(GRADIO_PORT)))
    app.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=port,
        inbrowser=True,
    )
