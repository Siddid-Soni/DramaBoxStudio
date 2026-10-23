// DramaBox Studio · Night City HUD — talks to the JSON API in app.py
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid);
  return n;
};

const store = {
  get(k, fallback) { try { const v = localStorage.getItem("dbx:" + k); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem("dbx:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

async function api(path, { method = "GET", json, form } = {}) {
  const opts = { method };
  if (json !== undefined) { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(json); }
  if (form) opts.body = form;
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
    throw new Error(msg || `${r.status} ${r.statusText}`);
  }
  return data;
}

let toastTimer;
function toast(msg, ok = false) {
  const t = $("#toast");
  t.textContent = (ok ? "✓ " : "✕ ") + msg;
  t.classList.toggle("ok", ok);
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ok ? 2600 : 6000);
}

const fmtTime = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
};
const pad = (n, w = 2) => String(n).padStart(w, "0");

// ── Presets the prompt editor offers ─────────────────
const SPEAKERS = [
  ["Gentle woman", "A young woman speaks warmly, "],
  ["Commanding man", "A deep, authoritative man speaks coldly, "],
  ["Fiery youth", "A young man speaks with fierce energy, "],
  ["Refined woman", "A refined woman speaks in a silken voice, "],
  ["Weathered man", "A middle-aged man speaks in a gravelly voice, raspy and weary, "],
  ["Playful girl", "A playful young woman speaks in a bright, singsong voice, "],
  ["Deep, warm man", "A deep, warm male voice speaks gently, "],
  ["Sinister villain", "A menacing figure speaks in a low, cold voice, "],
];
const CUES = [
  ["Sigh", "She sighs deeply. "], ["Gasp", "He gasps. "], ["Out of breath", "He wheezes, struggling for air. "],
  ["Groan", "She groans. "], ["Deep breath", "She breathes deeply. "], ["Sniffle", "He sniffles. "],
  ["Laugh", '"Hahaha." '], ["Giggle", '"Hehehe." '],
  ["Long pause", "A long pause. "], ["Clear throat", "He clears his throat. "], ["Nervous gulp", "He gulps nervously. "],
  ["Cough", "He coughs. "], ["Yawn", "He yawns. "],
  ["Whisper", "She drops to a whisper. "], ["Mutter", "He mutters. "], ["Mumble", "He mumbles. "],
  ["Trembling", "His voice trembles. "], ["Voice cracks", "His voice cracks. "],
];

const PARAMS = {
  cfg: { fmt: (v) => v.toFixed(1) },
  stg: { fmt: (v) => v.toFixed(1) },
  speed: { fmt: (v) => v.toFixed(2) + "×", def: 1 },
  gen_dur: { fmt: (v) => (v ? v + " S" : "AUTO"), def: 0 },
  dur_mult: { fmt: (v) => v.toFixed(2) + "×", def: 1.1 },
  ref_dur: { fmt: (v) => v + " S", def: 10 },
};

const S = {
  voices: [],
  selectedRef: store.get("ref", ""),
  upload: null,
  params: {},
  status: null,
  current: null,
  busy: false,
  dlg: { lines: [], characters: [], cast: {} },
};

// ══════════════════════════════════════════════════
// Views / FX
// ══════════════════════════════════════════════════

function setView(view) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-on", t.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== view; });
  document.body.classList.remove("view-generate", "view-dialogue", "view-voices");
  document.body.classList.add("view-" + view);
  store.set("view", view);
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setView(t.dataset.view)));

function setFx(on) {
  document.body.classList.toggle("fx-off", !on);
  $("#fx-toggle").setAttribute("aria-pressed", String(on));
  store.set("fx", on);
}
$("#fx-toggle").addEventListener("click", () => setFx(document.body.classList.contains("fx-off")));

// ══════════════════════════════════════════════════
// Status / ticker
// ══════════════════════════════════════════════════

async function pollStatus() {
  try {
    S.status = await api("/api/status");
    renderStatus();
  } catch {
    const st = $("#sys-state");
    st.className = "sys-state down";
    st.querySelector("b").textContent = "OFFLINE";
  }
}

function renderStatus() {
  const s = S.status;
  const st = $("#sys-state");
  const { running, queued } = s.jobs;
  st.className = "sys-state " + (running ? "busy" : "ready");
  st.querySelector("b").textContent = running ? (queued ? `RENDERING · ${queued} QUEUED` : "RENDERING") : "SYS READY";
  $("#sys-vram").textContent = s.vram ? `VRAM ${s.vram.used_gb}/${s.vram.total_gb} GB` : s.backend.toUpperCase();

  const parts = [
    "LTX-2.3 ONLINE", `TIER ${s.tier.toUpperCase()}`, `DIT INT8 ${s.int8 ? "ON" : "OFF"}`,
    `GEMMA OUT: ${s.gemma_output.toUpperCase()}`, `TRIM ${s.trim ? "ON" : "OFF"}`, `PARKING: ${s.park.toUpperCase()}`,
    s.vram ? `VRAM ${s.vram.used_gb}/${s.vram.total_gb} GB` : s.backend.toUpperCase(),
    `${S.voices.length} VOICES LOADED`, "ALL PROCESSING LOCAL",
  ];
  const line = "// " + parts.join(" // ") + " //";
  $("#ticker-a").textContent = line;
  $("#ticker-b").textContent = line;
}

function renderOptions() {
  const s = S.status;
  const trim = $("#opt-trim");
  trim.checked = s.trim;
  trim.disabled = s.always_trim;
  const sel = $("#opt-gemma");
  sel.replaceChildren(...s.gemma_choices.map((c) => el("option", { value: c.value, text: `${c.value} — ${c.label}` })));
  sel.value = s.gemma_output;
  $("#tier-info").textContent = `TIER ${s.tier} · PARKING ${s.park} · DIT INT8 ${s.int8 ? "on" : "off"} — set DRAMABOX_TIER / DRAMABOX_PARK / DRAMABOX_INT8 before starting (restart needed).`;
}

async function setOptions(patch) {
  try {
    const r = await api("/api/options", { method: "POST", json: patch });
    $("#opt-status").textContent = `✓ ${r.message} — applies to the next prompt`;
    S.status.trim = r.trim;
    S.status.gemma_output = r.gemma_output;
  } catch (e) {
    toast(e.message);
  }
  renderOptions();
  renderStatus();
}
$("#opt-trim").addEventListener("change", (e) => setOptions({ trim: e.target.checked }));
$("#opt-gemma").addEventListener("change", (e) => setOptions({ gemma_output: e.target.value }));

// ══════════════════════════════════════════════════
// Voices
// ══════════════════════════════════════════════════

const preview = new Audio();
let previewRef = null;
function togglePreview(ref, btn) {
  document.querySelectorAll(".voice-play.is-playing").forEach((b) => b.classList.remove("is-playing"));
  if (previewRef === ref && !preview.paused) { preview.pause(); previewRef = null; return; }
  preview.src = "/api/voices/audio?ref=" + encodeURIComponent(ref);
  preview.play().catch(() => toast("Could not play that clip."));
  previewRef = ref;
  btn && btn.classList.add("is-playing");
}
preview.addEventListener("ended", () => {
  previewRef = null;
  document.querySelectorAll(".voice-play.is-playing").forEach((b) => b.classList.remove("is-playing"));
});

const PLAY_SVG = () => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("width", "11"); s.setAttribute("height", "11"); s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("fill", "currentColor"); s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", "M7 4v16l13-8z");
  s.append(p);
  return s;
};

function voiceLabel(ref) {
  if (!ref) return "Default voice";
  if (S.upload && S.upload.ref === ref) return "Upload · " + S.upload.name;
  const v = S.voices.find((x) => x.ref === ref);
  return v ? v.name : ref.replace(/^\w+:/, "");
}

async function loadVoices() {
  try {
    S.voices = await api("/api/voices");
  } catch (e) {
    toast(e.message);
  }
  renderVoiceList();
  renderVoiceGrid();
  renderCast();
  if (S.status) renderStatus();
}

function selectVoice(ref) {
  S.selectedRef = ref;
  store.set("ref", ref);
  renderVoiceList();
  $("#ref-label").textContent = voiceLabel(ref).toUpperCase();
}

function renderVoiceList() {
  const q = $("#bank-search").value.trim().toLowerCase();
  const match = (name) => !q || name.toLowerCase().includes(q);
  const list = $("#voice-list");
  const rows = [];
  let i = 0;
  const row = (ref, name, tag, playable = true) => {
    i += 1;
    const on = S.selectedRef === ref;
    return el("div", { class: "voice" + (on ? " is-on" : "") },
      el("button", { class: "voice-sel", role: "option", "aria-selected": String(on), onclick: () => selectVoice(ref) },
        el("span", { class: "voice-idx", text: pad(i) }),
        el("span", { class: "voice-body" }, el("span", { class: "voice-name", text: name }), el("span", { class: "voice-tag", text: tag }))),
      playable ? el("button", { class: "voice-play", "aria-label": "Preview " + name, onclick: (e) => togglePreview(ref, e.currentTarget) }, PLAY_SVG()) : null);
  };
  const group = (title, items) => {
    if (!items.length) return;
    rows.push(el("div", { class: "voice-group", text: title }));
    rows.push(...items);
  };

  if (S.upload && match(S.upload.name)) group("// UPLOAD", [row(S.upload.ref, S.upload.name, `${S.upload.duration.toFixed(1)} s · unsaved`)]);
  if (match("default voice")) group("// MODEL", [row("", "Default voice", "no reference", false)]);
  const lib = S.voices.filter((v) => !v.builtin && match(v.name));
  const builtin = S.voices.filter((v) => v.builtin && match(v.name));
  group("// LIBRARY", lib.map((v) => row(v.ref, v.name, `${v.duration.toFixed(1)} s · saved ${v.created.slice(0, 10)}`)));
  group("// BUILT-IN", builtin.map((v) => row(v.ref, v.name, `${v.duration.toFixed(1)} s · bundled`)));
  if (!rows.length) rows.push(el("div", { class: "log-empty", text: "NO MATCH" }));
  list.replaceChildren(...rows);
  $("#bank-count").textContent = `[${pad(S.voices.length)}]`;
}
$("#bank-search").addEventListener("input", renderVoiceList);

async function uploadClip(file) {
  const form = new FormData();
  form.append("file", file);
  return api("/api/uploads", { method: "POST", form });
}

async function onRefFile(file) {
  if (!file) return;
  try {
    S.upload = await uploadClip(file);
    if (S.upload.duration && S.upload.duration < 10) toast(`Clip is ${S.upload.duration.toFixed(1)} s — 10 s or longer works best.`);
    $("#upload-name").value = S.upload.name;
    $("#upload-saver").hidden = false;
    selectVoice(S.upload.ref);
  } catch (e) {
    toast(e.message);
  }
}
$("#ref-file").addEventListener("change", (e) => { onRefFile(e.target.files[0]); e.target.value = ""; });
const dz = $("#dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("is-over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("is-over"));
dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("is-over"); onRefFile(e.dataTransfer.files[0]); });

$("#upload-save").addEventListener("click", async () => {
  if (!S.upload) return;
  try {
    const r = await api("/api/voices", { method: "POST", json: { name: $("#upload-name").value, ref: S.upload.ref } });
    S.upload = null;
    $("#upload-saver").hidden = true;
    await loadVoices();
    selectVoice(r.ref);
    toast("Voice saved to the bank.", true);
  } catch (e) {
    toast(e.message);
  }
});

// Voice manager view
function renderVoiceGrid() {
  const grid = $("#voice-grid");
  grid.replaceChildren(...S.voices.map((v) => el("div", { class: "panel vcard" },
    el("span", { class: "tick" }),
    el("span", { class: "badge" + (v.builtin ? "" : " lib"), text: v.builtin ? "BUILT-IN" : "LIBRARY" }),
    el("div", { class: "vcard-name", text: v.name }),
    el("div", { class: "vcard-meta", text: `${v.duration.toFixed(1)} S${v.created ? " · " + v.created : ""}` }),
    el("div", { class: "vcard-actions" },
      el("button", { class: "voice-play", "aria-label": "Preview " + v.name, onclick: (e) => togglePreview(v.ref, e.currentTarget) }, PLAY_SVG()),
      el("button", { class: "chip", text: "USE", onclick: () => { selectVoice(v.ref); setView("generate"); } }),
      v.builtin ? null : el("button", { class: "chip cue", text: "DELETE", onclick: () => deleteVoice(v) })))));
}

async function deleteVoice(v) {
  if (!confirm(`Delete the voice "${v.name}"? Its clip file is removed too.`)) return;
  try {
    await api("/api/voices/" + encodeURIComponent(v.name), { method: "DELETE" });
    if (S.selectedRef === v.ref) selectVoice("");
    await loadVoices();
    toast("Voice deleted.", true);
  } catch (e) {
    toast(e.message);
  }
}

let vmFile = null;
$("#vm-file").addEventListener("change", (e) => {
  vmFile = e.target.files[0] || null;
  $("#vm-file-name").textContent = vmFile ? vmFile.name : "NO FILE";
  if (vmFile && !$("#vm-name").value) $("#vm-name").value = vmFile.name.replace(/\.[^.]+$/, "");
});
$("#vm-save").addEventListener("click", async () => {
  if (!vmFile) return toast("Choose a reference clip first.");
  try {
    const up = await uploadClip(vmFile);
    await api("/api/voices", { method: "POST", json: { name: $("#vm-name").value, ref: up.ref } });
    vmFile = null;
    $("#vm-file").value = "";
    $("#vm-file-name").textContent = "NO FILE";
    $("#vm-name").value = "";
    await loadVoices();
    toast("Voice added.", true);
  } catch (e) {
    toast(e.message);
  }
});

// ══════════════════════════════════════════════════
// Prompt editor
// ══════════════════════════════════════════════════

const prompt = $("#prompt");
const promptHl = $("#prompt-hl");
const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function renderPrompt() {
  const text = prompt.value;
  const html = escapeHtml(text).replace(/("[^"\n]*"?|“[^”\n]*”?)/g, '<span class="q">$1</span>');
  promptHl.innerHTML = html + (text.endsWith("\n") ? " " : "");
  promptHl.scrollTop = prompt.scrollTop;
  $("#char-count").textContent = text.length;
  store.set("prompt", text);
}
prompt.addEventListener("input", renderPrompt);
prompt.addEventListener("scroll", () => { promptHl.scrollTop = prompt.scrollTop; });
new ResizeObserver(() => { promptHl.scrollTop = prompt.scrollTop; }).observe(prompt);

function insertText(text) {
  prompt.focus();
  // execCommand keeps the edit on the undo stack; setRangeText is the fallback
  if (!document.execCommand("insertText", false, text)) {
    prompt.setRangeText(text, prompt.selectionStart, prompt.selectionEnd, "end");
  }
  renderPrompt();
}

function insertAtCursor(text) {
  const p = prompt.selectionStart ?? prompt.value.length;
  const before = prompt.value[p - 1];
  insertText((p > 0 && before !== " " && before !== "\n" ? " " : "") + text);
}

$("#speaker-chips").replaceChildren(...SPEAKERS.map(([label, text]) =>
  el("button", { class: "chip", text: label, title: text.trim(), onclick: () => insertAtCursor(text) })));
$("#cue-chips").replaceChildren(...CUES.map(([label, text]) =>
  el("button", { class: "chip cue", text: "+" + label, title: text.trim(), onclick: () => insertAtCursor(text) })));

$("#wrap-quotes").addEventListener("click", () => {
  const a = prompt.selectionStart, b = prompt.selectionEnd;
  const sel = prompt.value.slice(a, b);
  insertText('"' + sel + '"');
  if (!sel) prompt.selectionStart = prompt.selectionEnd = a + 1;
});
$("#clear-prompt").addEventListener("click", () => { prompt.select(); insertText(""); });

prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); }
});

// ══════════════════════════════════════════════════
// Params
// ══════════════════════════════════════════════════

function setParam(key, value) {
  const box = document.querySelector(`.param[data-key="${key}"]`);
  const input = box.querySelector("input");
  input.value = value;
  const v = parseFloat(input.value);
  S.params[key] = v;
  box.querySelector("b").textContent = PARAMS[key].fmt(v);
  input.style.setProperty("--p", ((v - input.min) / (input.max - input.min)) * 100 + "%");
}

function setSeed(v) {
  S.params.seed = parseInt(v, 10) || 0;
  $("#seed").value = S.params.seed;
}

function renderParamSummary() {
  const p = S.params;
  $("#param-summary").textContent = `CFG ${p.cfg.toFixed(1)} · STG ${p.stg.toFixed(1)} · ${p.speed.toFixed(2)}× · SEED ${p.seed}`;
  store.set("params", p);
}

function initParams(defaults) {
  const saved = store.get("params", {});
  for (const key of Object.keys(PARAMS)) {
    const fallback = key in defaults ? defaults[key] : PARAMS[key].def;
    setParam(key, saved[key] ?? fallback);
    document.querySelector(`.param[data-key="${key}"] input`).addEventListener("input", (e) => {
      setParam(key, e.target.value);
      renderParamSummary();
    });
  }
  setSeed(saved.seed ?? defaults.seed);
  renderParamSummary();
}
$("#seed").addEventListener("change", (e) => { setSeed(e.target.value); renderParamSummary(); });
$("#seed-roll").addEventListener("click", () => { setSeed(Math.floor(Math.random() * 2 ** 31)); renderParamSummary(); });

// ══════════════════════════════════════════════════
// Presets
// ══════════════════════════════════════════════════

async function loadPresets() {
  try {
    const items = await api("/api/examples");
    $("#preset-grid").replaceChildren(...items.map((x, i) => el("button", {
      class: "preset",
      onclick: () => {
        prompt.value = x.prompt;
        renderPrompt();
        selectVoice(x.voice);
        prompt.focus();
      },
    }, el("small", { text: `P/${pad(i + 1)}` }), el("strong", { text: x.title }), el("span", { text: x.subtitle }))));
  } catch (e) {
    toast(e.message);
  }
}

// ══════════════════════════════════════════════════
// Player
// ══════════════════════════════════════════════════

const audio = $("#audio");
const canvas = $("#wave-canvas");
let peaks = null;
let audioCtx = null;
let rafId = 0;

async function computePeaks(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const data = (await audioCtx.decodeAudioData(buf)).getChannelData(0);
  const n = 160;
  const step = Math.max(1, Math.floor(data.length / n));
  const out = new Float32Array(n);
  let max = 0;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = i * step, end = Math.min(data.length, j + step); j < end; j += 4) m = Math.max(m, Math.abs(data[j]));
    out[i] = m;
    max = Math.max(max, m);
  }
  return out.map((v) => v / (max || 1));
}

function drawWave() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (!peaks) return;
  const frac = audio.duration ? audio.currentTime / audio.duration : 0;
  const bars = Math.max(20, Math.floor(w / 4));
  const fx = !document.body.classList.contains("fx-off");
  for (let i = 0; i < bars; i++) {
    const v = peaks[Math.floor((i / bars) * peaks.length)];
    const bh = Math.max(3, v * (h - 6));
    const lit = i / bars < frac;
    g.fillStyle = lit ? "#00f0ff" : "rgba(0,240,255,.22)";
    g.shadowColor = lit && fx ? "rgba(0,240,255,.8)" : "transparent";
    g.shadowBlur = lit && fx ? 6 : 0;
    g.fillRect(i * 4, (h - bh) / 2, 2, bh);
  }
  const ph = $("#playhead");
  ph.hidden = false;
  ph.style.left = `calc(${frac * 100}% - 1px)`;
}

function tick() {
  drawWave();
  $("#time").innerHTML = `${fmtTime(audio.currentTime)}<span class="dim"> / </span>${fmtTime(audio.duration)}`;
  if (!audio.paused) rafId = requestAnimationFrame(tick);
}

function setPlayIcon(playing) {
  $("#play-icon").setAttribute("d", playing ? "M6 4h4v16H6zM14 4h4v16h-4z" : "M7 4v16l13-8z");
  $("#play").setAttribute("aria-label", playing ? "Pause" : "Play");
}

$("#play").addEventListener("click", () => { if (audio.paused) audio.play(); else audio.pause(); });
audio.addEventListener("play", () => { setPlayIcon(true); cancelAnimationFrame(rafId); tick(); });
audio.addEventListener("pause", () => { setPlayIcon(false); tick(); });
audio.addEventListener("ended", () => { setPlayIcon(false); tick(); });
audio.addEventListener("loadedmetadata", tick);
$("#wave").addEventListener("click", (e) => {
  if (!audio.duration) return;
  const r = canvas.getBoundingClientRect();
  audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
  tick();
});
window.addEventListener("resize", drawWave);

function takeTitle(meta) {
  if (meta.kind === "dialogue") return `Dialogue · ${meta.lines.length} lines`;
  if (meta.kind === "take") {
    const quote = (meta.prompt.match(/"([^"]+)"?/) || [])[1];
    return `${voiceLabel(meta.voice)} · ${quote ? `"${quote.slice(0, 32)}${quote.length > 32 ? "…" : ""}"` : meta.prompt.slice(0, 36)}`;
  }
  return meta.file;
}

function takeMeta(meta) {
  if (meta.kind === "take") return `SEED ${meta.seed} · CFG ${meta.cfg} · ${Number(meta.speed).toFixed(2)}×`;
  if (meta.kind === "dialogue") return `${new Set(meta.lines.map((l) => l.character)).size} VOICES · ${Number(meta.speed).toFixed(2)}×`;
  return meta.created;
}

async function loadOutput(meta, autoplay = false) {
  S.current = meta;
  audio.src = meta.url;
  $("#out-title").textContent = meta.file.toUpperCase();
  $("#out-meta").textContent = meta.render_s != null ? `RENDER ${meta.render_s}S` : meta.created || "";
  $("#wave-empty").hidden = true;
  $("#play").disabled = false;
  const dl = $("#download");
  dl.href = meta.url;
  dl.setAttribute("download", meta.file);
  dl.hidden = false;
  $("#reuse").hidden = meta.kind === "legacy";
  peaks = null;
  drawWave();
  renderLog();
  try {
    peaks = await computePeaks(meta.url);
  } catch {
    peaks = new Float32Array(160).fill(0.4);
  }
  tick();
  if (autoplay) audio.play().catch(() => { /* needs a user gesture first */ });
}

$("#reuse").addEventListener("click", () => {
  const m = S.current;
  if (!m) return;
  if (m.kind === "take") {
    prompt.value = m.prompt;
    renderPrompt();
    selectVoice(m.voice);
    for (const key of Object.keys(PARAMS)) if (m[key] != null) setParam(key, m[key]);
    setSeed(m.seed);
    renderParamSummary();
    setView("generate");
  } else if (m.kind === "dialogue") {
    S.dlg = {
      lines: m.lines.map((l) => ({ ...l })),
      characters: [...new Set(m.lines.map((l) => l.character))],
      cast: { ...m.voices },
    };
    $("#dlg-speed").value = m.speed;
    $("#dlg-speed").dispatchEvent(new Event("input"));
    renderDialogue();
    setView("dialogue");
  }
  toast("Settings loaded.", true);
});

// ══════════════════════════════════════════════════
// Take log
// ══════════════════════════════════════════════════

let logItems = [];
async function loadLog() {
  try {
    logItems = await api("/api/outputs");
  } catch (e) {
    toast(e.message);
  }
  renderLog();
}

function renderLog() {
  const box = $("#log");
  if (!logItems.length) {
    box.replaceChildren(el("div", { class: "log-empty", text: "NO TAKES YET" }));
    return;
  }
  box.replaceChildren(...logItems.map((m, i) => el("button", {
    class: "log-row" + (S.current && S.current.file === m.file ? " is-on" : ""),
    onclick: () => loadOutput(m, true),
  },
  el("span", { class: "log-n", text: pad(logItems.length - i, 3) }),
  el("span", { class: "log-body" }, el("span", { class: "log-t", text: takeTitle(m) }), el("span", { class: "log-m", text: takeMeta(m) })),
  el("span", { class: "log-d", text: m.duration ? fmtTime(m.duration) : "" }))));
}
$("#refresh-log").addEventListener("click", loadLog);

// ══════════════════════════════════════════════════
// Jobs
// ══════════════════════════════════════════════════

function setBusy(on) {
  S.busy = on;
  $("#output").classList.toggle("is-busy", on);
  $("#go").disabled = on;
  $("#dlg-go").disabled = on;
  $("#go").textContent = on ? "RENDERING…" : "RENDER TAKE ▶";
  $("#dlg-go").textContent = on ? "RENDERING…" : "RENDER DIALOGUE ▶";
  $("#rec").textContent = on ? "● REC" : "● REC ARMED";
  $("#rec").classList.toggle("is-live", on);
  $("#progress").hidden = !on;
  $("#progress-label").hidden = !on;
  if (on) {
    $("#wave-empty").hidden = false;
    $("#wave-empty").textContent = "RENDERING";
    $("#out-title").textContent = "TAKE_IN_PROGRESS";
    $("#out-meta").textContent = "";
    peaks = null;
    $("#playhead").hidden = true;
    drawWave();
  } else {
    $("#wave-empty").textContent = "AWAITING TAKE";
  }
}

function showProgress(job) {
  const bar = $("#progress");
  const label = $("#progress-label");
  const p = job.progress;
  if (job.status === "queued") {
    bar.classList.add("is-indeterminate");
    label.textContent = `QUEUED · POSITION ${job.position}`;
  } else if (p && p.total) {
    bar.classList.remove("is-indeterminate");
    $("#progress-fill").style.width = (p.current / p.total) * 100 + "%";
    label.textContent = `[${p.current}/${p.total}] ${p.label}`;
    document.querySelectorAll(".line").forEach((li, i) => {
      li.classList.toggle("is-active", i === p.current - 1);
      li.classList.toggle("is-done", i < p.current - 1);
    });
  } else {
    bar.classList.add("is-indeterminate");
    label.textContent = ((p && p.label) || "Rendering").toUpperCase() + "…";
  }
}

async function track(job) {
  setBusy(true);
  showProgress(job);
  pollStatus();
  try {
    while (job.status === "queued" || job.status === "running") {
      await new Promise((r) => setTimeout(r, 800));
      job = await api("/api/jobs/" + job.id);
      showProgress(job);
    }
  } catch (e) {
    job = { status: "error", error: e.message };
  }
  setBusy(false);
  document.querySelectorAll(".line").forEach((li) => li.classList.remove("is-active", "is-done"));
  pollStatus();
  if (job.status === "error") {
    toast(job.error || "Generation failed.");
    if (S.current) loadOutput(S.current);
    return;
  }
  await loadLog();
  loadOutput(job.result, true);
}

async function generate() {
  if (S.busy) return;
  try {
    const job = await api("/api/generate", { method: "POST", json: { prompt: prompt.value, voice: S.selectedRef, ...S.params } });
    track(job);
  } catch (e) {
    toast(e.message);
  }
}
$("#go").addEventListener("click", generate);

// ══════════════════════════════════════════════════
// Dialogue
// ══════════════════════════════════════════════════

const script = $("#script");
script.addEventListener("input", () => store.set("script", script.value));

function voiceOptions(selected) {
  const lib = S.voices.filter((v) => !v.builtin);
  const builtin = S.voices.filter((v) => v.builtin);
  const opt = (v) => el("option", { value: v.ref, text: v.name, selected: v.ref === selected });
  return [
    el("option", { value: "", text: "Default voice", selected: !selected }),
    lib.length ? el("optgroup", { label: "Library" }, lib.map(opt)) : null,
    builtin.length ? el("optgroup", { label: "Built-in" }, builtin.map(opt)) : null,
  ].filter(Boolean);
}

function renderCast() {
  const { characters, cast } = S.dlg;
  $("#cast-panel").hidden = !characters.length;
  $("#cast").replaceChildren(...characters.map((c) => {
    const sel = el("select", { class: "field select mono", "aria-label": `Voice for ${c}`, onchange: (e) => { cast[c] = e.target.value; } }, voiceOptions(cast[c] || ""));
    return el("div", { class: "cast-row" },
      el("span", { class: "cast-name", text: c }),
      sel,
      el("button", { class: "voice-play", "aria-label": `Preview voice for ${c}`, onclick: (e) => sel.value && togglePreview(sel.value, e.currentTarget) }, PLAY_SVG()));
  }));
}

function renderLines() {
  const emotions = S.status ? S.status.emotions : ["Normal"];
  $("#lines-panel").hidden = !S.dlg.lines.length;
  $("#lines").replaceChildren(...S.dlg.lines.map((l, i) => el("li", { class: "line" },
    el("span", { class: "line-n", text: pad(i + 1) }),
    el("span", { class: "line-char", text: l.character, title: l.character }),
    el("input", { class: "field mono line-text", value: l.line, "aria-label": `Line ${i + 1}`, oninput: (e) => { l.line = e.target.value; } }),
    el("select", { class: "field select mono", "aria-label": `Emotion for line ${i + 1}`, onchange: (e) => { l.emotion = e.target.value; } },
      emotions.map((em) => el("option", { value: em, text: em, selected: em === l.emotion }))))));
  $("#emotion-all").replaceChildren(el("option", { value: "", text: "—" }), ...emotions.map((em) => el("option", { value: em, text: em })));
}

function renderDialogue() {
  renderCast();
  renderLines();
  const n = S.dlg.lines.length;
  $("#parse-status").textContent = n ? `FOUND ${n} LINES · ${S.dlg.characters.length} CHARACTERS` : "PASTE A SCRIPT, THEN PARSE";
}

$("#parse").addEventListener("click", async () => {
  try {
    const r = await api("/api/dialogue/parse", { method: "POST", json: { script: script.value } });
    // Give each character a voice in turn: library first, then built-in
    const pool = [...S.voices.filter((v) => !v.builtin), ...S.voices.filter((v) => v.builtin)].map((v) => v.ref);
    const cast = {};
    r.characters.forEach((c, i) => { cast[c] = S.dlg.cast[c] ?? (pool.length ? pool[i % pool.length] : ""); });
    S.dlg = { lines: r.lines.map((l) => ({ ...l, emotion: "Normal" })), characters: r.characters, cast };
    renderDialogue();
  } catch (e) {
    toast(e.message);
  }
});

$("#emotion-all").addEventListener("change", (e) => {
  if (!e.target.value) return;
  S.dlg.lines.forEach((l) => { l.emotion = e.target.value; });
  renderLines();
});

$("#dlg-speed").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  $("#dlg-speed-v").textContent = v.toFixed(2) + "×";
  e.target.style.setProperty("--p", ((v - 0.5) / 1.5) * 100 + "%");
});

$("#dlg-go").addEventListener("click", async () => {
  if (S.busy) return;
  try {
    const job = await api("/api/dialogue/generate", {
      method: "POST",
      json: { lines: S.dlg.lines, voices: S.dlg.cast, speed: parseFloat($("#dlg-speed").value) },
    });
    track(job);
  } catch (e) {
    toast(e.message);
  }
});

// ══════════════════════════════════════════════════
// Boot
// ══════════════════════════════════════════════════

(async function boot() {
  setFx(store.get("fx", true));
  setView(store.get("view", "generate"));
  prompt.value = store.get("prompt", "");
  renderPrompt();
  script.value = store.get("script", "");
  $("#dlg-speed").dispatchEvent(new Event("input"));
  drawWave();

  await pollStatus();
  initParams(S.status ? S.status.defaults : { cfg: 2.5, stg: 1.5, seed: 42 });
  if (S.status) renderOptions();
  await loadVoices();
  if (S.selectedRef.startsWith("upload:") || (S.selectedRef && !S.voices.some((v) => v.ref === S.selectedRef))) S.selectedRef = "";
  selectVoice(S.selectedRef);
  loadPresets();
  await loadLog();
  setInterval(pollStatus, 4000);
})();
