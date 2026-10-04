/* Jazz Stems: in-browser 6-stem separation with Demucs htdemucs_6s.
 * No build step. onnxruntime-web loads from CDN, model weights load from
 * Hugging Face once and are cached in the browser via the Cache API.
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";
import {
  SAMPLE_RATE, N_SAMPLES, STRIDE,
  formatTime, formatEta, encodeWav, makeZip, separateStems,
} from "./lib.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_URL =
  "https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx";
const CACHE_NAME = "jazz-stems-v1";
const MAX_DURATION_S = 480; // 8 minute cap, keeps memory in check

// Stem order in the model output (matches SOURCES in StemSplit's infer.py)
const MODEL_ORDER = ["drums", "bass", "other", "vocals", "guitar", "piano"];
// Display order in the mixer: Elis mutes guitar or piano to practice comping
const STEMS = ["guitar", "piano", "bass", "drums", "vocals", "other"];

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const modelBtn = $("model-btn");
const modelStatus = $("model-status");
const providerPill = $("provider-pill");
const modelProgressWrap = $("model-progress-wrap");
const modelBar = $("model-bar");
const modelProgressLabel = $("model-progress-label");
const modelProgressPct = $("model-progress-pct");
const dropzone = $("dropzone");
const fileInput = $("file-input");
const fileStatus = $("file-status");
const progressCard = $("progress-card");
const sepBar = $("sep-bar");
const sepLabel = $("sep-label");
const sepEta = $("sep-eta");
const studioCard = $("studio-card");
const playBtn = $("play-btn");
const seek = $("seek");
const timeLabel = $("time-label");
const masterVol = $("master-vol");
const dlAllBtn = $("dl-all");
const resetBtn = $("reset-btn");
const strips = [...document.querySelectorAll(".strip")];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let session = null;
let providerUsed = "";
let modelBytes = null; // Uint8Array, kept after first load
let stems = null; // display order: [{l: Float32Array, r: Float32Array} x6]
let stemBuffers = null; // AudioBuffers in display order
let trackName = "track";
let durationS = 0;
let busy = false;

const mixState = STEMS.map(() => ({ muted: false, solo: false, vol: 1 }));

// Playback engine state
let actx = null;
let masterGain = null;
let stemGains = [];
let sources = [];
let playing = false;
let playOffset = 0; // seconds, where the current play segment started
let playCtxTime = 0; // actx.currentTime when the segment started
let rafId = 0;
let scrubbing = false;

// ---------------------------------------------------------------------------
// Provider detection
// ---------------------------------------------------------------------------

if (navigator.gpu) {
  providerPill.textContent = "WebGPU available";
  providerPill.classList.add("ok");
} else {
  providerPill.textContent = "CPU only, WASM fallback";
  providerPill.classList.add("warn");
}

// ---------------------------------------------------------------------------
// Model download + cache
// ---------------------------------------------------------------------------

async function ensureModelBytes(onProgress) {
  if (modelBytes) return modelBytes;
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(MODEL_URL);
  if (hit) {
    modelStatus.innerHTML = "Model ready, <strong>loaded from browser cache</strong>.";
    modelBtn.disabled = true;
    modelBtn.textContent = "Model cached";
    modelBytes = new Uint8Array(await hit.arrayBuffer());
    return modelBytes;
  }
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error("Model download failed: HTTP " + res.status);
  const total = parseInt(res.headers.get("Content-Length") || "0", 10);
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const flat = new Uint8Array(received);
  let p = 0;
  for (const c of chunks) { flat.set(c, p); p += c.length; }
  await cache.put(
    MODEL_URL,
    new Response(flat, { headers: { "Content-Type": "application/octet-stream" } })
  );
  modelStatus.innerHTML = "Model ready, <strong>cached for next time</strong>.";
  modelBtn.disabled = true;
  modelBtn.textContent = "Model cached";
  modelBytes = flat;
  return modelBytes;
}

function showModelProgress(received, total) {
  modelProgressWrap.hidden = false;
  const mb = (received / 1048576).toFixed(0) + " MB";
  if (total > 0) {
    const pct = (100 * received / total).toFixed(0);
    modelBar.style.width = pct + "%";
    modelProgressPct.textContent = pct + "%";
    modelProgressLabel.textContent = "Downloading model, " + mb + " of " + (total / 1048576).toFixed(0) + " MB";
  } else {
    modelProgressLabel.textContent = "Downloading model, " + mb + " so far";
  }
}

modelBtn.addEventListener("click", async () => {
  modelBtn.disabled = true;
  try {
    await ensureModelBytes(showModelProgress);
  } catch (e) {
    console.error(e);
    modelStatus.textContent = "Download failed: " + e.message + ". Try again.";
    modelBtn.disabled = false;
  } finally {
    modelProgressWrap.hidden = true;
  }
});

// Pre-check the cache on load so repeat visits show "cached" immediately.
(async () => {
  try {
    const cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(MODEL_URL);
    if (hit) {
      modelBytes = new Uint8Array(await hit.arrayBuffer());
      modelStatus.innerHTML = "Model ready, <strong>cached from a previous visit</strong>.";
      modelBtn.disabled = true;
      modelBtn.textContent = "Model cached";
    }
  } catch (e) { console.warn("Cache check failed", e); }
})();

// ---------------------------------------------------------------------------
// ORT session with WebGPU first, WASM fallback
// ---------------------------------------------------------------------------

async function createSession(bytes, providers, optLevel) {
  return await ort.InferenceSession.create(bytes, {
    graphOptimizationLevel: optLevel,
    executionProviders: providers,
  });
}

async function ensureSession() {
  if (session) return session;
  const bytes = await ensureModelBytes(showModelProgress);
  modelProgressWrap.hidden = true;

  // Session creation compiles GPU shaders on first run and can take a
  // minute or two with no progress events. Say so explicitly, otherwise
  // this phase looks exactly like a frozen page.
  fileStatus.textContent =
    "Starting the model (first run compiles GPU shaders, can take a minute or two) ...";

  // Try WebGPU first (fast). Fall back to WASM, and if the optimizing
  // session build runs out of memory, retry with optimizations off.
  // Chain: webgpu/all -> wasm/all -> wasm/disabled.
  const attempts = [
    { providers: ["webgpu"], opt: "all", needsGpu: true },
    { providers: ["wasm"], opt: "all" },
    { providers: ["wasm"], opt: "disabled" },
  ];
  let lastErr = null;
  for (const a of attempts) {
    if (a.needsGpu && !navigator.gpu) continue;
    try {
      if (a.providers[0] === "wasm") {
        // No cross-origin isolation on static hosting, so single-threaded WASM.
        ort.env.wasm.numThreads = 1;
      }
      session = await createSession(bytes, a.providers, a.opt);
      providerUsed = a.providers[0];
      break;
    } catch (e) {
      console.warn("Session creation failed (" + a.providers[0] + "/" + a.opt + "):", e);
      lastErr = e;
    }
  }
  if (!session) throw new Error("Could not start the model: " + (lastErr && lastErr.message));

  if (providerUsed === "webgpu") {
    providerPill.textContent = "Running on WebGPU";
    providerPill.classList.add("ok");
  } else {
    providerPill.textContent = "Running on CPU (WASM)";
    fileStatus.textContent = "WebGPU unavailable, using CPU (WASM). Separation will be slower.";
  }
  return session;
}

// ---------------------------------------------------------------------------
// File intake
// ---------------------------------------------------------------------------

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
for (const ev of ["dragenter", "dragover"]) {
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); });
}
for (const ev of ["dragleave", "drop"]) {
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); });
}
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  if (f) handleFile(f);
  fileInput.value = "";
});

async function decodeTo44k(file) {
  const buf = await file.arrayBuffer();
  const dctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  let audio;
  try {
    audio = await dctx.decodeAudioData(buf);
  } finally {
    dctx.close().catch(() => {});
  }
  if (audio.duration > MAX_DURATION_S) {
    throw new Error(
      "Track is " + formatTime(audio.duration) + ", over the 8 minute cap. " +
      "Trim it down and try again."
    );
  }
  const l = audio.getChannelData(0);
  const r = audio.numberOfChannels > 1 ? audio.getChannelData(1) : l;
  return { l: Float32Array.from(l), r: Float32Array.from(r), duration: audio.duration };
}

async function handleFile(file) {
  if (busy) return;
  busy = true;
  resetStudio();
  try {
    fileStatus.textContent = "Decoding " + file.name + " ...";
    const { l, r, duration } = await decodeTo44k(file);
    trackName = file.name.replace(/\.[^.]+$/, "").replace(/[^\w\- ]+/g, "").trim() || "track";
    durationS = duration;
    fileStatus.innerHTML = "Decoded <strong>" + file.name + "</strong>, " +
      formatTime(duration) + ". Loading model ...";

    await ensureSession();
    progressCard.hidden = false;
    fileStatus.textContent = "Separating into 6 stems ...";

    const separated = await separateStems(session, l, r, (frac, etaS, i, n) => {
      sepBar.style.width = (frac * 100).toFixed(1) + "%";
      sepLabel.textContent = "Chunk " + i + " of " + n + " (" + providerUsed.toUpperCase() + ")";
      sepEta.textContent = formatEta(etaS);
    }, (buf) => new ort.Tensor("float32", buf, [1, 2, N_SAMPLES]));

    // Reorder model output (drums, bass, other, vocals, guitar, piano)
    // into mixer display order (guitar, piano, bass, drums, vocals, other).
    stems = STEMS.map((name) => separated[MODEL_ORDER.indexOf(name)]);
    buildPlaybackGraph();
    progressCard.hidden = true;
    studioCard.hidden = false;
    fileStatus.innerHTML = "Done. <strong>" + file.name + "</strong> is ready to mix.";
    studioCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    console.error(e);
    fileStatus.textContent = "Error: " + e.message;
    progressCard.hidden = true;
  } finally {
    busy = false;
  }
}

// ---------------------------------------------------------------------------
// Playback engine: one source per stem, started together, per-stem gains
// ---------------------------------------------------------------------------

function buildPlaybackGraph() {
  teardownAudio();
  actx = new (window.AudioContext || window.webkitAudioContext)();
  masterGain = actx.createGain();
  masterGain.gain.value = masterVol.value / 100;
  masterGain.connect(actx.destination);
  stemBuffers = stems.map(({ l, r }) => {
    const buf = actx.createBuffer(2, l.length, SAMPLE_RATE);
    buf.copyToChannel(l, 0);
    buf.copyToChannel(r, 1);
    return buf;
  });
  stemGains = stems.map(() => {
    const g = actx.createGain();
    g.connect(masterGain);
    return g;
  });
  mixState.forEach((s) => { s.muted = false; s.solo = false; s.vol = 1; });
  strips.forEach((strip) => {
    strip.querySelector(".mute-btn").classList.remove("active");
    strip.querySelector(".solo-btn").classList.remove("active");
    strip.querySelector(".vol").value = 100;
  });
  applyMixer();
  playOffset = 0;
  updateTransportUI();
}

function applyMixer() {
  if (!actx) return;
  const anySolo = mixState.some((s) => s.solo);
  const t = actx.currentTime;
  mixState.forEach((s, i) => {
    let v = s.vol;
    if (s.muted) v = 0;
    if (anySolo && !s.solo) v = 0;
    stemGains[i].gain.setTargetAtTime(v, t, 0.015);
  });
}

function currentPos() {
  if (!actx) return playOffset;
  return playing
    ? Math.min(playOffset + (actx.currentTime - playCtxTime), durationS)
    : playOffset;
}

function startSources(offset) {
  stopSources();
  offset = Math.max(0, Math.min(offset, Math.max(0, durationS - 0.05)));
  sources = stemBuffers.map((buf, i) => {
    const src = actx.createBufferSource();
    src.buffer = buf;
    src.connect(stemGains[i]);
    src.start(0, offset);
    return src;
  });
  playOffset = offset;
  playCtxTime = actx.currentTime;
  playing = true;
  updateTransportUI();
  tick();
}

function stopSources() {
  for (const s of sources) { try { s.stop(); } catch (e) { /* already stopped */ } }
  sources = [];
  playing = false;
}

function play() {
  if (!actx || !stemBuffers) return;
  actx.resume();
  let off = currentPos();
  if (off >= durationS - 0.1) off = 0;
  startSources(off);
}

function pause() {
  if (!playing) return;
  playOffset = currentPos();
  stopSources();
  updateTransportUI();
}

function togglePlay() {
  if (playing) pause();
  else play();
}

function tick() {
  cancelAnimationFrame(rafId);
  if (!playing) return;
  const pos = currentPos();
  if (!scrubbing) seek.value = pos;
  timeLabel.textContent = formatTime(pos) + " / " + formatTime(durationS);
  if (pos >= durationS - 0.05) {
    pause();
    playOffset = 0;
    updateTransportUI();
    return;
  }
  rafId = requestAnimationFrame(tick);
}

function updateTransportUI() {
  playBtn.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  seek.max = durationS;
  if (!scrubbing) seek.value = currentPos();
  timeLabel.textContent = formatTime(currentPos()) + " / " + formatTime(durationS);
}

playBtn.addEventListener("click", togglePlay);

seek.addEventListener("pointerdown", () => { scrubbing = true; });
seek.addEventListener("pointerup", () => {
  scrubbing = false;
  const v = parseFloat(seek.value);
  if (playing) startSources(v);
  else { playOffset = v; updateTransportUI(); }
});
seek.addEventListener("input", () => {
  if (scrubbing) timeLabel.textContent = formatTime(parseFloat(seek.value)) + " / " + formatTime(durationS);
});
seek.addEventListener("change", () => {
  const v = parseFloat(seek.value);
  if (playing) startSources(v);
  else { playOffset = v; updateTransportUI(); }
});

masterVol.addEventListener("input", () => {
  if (masterGain) masterGain.gain.setTargetAtTime(masterVol.value / 100, actx.currentTime, 0.015);
});

// ---------------------------------------------------------------------------
// Mixer wiring
// ---------------------------------------------------------------------------

strips.forEach((strip, i) => {
  const muteBtn = strip.querySelector(".mute-btn");
  const soloBtn = strip.querySelector(".solo-btn");
  const vol = strip.querySelector(".vol");
  muteBtn.addEventListener("click", () => {
    mixState[i].muted = !mixState[i].muted;
    muteBtn.classList.toggle("active", mixState[i].muted);
    applyMixer();
  });
  soloBtn.addEventListener("click", () => {
    mixState[i].solo = !mixState[i].solo;
    soloBtn.classList.toggle("active", mixState[i].solo);
    applyMixer();
  });
  vol.addEventListener("input", () => {
    mixState[i].vol = vol.value / 100;
    applyMixer();
  });
  strip.querySelector(".dl-btn").addEventListener("click", () => downloadStem(i));
});

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

function saveBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
}

function downloadStem(i) {
  if (!stems) return;
  const wav = encodeWav([stems[i].l, stems[i].r], SAMPLE_RATE);
  saveBlob(new Blob([wav], { type: "audio/wav" }), trackName + "_" + STEMS[i] + ".wav");
}

dlAllBtn.addEventListener("click", () => {
  if (!stems) return;
  dlAllBtn.disabled = true;
  dlAllBtn.textContent = "Zipping ...";
  setTimeout(() => {
    try {
      const files = stems.map(({ l, r }, i) => ({
        name: trackName + "_" + STEMS[i] + ".wav",
        data: encodeWav([l, r], SAMPLE_RATE),
      }));
      const zip = makeZip(files);
      saveBlob(new Blob([zip], { type: "application/zip" }), trackName + "_stems.zip");
    } finally {
      dlAllBtn.disabled = false;
      dlAllBtn.textContent = "Download all stems (.zip)";
    }
  }, 30);
});

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

function teardownAudio() {
  cancelAnimationFrame(rafId);
  stopSources();
  if (actx) { actx.close().catch(() => {}); actx = null; }
  stemGains = [];
  stemBuffers = null;
  playOffset = 0;
  playing = false;
}

function resetStudio() {
  teardownAudio();
  stems = null;
  studioCard.hidden = true;
  progressCard.hidden = true;
  sepBar.style.width = "0%";
}

resetBtn.addEventListener("click", () => {
  resetStudio();
  fileStatus.textContent = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});
