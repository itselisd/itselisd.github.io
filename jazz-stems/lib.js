/* Jazz Stems - pure utility functions (no DOM, no ORT). Tested under node. */

// Demucs htdemucs_6s constants: the ONNX graph is hard-bound to these.
export const SAMPLE_RATE = 44100;
export const N_SAMPLES = 343980; // 7.8 s segment
export const OVERLAP = Math.floor(N_SAMPLES / 4);
export const STRIDE = N_SAMPLES - OVERLAP;

// Linear fade window matching StemSplit's Python infer.py _make_window:
// ones in the middle, linear ramps of length `overlap` at both edges
// (np.linspace(0, 1, overlap): endpoints inclusive).
export function makeWindow(segment, overlap) {
  const w = new Float32Array(segment);
  w.fill(1);
  for (let i = 0; i < overlap; i++) {
    const v = i / (overlap - 1);
    w[i] = v;
    w[segment - 1 - i] = v;
  }
  return w;
}

// Chunk plan for overlap-add separation. Returns { nChunks, starts }.
export function planChunks(totalLen, nSamples, stride) {
  const nChunks = Math.max(1, Math.ceil(totalLen / stride));
  const starts = [];
  for (let i = 0; i < nChunks; i++) starts.push(i * stride);
  return { nChunks, starts };
}

// Chunked overlap-add inference, mirrors StemSplit's Python infer.py.
// session: object with async run({mix}) returning { stems: { data } },
//   where data is a Float32Array of shape (1, 6, 2, N_SAMPLES), contiguous.
// mixL/mixR: Float32Array pair at 44100 Hz.
// Returns 6 stems in model order (drums, bass, other, vocals, guitar, piano),
// each [Float32Array L, Float32Array R].
// onProgress(frac, etaSeconds, chunkIndex, chunkCount) is called per chunk.
// makeTensor converts a Float32Array chunk into the tensor object the
// session expects (real ort.Tensor in the browser, plain object in tests).
export async function separateStems(session, mixL, mixR, onProgress, makeTensor) {
  const tf = makeTensor || ((buf) => ({ data: buf, dims: [1, 2, N_SAMPLES] }));
  const totalLen = mixL.length;
  const nChunks = Math.max(1, Math.ceil(totalLen / STRIDE));
  const out = [];
  for (let s = 0; s < 6; s++) out.push([new Float32Array(totalLen), new Float32Array(totalLen)]);
  const weight = new Float32Array(totalLen);
  const win = makeWindow(N_SAMPLES, OVERLAP);
  const chunkBuf = new Float32Array(2 * N_SAMPLES);
  const t0 = Date.now();

  for (let i = 0; i < nChunks; i++) {
    const start = i * STRIDE;
    const end = Math.min(start + N_SAMPLES, totalLen);
    const chunkLen = end - start;
    chunkBuf.fill(0);
    chunkBuf.set(mixL.subarray(start, end), 0);
    chunkBuf.set(mixR.subarray(start, end), N_SAMPLES);
    const result = await session.run({ mix: tf(chunkBuf) });
    // Output "stems" is (1, 6, 2, N_SAMPLES), contiguous.
    const data = result.stems.data;
    for (let s = 0; s < 6; s++) {
      const base = s * 2 * N_SAMPLES;
      for (let c = 0; c < 2; c++) {
        const dst = out[s][c];
        const rowOff = base + c * N_SAMPLES;
        for (let n = 0; n < chunkLen; n++) {
          dst[start + n] += data[rowOff + n] * win[n];
        }
      }
    }
    for (let n = 0; n < chunkLen; n++) weight[start + n] += win[n];

    const elapsed = (Date.now() - t0) / 1000;
    const eta = (elapsed / (i + 1)) * (nChunks - i - 1);
    if (onProgress) onProgress((i + 1) / nChunks, eta, i + 1, nChunks);
    // Yield so the browser repaints the progress bar between chunks.
    // Without this the tab looks frozen during long runs.
    await new Promise((r) => setTimeout(r, 0));
  }

  for (let s = 0; s < 6; s++) {
    for (let c = 0; c < 2; c++) {
      const dst = out[s][c];
      for (let n = 0; n < totalLen; n++) dst[n] /= Math.max(weight[n], 1e-8);
    }
  }
  return out;
}
// Format seconds as m:ss
export function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m + ":" + String(s).padStart(2, "0");
}

// Format an ETA in seconds as a short human string.
export function formatEta(sec) {
  if (!isFinite(sec) || sec < 0) return "";
  if (sec < 5) return "almost done";
  if (sec < 90) return "about " + Math.round(sec) + "s left";
  const m = Math.round(sec / 60);
  return "about " + m + " min left";
}

// 16-bit PCM stereo WAV encoder. stereo = [Float32Array L, Float32Array R].
export function encodeWav(stereo, sampleRate) {
  const n = stereo[0].length;
  const buf = new ArrayBuffer(44 + n * 4);
  const view = new DataView(buf);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + n * 4, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 2, true); // stereo
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true); // byte rate
  view.setUint16(32, 4, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(view, 36, "data");
  view.setUint32(40, n * 4, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 2; c++) {
      const v = Math.max(-1, Math.min(1, stereo[c][i]));
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      off += 2;
    }
  }
  return new Uint8Array(buf);
}

function writeAscii(view, off, s) {
  for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
}

// CRC32 (IEEE) over a Uint8Array.
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(data) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const te = new TextEncoder();

// Minimal ZIP writer, stored (no compression). files = [{name, data: Uint8Array}].
// Returns a Uint8Array with the full archive.
export function makeZip(files) {
  const enc = files.map((f) => ({ name: te.encode(f.name), data: f.data }));
  const parts = [];
  const central = [];
  let offset = 0;

  for (const f of enc) {
    const crc = crc32(f.data);
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true);
    lh.setUint16(4, 20, true); // version needed
    lh.setUint16(6, 0x0800, true); // UTF-8 filename flag
    lh.setUint16(8, 0, true); // stored
    lh.setUint16(10, 0, true); // mod time
    lh.setUint16(12, 0, true); // mod date
    lh.setUint32(14, crc, true);
    lh.setUint32(18, f.data.length, true);
    lh.setUint32(22, f.data.length, true);
    lh.setUint16(26, f.name.length, true);
    lh.setUint16(28, 0, true);
    parts.push(new Uint8Array(lh.buffer), f.name, f.data);

    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 0x031e, true); // made by (unix)
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0x0800, true);
    cd.setUint16(10, 0, true);
    cd.setUint16(12, 0, true);
    cd.setUint16(14, 0, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, f.data.length, true);
    cd.setUint32(24, f.data.length, true);
    cd.setUint16(28, f.name.length, true);
    cd.setUint16(30, 0, true);
    cd.setUint16(32, 0, true);
    cd.setUint16(34, 0, true);
    cd.setUint16(36, 0, true);
    cd.setUint32(38, 0, true);
    cd.setUint32(42, offset, true);
    central.push(new Uint8Array(cd.buffer), f.name);
    offset += 30 + f.name.length + f.data.length;
  }

  const cdStart = offset;
  let cdSize = 0;
  for (const p of central) cdSize += p.length;
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(4, 0, true);
  eocd.setUint16(6, 0, true);
  eocd.setUint16(8, enc.length, true);
  eocd.setUint16(10, enc.length, true);
  eocd.setUint32(12, cdSize, true);
  eocd.setUint32(16, cdStart, true);
  eocd.setUint16(20, 0, true);

  const total = offset + cdSize + 22;
  const out = new Uint8Array(total);
  let p = 0;
  for (const part of parts) { out.set(part, p); p += part.length; }
  for (const part of central) { out.set(part, p); p += part.length; }
  out.set(new Uint8Array(eocd.buffer), p);
  return out;
}
