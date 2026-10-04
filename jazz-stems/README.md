# Jazz Stems

In-browser 6-stem music separation for jazz practice. Drop a track, get
vocals / drums / bass / guitar / piano / other stems, mute the guitar or
piano and comp over the rest. No build step, plain HTML/CSS/JS.

Live at https://itselisd.github.io/jazz-stems/

## Files

- `index.html` - UI and styles
- `app.js` - model loading, separation pipeline, mixer, transport, downloads
- `lib.js` - pure functions (windowing, WAV encoder, ZIP writer, chunked
  overlap-add inference). Unit tested under node, see below.

## Model

`htdemucs_6s` (Demucs by Meta AI Research) exported to ONNX by StemSplit,
fp16-weights variant, loaded at runtime from Hugging Face and cached in the
browser via the Cache API:

https://huggingface.co/StemSplitio/htdemucs-6s-onnx

- File: `htdemucs_6s_fp16weights.onnx` (130 MB)
- Input `mix`: float32 `[1, 2, 343980]` (7.8 s at 44.1 kHz stereo)
- Output `stems`: float32 `[1, 6, 2, 343980]`, stem order
  drums, bass, other, vocals, guitar, piano
- Chunking mirrors StemSplit's `infer.py`: 25% overlap, linear fade window,
  overlap-add with weight normalization.

## Inference providers

WebGPU first (fast), falls back to single-threaded WASM with a notice.
GitHub Pages cannot send COOP/COEP headers, so multithreaded WASM is not
available there; WebGPU is the recommended path.

## Running the tests

The pure pipeline logic is tested in node without a GPU:

```
node /tmp/test-lib.mjs   # windowing, WAV, CRC32, ZIP (validated by python zipfile)
node /tmp/test-sep.mjs   # chunked overlap-add vs a mock session, reconstruction to 3e-8
```

What was verified vs not (Oct 2026):

- Verified: model file downloads completely (136,428,532 bytes), parses as
  valid ONNX (checker passes), exposes `mix [1,2,343980]` in and `stems`
  out. Live inference ran on CPU via onnxruntime: one 7.8 s noise chunk
  produced shape (1, 6, 2, 343980) with sane per-stem energy. (Note: session
  creation needed optimizations disabled on the 8 GB test VM; the app tries
  full optimization first and falls back automatically.) The JS overlap-add
  pipeline reconstructs a mock session's input to 3e-8. WAV and ZIP outputs
  validate with Python's wave/zipfile modules. Page markup, module syntax,
  and the jsdelivr onnxruntime-web 1.30.0 URL all resolve.
- Not verified live: in-browser WebGPU inference (no GPU in this
  environment). The pipeline is a close port of StemSplit's own browser demo
  for this exact model, so first real-world run should still be smoke-tested
  in Chrome.

## Publishing

This folder is copied to `jazz-stems/` at the root of the
`itselisd/itselisd.github.io` repo (branch `master`). GitHub Pages serves it
at https://itselisd.github.io/jazz-stems/. No build step.
