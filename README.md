# OpScale – The Open ONNX Video Upscaler

Ein minimalistischer, robuster Offline-Video-Upscaler für Windows mit AMD-GPUs.

## Ablaufdiagramm

```text
Video (input/)
  │
  ▼
FFmpeg (Single-Pass Analyse & Frame-Extraktion)
  │
  ▼
PNG Frames (temp/)
  │
  ▼
ONNX Runtime (DirectML Inferenz & In-Place Tiling)
  │
  ▼
Hochskalierte PNG Frames (temp/)
  │
  ▼
FFmpeg (Muxing: Codec-Übernahme + Audio / Untertitel Copy)
  │
  ▼
Ausgabevideo (output/)

## Models

OpScale does not include ONNX models.

Download the supported models and place them into the `models/` directory before running the application.