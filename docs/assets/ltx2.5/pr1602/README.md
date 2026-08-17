# LTX-2.5 inference sample for PR #1602

This directory contains the current implementation's P0 Distilled text-to-audio-video sample for [PR #1602](https://github.com/modelscope/DiffSynth-Studio/pull/1602).

- `p0-distilled-t2av-960x576-121f.mp4`: 960×576, 121 frames at 24 FPS; H.264 video with 48 kHz stereo AAC audio.
- `p0-distilled-t2av-960x576-121f-preview.gif`: 480×288 animated preview without audio for inline PR display.
- `SHA256SUMS`: checksums for the media files.

The sample was generated from commit `e587d1e33c748ca8044d05a196ef2b6459bb7d2d` with the documented low-VRAM P0 Distilled pipeline. It is inference evidence, not a model weight or benchmark artifact.
