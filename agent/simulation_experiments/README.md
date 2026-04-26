# Simulation Experiments

This folder tracks the "simulate SPAD measurements from RGB input" line of work.

**End goal:** Given an online sRGB image (from a third-party dataset), produce a
plausible simulated SPAD measurement — either a per-channel set of binary
captures or a single monochrome binary stream — to enable large-scale dataset
augmentation beyond what we can physically capture with our own SPAD rig.

## Index

| File | What it covers |
|------|----------------|
| `EXPERIMENTS_LOG.md` | Chronological log of simulation experiments with code, results, and open issues |
| `CALIBRATION_DESIGN_NOTES.md` | Deeper design discussion: what space to calibrate in, preprocessing choices |
| `chat_exports/2026-04_calibration_discussion.md` | Conversation-style log of the calibration design discussion |

## Related paths

- Calibration script + outputs: `/nfs/horai.dgpsrv/ondemand30/jw954/calibration/`
- Raw SPAD binaries: `/nfs/horai.dgpsrv/ondemand30/jw954/images/`
- Processed dataset: `/nfs/horai.dgpsrv/ondemand30/jw954/spad_dataset/`
- Non-parametric sim scripts in `spad-diffusion/spad_dataset/` are **ignored**
  (agreed with user: ignore those, build a parametric pipeline from scratch)
