# PR 描述模板

> **语言要求：以下所有内容必须使用英文填写，不要使用中文或其他语言。**

```markdown
## {Model Name} Integration

### Summary

Integrate [{Model Name}](官方链接) into DiffSynth-Studio, supporting:
- ✅ {Feature 1}
- ✅ {Feature 2}
- ✅ {Feature 3}

### Model Components

| Component | File | Params |
|-----------|------|--------|
| DiT | `diffsynth/models/{model}_dit.py` | {X}B |
| VAE | `diffsynth/models/{model}_vae.py` | {X}M |
| Text Encoder | `diffsynth/models/{model}_text_encoder.py` | {X}B |

### Dependencies Added

- `some_package>=x.y.z`
- ...

### Notes

- {Any limitations, hardware requirements, or known issues}
```
