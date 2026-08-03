# Training Model Configurations

This directory is the runtime source of truth for models exposed by the training UI.

Directory layout:

```text
model_configs/<examples-family>/<training-model>/default.json
```

For example:

```text
model_configs/flux2/FLUX.2-klein-base-4B/default.json
```

Each config has four parameter layers:

- `defaults` explicitly contains every model-dependent value editable in the job form;
- `training_args` contains fixed arguments shared by all training stages;
- `stages` contains only per-stage overrides and is usually `[{}]` for a single-stage model;
- `sampling` is the final top-level field and contains `validate_script`, the explicit
  `.jpg`/`.mp4`/`.wav` `output_extension`, and the default `sample_prompts`.

Keep `sampling.output_extension` consistent with the media file written by the
corresponding validation script. The sampling worker uses this field directly when
constructing `final_samples/sample_NNN<extension>`.

The editable training-option defaults are `gradient_accumulation`,
`dataset_num_workers`, `find_unused_parameters`, and `extra_inputs`. The UI does
not expose or store `trigger_word`, `seed`, or `save_every`. Keep
`dataset_num_workers` because several supported example scripts explicitly set it
to `8`; models that omit it use the parser default `0`.

`num_frames` is present only for video models. Image and audio configs omit it;
the runtime also ignores legacy `num_frames` values for non-video jobs.

`family` and `name` are explicit and must match the directory path. Configs do not
use a `schema_version` field. Dataset paths and `output_path` are task-specific and
must not be stored in a model default config.

Defaults explicitly present in an example shell use the shell value. Missing form
values use the defaults declared by the corresponding `train.py` parser in
`diffsynth/diffusion/parsers.py`. Runtime code reads these values directly instead
of deriving form defaults from the first stage.

Runtime code reads these JSON files directly and does not parse shell scripts. After
changing an example LoRA shell script or a model config, run:

```bash
python training_ui/scripts/validate_model_configs.py
```

The command fails if an explicit form default, fixed argument, or stage override
differs from its source shell script and parser defaults.
