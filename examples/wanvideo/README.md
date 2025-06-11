

* dataset
  * `--dataset_base_path`: Base path of the Dataset.
  * `--dataset_metadata_path`: Metadata path of the Dataset.
  * `--height`: Image or video height. Leave `height` and `width` None to enable dynamic resolution.
  * `--width`: Image or video width. Leave `height` and `width` None to enable dynamic resolution.
  * `--num_frames`: Number of frames in each video. The frames are sampled from the prefix.
  * `--data_file_keys`: Data file keys in metadata. Separated by commas.
  * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
* Model
  * `--model_paths`: Model paths to be loaded. JSON format.
  * `--model_id_with_origin_paths`: Model ID with original path, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Separated by commas.
* Training
  * `--learning_rate`: Learning rate.
  * `--num_epochs`: Number of epochs.
  * `--output_path`: Save path.
  * `--remove_prefix_in_ckpt`: Remove prefix in ckpt.
* Trainable module
  * `--trainable_models`: Trainable models, e.g., dit, vae, text_encoder.
  * `--lora_base_model`: Add LoRA on which model.
  * `--lora_target_modules`: Add LoRA on which layer.
  * `--lora_rank`: LoRA rank.
* Extra model input
  * `--input_contains_input_image`: Model input contains `input_image`
  * `--input_contains_end_image`: Model input contains `end_image`.
  * `--input_contains_control_video`: Model input contains `control_video`.
  * `--input_contains_reference_image`: Model input contains `reference_image`.
  * `--input_contains_vace_video`: Model input contains `vace_video`.
  * `--input_contains_vace_reference_image`: Model input contains `vace_reference_image`.
  * `--input_contains_motion_bucket_id`: Model input contains `motion_bucket_id`.

