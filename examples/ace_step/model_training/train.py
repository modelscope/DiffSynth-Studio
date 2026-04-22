import torch, os, argparse, accelerate, warnings, torchaudio
import math
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ToAbsolutePath, RouteByType, DataProcessingOperator, LoadPureAudioWithTorchaudio
from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LoadAceStepAudio(DataProcessingOperator):
    """Load audio file and return waveform tensor [2, T] at 48kHz."""
    def __init__(self, target_sr=48000):
        self.target_sr = target_sr

    def __call__(self, data: str):
        try:
            waveform, sample_rate = torchaudio.load(data)
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            return waveform
        except Exception as e:
            warnings.warn(f"Cannot load audio from {data}: {e}")
            return None


class AceStepTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, silence_latent_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # ===== 解析模型配置（固定写法） =====
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # ===== Tokenizer 配置 =====
        text_tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"))
        silence_latent_config = self.parse_path_or_model_id(silence_latent_path, default_value=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/silence_latent.pt"))
        # ===== 构建 Pipeline =====
        self.pipe = AceStepPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, text_tokenizer_config=text_tokenizer_config, silence_latent_config=silence_latent_config)
        # ===== 拆分 Pipeline Units（固定写法） =====
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # ===== 切换到训练模式（固定写法） =====
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # ===== 其他配置（固定写法） =====
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        # ===== 任务模式路由（固定写法） =====
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"], "positive": True}
        inputs_nega = {"positive": False}
        duration = math.floor(data['audio'][0].shape[1] / data['audio'][1]) if data.get("audio") is not None else data.get("duration", 60)
        # ===== 共享参数 =====
        inputs_shared = {
            # ===== 核心字段映射 =====
            "input_audio": data["audio"],
            # ===== 音频生成任务所需元数据 =====
            "lyrics": data["lyrics"],
            "task_type": "text2music",
            "duration": duration,
            "bpm": data.get("bpm", 100),
            "keyscale": data.get("keyscale", "C major"),
            "timesignature": data.get("timesignature", "4"),
            "vocal_language": data.get("vocal_language", "unknown"),
            # ===== 框架控制参数（固定写法） =====
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        # ===== 额外字段注入：通过 --extra_inputs 配置的数据集列名（固定写法） =====
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # ===== 标准实现，不要修改（固定写法） =====
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def ace_step_parser():
    parser = argparse.ArgumentParser(description="ACE-Step training.")
    parser = add_general_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path in format model_id:origin_pattern.")
    parser.add_argument("--silence_latent_path", type=str, default=None, help="Silence latent path in format model_id:origin_pattern.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = ace_step_parser()
    args = parser.parse_args()
    # ===== Accelerator 配置（固定写法） =====
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # ===== 数据集定义 =====
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=None,
        special_operator_map={
            "audio": ToAbsolutePath(args.dataset_base_path) >> LoadPureAudioWithTorchaudio(target_sample_rate=48000),
        },
    )
    # ===== TrainingModule =====
    model = AceStepTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        silence_latent_path=args.silence_latent_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    # ===== ModelLogger（固定写法） =====
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    # ===== 任务路由（固定写法） =====
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
