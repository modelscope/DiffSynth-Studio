import os, torch
from accelerate import Accelerator


class TensorBoardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard is enabled. Run `tensorboard --logdir={log_dir}` to visualize the training progress.")

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class SwanLabLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import swanlab
        project_name = os.environ.get("SWANLAB_PROJECT", project_name)
        self.swanlab = swanlab
        self.swanlab.init(project=project_name, logdir=log_dir)
        print(f"SwanLab is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.swanlab.log({key: value}, step=step)

    def close(self):
        self.swanlab.finish()


class WandbLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import wandb
        project_name = os.environ.get("WANDB_PROJECT", project_name)
        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, dir=log_dir)
        print(f"Wandb is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.wandb.log({key: value}, step=step)

    def close(self):
        self.wandb.finish()


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
        enable_tensorboard_log=False,
        enable_swanlab_log=False, swanlab_project="DiffSynth-Studio",
        enable_wandb_log=False, wandb_project="DiffSynth-Studio",
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # Loggers
        self.enable_tensorboard_log = enable_tensorboard_log
        self.enable_swanlab_log = enable_swanlab_log
        self.swanlab_project = swanlab_project
        self.enable_wandb_log = enable_wandb_log
        self.wandb_project = wandb_project
        self.loggers = []
        self.loggers_initialized = False

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(TensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(SwanLabLogger(project_name=self.swanlab_project, log_dir=os.path.join(self.output_path, "swanlab_log")))
        if self.enable_wandb_log:
            self.loggers.append(WandbLogger(project_name=self.wandb_project, log_dir=os.path.join(self.output_path, "wandb_log")))
        self.loggers_initialized = True

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if accelerator.is_main_process:
            if not self.loggers_initialized:
                self.init_loggers()
            loss = kwargs.get("loss")
            if loss is not None:
                for logger in self.loggers:
                    logger.log("loss", loss, self.num_steps)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        for logger in self.loggers:
            logger.close()

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
