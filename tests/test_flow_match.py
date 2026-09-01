import importlib.util
from pathlib import Path

import torch

module_path = Path(__file__).parents[1] / "diffsynth" / "diffusion" / "flow_match.py"
module_spec = importlib.util.spec_from_file_location("flow_match", module_path)
flow_match = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(flow_match)


def test_z_image_target_timesteps_keep_matching_sigmas():
    student = flow_match.FlowMatchScheduler("Z-Image")
    student.set_timesteps(8)

    teacher = flow_match.FlowMatchScheduler("Z-Image")
    teacher.set_timesteps(50, target_timesteps=student.timesteps)

    for target_timestep in student.timesteps:
        timestep_id = torch.argmin((teacher.timesteps - target_timestep).abs())
        torch.testing.assert_close(teacher.timesteps[timestep_id], target_timestep)

    torch.testing.assert_close(
        teacher.timesteps,
        teacher.sigmas * teacher.num_train_timesteps,
    )
