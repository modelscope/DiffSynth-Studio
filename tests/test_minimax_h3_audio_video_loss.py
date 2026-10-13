from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.loss import FlowMatchSFTMiniMaxH3AudioVideoLoss


def test_missing_audio_input_latents_raises_before_model_fn():
    pipe = SimpleNamespace(
        scheduler=FlowMatchScheduler("MiniMax-H3"),
        scheduler_audio=FlowMatchScheduler("MiniMax-H3"),
        device="cpu",
        in_iteration_models=(),
        model_fn=Mock(),
    )
    for scheduler in (pipe.scheduler, pipe.scheduler_audio):
        scheduler.set_timesteps(8, training=True)

    with pytest.raises(ValueError, match="audio_latents.*audio_input_latents.*sft:data_process"):
        FlowMatchSFTMiniMaxH3AudioVideoLoss(
            pipe, input_latents=torch.zeros(1, 24, 2, 2, 2),
        )
    pipe.model_fn.assert_not_called()
