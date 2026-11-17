import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.loss import FlowMatchSFTAudioVideoLoss, FlowMatchSFTLoss


class TinyPipeline:
    def __init__(self, dtype, audio_video=False):
        self.device = torch.device("cpu")
        self.torch_dtype = dtype
        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(1000, training=True)
        self.in_iteration_models = ()
        self.prediction = torch.nn.Parameter(torch.tensor(0.25, dtype=dtype))
        self.audio_video = audio_video
        self.model_inputs = None

    def model_fn(self, **inputs):
        self.model_inputs = inputs
        key = "video_latents" if self.audio_video else "latents"
        video = self.prediction.expand_as(inputs[key])
        if self.audio_video:
            audio = None
            if inputs.get("audio_latents") is not None:
                audio = self.prediction.expand_as(inputs["audio_latents"])
            return video, audio
        if "first_frame_latents" in inputs:
            # A large error on the conditioning frame must not enter the loss.
            video = torch.cat((torch.full_like(video[:, :, :1], 1000), video[:, :, 1:]), dim=2)
        return video


class FlowMatchSFTLossTest(unittest.TestCase):
    def run_loss(self, pipe, timestep_id, inputs):
        loss_fn = FlowMatchSFTAudioVideoLoss if pipe.audio_video else FlowMatchSFTLoss
        # Exercise both sides of the Wan high/low expert split.
        lower, upper = (0, 0.358) if timestep_id < 358 else (0.358, 1)
        with patch("torch.randint", return_value=torch.tensor([timestep_id])) as sample:
            with patch("torch.randn_like", side_effect=torch.zeros_like):
                loss = loss_fn(pipe, **inputs, min_timestep_boundary=lower, max_timestep_boundary=upper)
        sample.assert_called_once_with(int(lower * 1000), int(upper * 1000), (1,))
        model_timestep = pipe.model_inputs["timestep"]
        self.assertEqual(model_timestep.dtype, pipe.torch_dtype)
        self.assertEqual(model_timestep.device, pipe.device)
        torch.testing.assert_close(
            model_timestep,
            pipe.scheduler.timesteps[[timestep_id]].to(dtype=pipe.torch_dtype),
            rtol=0, atol=0,
        )
        return loss

    def expected_loss(self, pipe, timestep_id, clean):
        # Zero noise makes the flow-matching target -clean.
        prediction = pipe.prediction.detach().expand_as(clean)
        return F.mse_loss(prediction.float(), -clean.float()) * pipe.scheduler.linear_timesteps_weights[timestep_id]

    def assert_gradient(self, pipe, timestep_id, loss):
        loss.backward()
        self.assertTrue(torch.isfinite(pipe.prediction.grad).all())
        if timestep_id == 0:
            # The scheduler intentionally gives the true endpoint zero weight.
            self.assertEqual(loss.item(), 0)
            self.assertEqual(pipe.prediction.grad.item(), 0)
        else:
            self.assertGreater(loss.item(), 0)
            self.assertGreater(pipe.prediction.grad.item(), 0)

    def test_sft_preserves_sampled_schedule_entry(self):
        # IDs 1..9 round to timestep 1000 in BF16, as does the real endpoint 0.
        ids = list(range(11)) + [357, 358, 500, 999]
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for timestep_id in ids:
                with self.subTest(dtype=dtype, timestep_id=timestep_id):
                    pipe = TinyPipeline(dtype)
                    clean = torch.arange(1, 13, dtype=dtype).reshape(1, 1, 3, 2, 2)
                    loss = self.run_loss(pipe, timestep_id, {"input_latents": clean})
                    sigma = pipe.scheduler.sigmas[timestep_id]
                    torch.testing.assert_close(pipe.model_inputs["latents"], (1 - sigma) * clean, rtol=0, atol=0)
                    torch.testing.assert_close(loss, self.expected_loss(pipe, timestep_id, clean))
                    self.assert_gradient(pipe, timestep_id, loss)

    def test_audio_video_preserves_sampled_schedule_entry(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for timestep_id in (0, 1, 9, 10, 357, 358, 999):
                for with_audio in (False, True):
                    with self.subTest(dtype=dtype, timestep_id=timestep_id, with_audio=with_audio):
                        pipe = TinyPipeline(dtype, audio_video=True)
                        video = torch.arange(1, 13, dtype=dtype).reshape(1, 1, 3, 2, 2)
                        audio = torch.arange(1, 5, dtype=dtype).reshape(1, 1, 4)
                        inputs = {"input_latents": video, "audio_input_latents": audio if with_audio else None}
                        loss = self.run_loss(pipe, timestep_id, inputs)
                        sigma = pipe.scheduler.sigmas[timestep_id]
                        torch.testing.assert_close(pipe.model_inputs["video_latents"], (1 - sigma) * video, rtol=0, atol=0)
                        expected = self.expected_loss(pipe, timestep_id, video)
                        if with_audio:
                            torch.testing.assert_close(pipe.model_inputs["audio_latents"], (1 - sigma) * audio, rtol=0, atol=0)
                            expected = expected + self.expected_loss(pipe, timestep_id, audio)
                        torch.testing.assert_close(loss, expected)
                        self.assert_gradient(pipe, timestep_id, loss)

    def test_i2v_conditioning_frame_is_excluded(self):
        pipe = TinyPipeline(torch.bfloat16)
        clean = torch.arange(1, 13, dtype=pipe.torch_dtype).reshape(1, 1, 3, 2, 2)
        first_frame = torch.full_like(clean[:, :, :1], 32)
        loss = self.run_loss(pipe, 1, {"input_latents": clean, "first_frame_latents": first_frame})
        torch.testing.assert_close(pipe.model_inputs["latents"][:, :, :1], first_frame)
        torch.testing.assert_close(loss, self.expected_loss(pipe, 1, clean[:, :, 1:]))
        self.assert_gradient(pipe, 1, loss)


if __name__ == "__main__":
    unittest.main()
