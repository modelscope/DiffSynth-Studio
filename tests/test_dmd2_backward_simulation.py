import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from diffsynth.diffusion.dmd2 import DMD2Config, DMD2Loss, _ode_step


class _RecordingStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, latents, timestep):
        self.calls.append((timestep.detach().clone(), torch.is_grad_enabled()))
        return torch.ones_like(latents) * self.flow


def _model_fn(pipe, model, timestep, progress_id, num_inference_steps, inputs_shared, inputs_posi):
    return model(inputs_shared["latents"], timestep)


def _make_module():
    student = _RecordingStudent()
    pipe = SimpleNamespace(
        scheduler=SimpleNamespace(num_train_timesteps=1000),
        student=student,
    )
    module = SimpleNamespace(
        pipe=pipe,
        dmd2_student_model_name="student",
        dmd2_model_fn_student=_model_fn,
    )
    return module, student


class DMD2BackwardSimulationTests(unittest.TestCase):
    def _generate(self, real_data, sample_type="sde"):
        module, student = _make_module()
        loss = DMD2Loss(DMD2Config(student_sample_steps=3, student_sample_type=sample_type))
        with patch("diffsynth.diffusion.dmd2._sample_dmd2_student_step", return_value=2):
            gen_data, input_student = loss._generate_student_data(module, real_data, {}, {})
        return gen_data, input_student, student

    def test_multistep_rollout_does_not_use_real_sample_values(self):
        torch.manual_seed(123)
        generated_a, input_a, _ = self._generate(torch.zeros(2, 3))
        torch.manual_seed(123)
        generated_b, input_b, _ = self._generate(torch.full((2, 3), 100.0))

        torch.testing.assert_close(input_a, input_b)
        torch.testing.assert_close(generated_a, generated_b)

    def test_only_selected_rollout_step_tracks_gradients(self):
        torch.manual_seed(7)
        generated, input_student, student = self._generate(torch.zeros(2, 3))

        self.assertEqual(len(student.calls), 3)
        self.assertEqual([grad_enabled for _, grad_enabled in student.calls], [False, False, True])
        torch.testing.assert_close(
            torch.stack([timestep[0] for timestep, _ in student.calls]),
            torch.tensor([999.0, 666.0, 333.0], dtype=torch.float64),
        )
        self.assertFalse(input_student.requires_grad)
        self.assertTrue(generated.requires_grad)

        generated.sum().backward()
        self.assertIsNotNone(student.flow.grad)
        self.assertNotEqual(student.flow.grad.item(), 0.0)

    def test_ode_step_matches_flow_scheduler_update(self):
        latents = torch.tensor([[2.0]], dtype=torch.float32)
        x0 = torch.tensor([[1.0]], dtype=torch.float32)
        result = _ode_step(latents, x0, torch.tensor([0.8]), torch.tensor([0.4]))
        torch.testing.assert_close(result, torch.tensor([[1.5]], dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
